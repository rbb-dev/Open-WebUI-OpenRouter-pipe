"""Model catalog management subsystem.

This module handles synchronization of model metadata (capabilities, icons, descriptions)
from OpenRouter's frontend catalog to Open WebUI's model database.

Key responsibilities:
- Schedule metadata sync tasks on pipe invocation
- Fetch frontend catalog and extract icon/web-search mappings
- Download and convert profile images to data URLs
- Update or insert model records with metadata in OWUI database
- Auto-attach companion filters (Web Tools, Image Gen, Direct Uploads)
- Respect per-model advanced params (disable_model_metadata_sync, etc.)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeGuard
from urllib.parse import quote

import aiohttp

from ..core.logging_system import SessionLogger
from ..core.timing_logger import timed
from ..core.url_scheme import is_inline_data_url, url_scheme
from ..core.warn_latch import warn_level

try:
    from open_webui.models.models import ModelForm
except ImportError:
    ModelForm = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.models failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    ModelForm = None  # type: ignore

from ..core.config import (
    _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
    _OPENROUTER_FRONTEND_MODELS_URL,
    _OPENROUTER_MODEL_ENDPOINTS_URL_TEMPLATE,
    _OPENROUTER_SITE_URL,
    _OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
    _PIPE_METADATA_KEY,
    _PROVIDER_ROUTING_MAX_PROVIDERS,
    _PROVIDER_ROUTING_OVERLAY_MAX_MODELS,
    openrouter_attribution_headers,
)
from ..integrations.provider_options import options_key
from .registry import ModelFamily, OpenRouterModelRegistry, uses_dedicated_image_api

if TYPE_CHECKING:
    from ..pipe import Pipe


def _params_as_mapping(existing: Any) -> dict[str, Any]:
    if existing is None:
        return {}
    dump = getattr(existing, "model_dump", None)
    source: Any = dump() if callable(dump) else existing
    if not isinstance(source, Mapping):
        return {}
    return {str(key): value for key, value in source.items()}


def _params_without_tag_scanning(params_cls: Any, existing: Any) -> Any:
    data = _params_as_mapping(existing)
    data["reasoning_tags"] = data.get("reasoning_tags", False)
    return params_cls(**data)


def _dedupe_preserve_order(entries: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return deduped


def _normalize_id_list(meta_dict: dict, key: str) -> list[str]:
    current = meta_dict.get(key, [])
    if not isinstance(current, list):
        return []
    normalized: list[str] = []
    for entry in current:
        if isinstance(entry, str) and entry:
            normalized.append(entry)
    return normalized


_MAKER_SOURCE_KIND = "maker"
_FRONTEND_SOURCE_KIND = "frontend"
_APPLY_YIELD_EVERY = 64


def _frontend_catalog_answered(frontend_data: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(frontend_data, dict) and isinstance(frontend_data.get("data"), list)


def _ensure_pipe_meta(meta_dict: dict) -> dict:
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    if isinstance(pipe_meta, dict):
        return pipe_meta
    pipe_meta = {}
    meta_dict[_PIPE_METADATA_KEY] = pipe_meta
    return pipe_meta


def _stampable_icon_source(url: str | None) -> str | None:
    if not isinstance(url, str) or url_scheme(url) == "data":
        return None
    return url


def _icon_was_stored(meta_obj: Any) -> bool:
    dump = getattr(meta_obj, "model_dump", None)
    stored = dump() if callable(dump) else meta_obj
    if not isinstance(stored, dict):
        return True
    return bool(stored.get("profile_image_url"))


def _drop_unsaved_icon_source(meta_dict: dict[str, Any]) -> None:
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    if not isinstance(pipe_meta, dict):
        return
    pipe_meta.pop("image_source_url", None)
    pipe_meta.pop("image_source_kind", None)


def _covered_icon(row: Any, icon_url: str) -> str | None:
    if row is None:
        return None
    data_url, stamp = row[0], row[1]
    if data_url and stamp == icon_url:
        return data_url
    return None


def _convergent_stamp(row: tuple[str | None, str | None, bool, bool, str | None]) -> str:
    if row[1] and row[4] == _MAKER_SOURCE_KIND:
        return row[1]
    return ""


def makers_needing_a_page(
    models: list[dict[str, Any]],
    icon_mapping: dict[str, str],
    stored_icons: dict[str, tuple[str | None, str | None, bool, bool, str | None]],
    pipe_identifier: str,
    frontend_answered: bool = True,
) -> tuple[set[str], dict[str, str]]:
    if not frontend_answered:
        return set(), {}
    stamps: dict[str, set[str]] = {}
    for model in models:
        original_id = model.get("original_id")
        if not isinstance(original_id, str) or not original_id:
            continue
        if original_id in icon_mapping:
            continue
        maker_id = original_id.split("/", 1)[0]
        if not maker_id:
            continue
        row = stored_icons.get(f"{pipe_identifier}.{model.get('id')}")
        if row is not None and (row[2] or row[3]):
            continue
        stamps.setdefault(maker_id, set()).add(
            _convergent_stamp(row) if row is not None else ""
        )

    covered = {maker for maker, seen in stamps.items() if len(seen) == 1 and "" not in seen}
    missing = set(stamps) - covered
    seeded = {maker: seen.pop() for maker, seen in stamps.items() if maker in covered}
    return missing, seeded


def _warn_on_empty_read(
    ids: list[str],
    rows: Any,
    logger: Any,
) -> dict[str, tuple[str | None, str | None, bool, bool]]:
    if rows:
        return {}
    logger.warning(
        "Stored model icon read failed (returned no rows for %d ids); "
        "every icon will be re-fetched",
        len(ids),
    )
    return {}


_ROW_NOT_FETCHED = object()
_ROWS_UNREADABLE = object()

_warned_video_gen_filter_ensure: set[str] = set()

_MODEL_ROW_READ_CHUNK = 1000


class _ModelWriteRefused(Exception):
    pass


async def _read_model_rows(ids: list[str], logger: Any) -> Any:
    from open_webui.models.models import Models

    stored: dict[str, Any] = {}
    chunk = _MODEL_ROW_READ_CHUNK
    for start in range(0, len(ids), chunk):
        batch = ids[start : start + chunk]
        try:
            rows = await Models.get_models_by_ids(batch)
        except Exception as exc:
            logger.warning(
                "Stored model row read failed; every model will be read on its own: %s",
                exc,
                exc_info=True,
            )
            return None
        if not rows:
            try:
                await Models.get_all_models()
            except Exception as exc:
                logger.warning(
                    "Stored model row read could not be answered at all: the models table "
                    "itself did not read (%s); every model on this pass is left exactly as "
                    "it is, and nothing is written",
                    exc,
                    exc_info=True,
                )
                return _ROWS_UNREADABLE
            logger.debug(
                "Stored model row read returned no rows for %d ids; "
                "every model will be read on its own",
                len(batch),
            )
            return None
        for row in rows:
            model_id = getattr(row, "id", None)
            if not isinstance(model_id, str) or not model_id:
                continue
            stored[model_id] = row
    return stored if ids else None


async def _gather_in_chunks(
    apply_one: Callable[[Any], Awaitable[None]],
    items: list[Any],
    chunk: int,
) -> list[Any]:
    size = max(1, chunk)
    results: list[Any] = []
    for start in range(0, len(items), size):
        results.extend(
            await asyncio.gather(
                *(apply_one(item) for item in items[start : start + size]),
                return_exceptions=True,
            )
        )
    return results


def _stored_profile_images(
    ids: list[str],
    rows: dict[str, Any] | None,
    logger: Any,
) -> dict[str, tuple[str | None, str | None, bool, bool, str | None]]:
    from ..api.transforms import _get_disable_param

    if not ids:
        return {}
    if not rows:
        _warn_on_empty_read(ids, rows, logger)
        return {}

    stored: dict[str, tuple[str | None, str | None, bool, bool, str | None]] = {}
    for row in (rows or {}).values():
        model_id = getattr(row, "id", None)
        if not isinstance(model_id, str) or not model_id:
            continue
        meta = getattr(row, "meta", None)
        dump = getattr(meta, "model_dump", None)
        meta_dict = dump() if callable(dump) else meta
        if not isinstance(meta_dict, dict):
            meta_dict = {}
        pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
        if not isinstance(pipe_meta, dict):
            pipe_meta = {}
        data_url = meta_dict.get("profile_image_url")
        stamp = pipe_meta.get("image_source_url")
        kind = pipe_meta.get("image_source_kind")
        params = getattr(row, "params", None)
        stored[model_id] = (
            data_url if isinstance(data_url, str) and data_url else None,
            stamp if isinstance(stamp, str) and stamp else None,
            bool(_get_disable_param(params, "disable_image_updates")),
            bool(_get_disable_param(params, "disable_model_metadata_sync")),
            kind if isinstance(kind, str) and kind else None,
        )
    return stored


def _detach_record(
    pipe_meta: dict, record_key: str, keep_legacy: tuple[str, ...]
) -> bool:
    changed = record_key in pipe_meta
    pipe_meta.pop(record_key, None)
    for legacy_key in keep_legacy:
        if legacy_key in pipe_meta:
            pipe_meta.pop(legacy_key, None)
            changed = True
    return changed


def _record_ownership(
    meta_dict: dict,
    *,
    record_key: str,
    owned: str | list[str] | None,
    attaching: bool,
    keep_legacy: tuple[str, ...] = (),
) -> bool:
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    if attaching:
        pipe_meta = _ensure_pipe_meta(meta_dict)
        pipe_meta[record_key] = owned
        meta_dict[_PIPE_METADATA_KEY] = pipe_meta
        return False
    if not isinstance(pipe_meta, dict):
        return False
    return _detach_record(pipe_meta, record_key, keep_legacy)


def _write_settled_filter_ids(
    meta_dict: dict,
    normalized: list[str],
    *,
    wanted: set[str],
    attaching: bool,
    offered: list[str],
    record_key: str,
    owned: str | list[str] | None,
    keep_legacy: tuple[str, ...] = (),
) -> bool:
    if attaching:
        for fid in offered:
            if fid not in normalized:
                normalized.append(fid)
    normalized = [fid for fid in normalized if fid in wanted]
    meta_dict["filterIds"] = _dedupe_preserve_order(normalized)
    _record_ownership(
        meta_dict,
        record_key=record_key,
        owned=owned,
        attaching=attaching,
        keep_legacy=keep_legacy,
    )
    return True


def _record_needs_repair(previous, offered, *, attaching: bool) -> bool:
    if not attaching or not previous:
        return False
    if isinstance(previous, str):
        previous = [previous]
    return _dedupe_preserve_order(list(previous)) != _dedupe_preserve_order(list(offered))


def _apply_list_filter_ids(
    meta_dict: dict,
    *,
    filter_function_ids: list[str] | None,
    filter_supported: bool,
    auto_attach: bool,
    prune_key: str,
    hands_off: bool = False,
    retired_ids: frozenset[str] = frozenset(),
) -> bool:
    if hands_off:
        return False
    filter_function_ids = filter_function_ids or []
    normalized = _normalize_id_list(meta_dict, "filterIds")
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    previous_ids: list[str] = []
    if isinstance(pipe_meta, dict):
        prev = pipe_meta.get(prune_key)
        if isinstance(prev, list):
            previous_ids = [p for p in prev if isinstance(p, str) and p]
    current_set = set(filter_function_ids)
    had = set(normalized)
    wanted = set(had)
    attaching = filter_supported and auto_attach
    if attaching:
        wanted |= current_set
    for prev_fid in previous_ids:
        if not attaching or prev_fid not in current_set:
            wanted.discard(prev_fid)
    wanted -= retired_ids
    if wanted == had:
        if not attaching and _record_ownership(
            meta_dict, record_key=prune_key, owned=None, attaching=False
        ):
            return True
        if attaching and _record_needs_repair(
            previous_ids, filter_function_ids, attaching=attaching
        ):
            _record_ownership(
                meta_dict,
                record_key=prune_key,
                owned=list(filter_function_ids),
                attaching=True,
            )
            return True
        return False
    return _write_settled_filter_ids(
        meta_dict, normalized,
        wanted=wanted,
        attaching=attaching,
        offered=filter_function_ids,
        record_key=prune_key,
        owned=list(filter_function_ids),
    )


_LEGACY_RECORD_KEYS = {"web_tools_attached_id": "web_tools_filter_id"}

_SYNC_RETRY_FLOOR_SECONDS = 60.0

_ICON_SWEEP_BUDGET_SECONDS = 45


def _web_tools_owned(
    pipe_meta: dict,
    fid: str,
    previous_id_str: str,
    *,
    id_from_record: bool,
) -> bool:
    return bool(pipe_meta.get("web_tools_default_seeded")) or bool(previous_id_str)


def _web_tools_seeded_entry(pipe_meta: dict, fid: str) -> bool:
    return isinstance(pipe_meta.get("web_tools_seeded_id"), str) and pipe_meta["web_tools_seeded_id"] == fid


def _release_stale_seed(
    pipe_meta: dict,
    default_ids: list[str],
    *,
    previous_id_str: str,
    owned_id_str: str,
    owned_id: str | None,
    filter_function_id: str | None,
    seeded_key: str,
) -> tuple[list[str], bool]:
    if not (
        previous_id_str
        and _web_tools_seeded_entry(pipe_meta, previous_id_str)
        and previous_id_str in default_ids
    ):
        return default_ids, False
    remaining = [fid for fid in default_ids if fid != previous_id_str]
    pipe_meta[seeded_key] = False
    if filter_function_id and pipe_meta.get("web_tools_filter_id") == previous_id_str:
        pipe_meta["web_tools_filter_id"] = owned_id
    return remaining, True


def _single_id_is_transient(
    filter_function_id: str | None,
    supported: bool,
    auto_attach: bool,
    family_off: bool,
    blank_is_a_decision: bool,
) -> bool:
    blank_id_release = bool(
        not filter_function_id
        and (family_off or (blank_is_a_decision and (not supported or not auto_attach)))
    )
    return not filter_function_id and not blank_id_release


def _apply_single_id_filter_ids(
    meta_dict: dict,
    *,
    filter_function_id: str | None,
    supported: bool,
    auto_attach: bool,
    record_key: str,
    hands_off: bool = False,
    blank_is_a_decision: bool = False,
    family_off: bool = False,
) -> bool:
    if hands_off:
        return False
    if _single_id_is_transient(
        filter_function_id, supported, auto_attach, family_off, blank_is_a_decision
    ):
        return False
    offered_id = filter_function_id or ""
    normalized = _normalize_id_list(meta_dict, "filterIds")
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    owned_id = None
    if isinstance(pipe_meta, dict):
        prev = pipe_meta.get(record_key)
        if isinstance(prev, str) and prev:
            owned_id = prev
        else:
            legacy_key = _LEGACY_RECORD_KEYS.get(record_key)
            legacy = pipe_meta.get(legacy_key) if legacy_key else None
            if isinstance(legacy, str) and legacy:
                owned_id = legacy
    had = set(normalized)
    wanted = set(had)
    if owned_id:
        wanted.discard(owned_id)
    attaching = supported and auto_attach and bool(offered_id)
    if attaching:
        wanted.add(offered_id)
    elif not supported:
        wanted.discard(offered_id)
    legacy_key = _LEGACY_RECORD_KEYS.get(record_key)
    keep_legacy = (legacy_key,) if legacy_key else ()
    if wanted == had:
        if not attaching and _record_ownership(
            meta_dict, record_key=record_key, owned=None, attaching=False,
            keep_legacy=keep_legacy,
        ):
            return True
        if attaching and _record_needs_repair(owned_id, [offered_id], attaching=attaching):
            _record_ownership(
                meta_dict, record_key=record_key, owned=offered_id, attaching=True
            )
            return True
        return False
    return _write_settled_filter_ids(
        meta_dict, normalized,
        wanted=wanted,
        attaching=attaching,
        offered=[offered_id] if attaching else [],
        record_key=record_key,
        owned=offered_id,
        keep_legacy=() if attaching else keep_legacy,
    )


def _detached_by_this_pass(
    meta_dict: dict, *, prune_key: str, filter_function_ids: list[str] | None
) -> set[str]:
    """Ids this family attached last time and is not attaching now.

    Read before the attach pass, which rewrites the ownership record with the current
    ids -- so afterwards there is nothing left to compare against.
    """
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    previous: list[str] = []
    if isinstance(pipe_meta, dict):
        recorded = pipe_meta.get(prune_key)
        if isinstance(recorded, str) and recorded:
            previous = [recorded]
        elif isinstance(recorded, list):
            previous = [p for p in recorded if isinstance(p, str) and p]
    return set(previous) - set(filter_function_ids or [])


def _recorded_filter_id(meta_dict: dict, record_key: str) -> str:
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    if not isinstance(pipe_meta, dict):
        return ""
    recorded = pipe_meta.get(record_key)
    if isinstance(recorded, str) and recorded:
        return recorded
    return ""


def _apply_single_id_default_filter_ids(
    meta_dict: dict,
    *,
    owned_id: str,
    detached: set[str] | None = None,
    hands_off: bool = False,
) -> bool:
    if hands_off:
        return False
    if not owned_id or owned_id not in (detached or set()):
        return False
    default_ids = _normalize_id_list(meta_dict, "defaultFilterIds")
    kept = [fid for fid in default_ids if fid != owned_id]
    if len(kept) == len(default_ids):
        return False
    meta_dict["defaultFilterIds"] = _dedupe_preserve_order(kept)
    return True


def _recorded_ids_any_shape(meta_dict: dict, *, prune_key: str) -> set[str]:
    pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
    recorded: list[str] = []
    if isinstance(pipe_meta, dict):
        prev = pipe_meta.get(prune_key)
        if isinstance(prev, str) and prev:
            recorded = [prev]
        elif isinstance(prev, list):
            recorded = [p for p in prev if isinstance(p, str) and p]
    return set(recorded)


def _detached_with_default_off(
    meta_dict: dict,
    *,
    prune_key: str,
    filter_function_ids: list[str] | None,
    auto_default: bool,
) -> set[str]:
    detached = _detached_by_this_pass(
        meta_dict, prune_key=prune_key, filter_function_ids=filter_function_ids
    )
    if not auto_default:
        detached |= _recorded_ids_any_shape(meta_dict, prune_key=prune_key)
    return detached


def _apply_list_default_filter_ids(
    meta_dict: dict,
    *,
    filter_function_ids: list[str] | None,
    filter_supported: bool,
    auto_default: bool,
    detached: set[str] | None = None,
    hands_off: bool = False,
) -> bool:
    if hands_off:
        return False
    filter_ids = _normalize_id_list(meta_dict, "filterIds")
    default_ids = _normalize_id_list(meta_dict, "defaultFilterIds")
    changed = False

    # A filter this routine detached must not stay on by default, or a superseded one
    # goes on applying itself to every request. `detached` is computed by the caller
    # before the attach pass runs, because that pass rewrites the ownership record with
    # the current ids -- reading it here would find nothing to prune. Scoped rather than
    # blanket, because Open WebUI lets a *global* filter be default-on for a model
    # without ever appearing in filterIds, and those belong to other owners.
    kept = [fid for fid in default_ids if fid not in (detached or set())]
    if len(kept) != len(default_ids):
        default_ids = kept
        changed = True

    if auto_default and filter_supported:
        for fid in filter_function_ids or []:
            if fid in filter_ids and fid not in default_ids:
                default_ids.append(fid)
                changed = True

    if not changed:
        return False
    meta_dict["defaultFilterIds"] = _dedupe_preserve_order(default_ids)
    return True


def _apply_video_gen_filter_ids(
    meta_dict: dict,
    *,
    video_gen_filter_function_id: str | None,
    video_gen_filter_supported: bool,
    auto_attach_video_gen_filter: bool,
    hands_off: bool = False,
    family_off: bool = False,
) -> bool:
    """Apply video-gen filter auto-attach to `meta_dict["filterIds"]`.

    Single-id form, the same shape image now uses: one filter per model.
    """
    return _apply_single_id_filter_ids(
        meta_dict,
        filter_function_id=video_gen_filter_function_id,
        supported=video_gen_filter_supported,
        auto_attach=auto_attach_video_gen_filter,
        record_key="video_gen_filter_id",
        hands_off=hands_off,
        blank_is_a_decision=True,
        family_off=family_off,
    )


def _apply_video_default_filter_ids(
    meta_dict: dict,
    *,
    video_gen_filter_function_id: str | None,
    video_gen_filter_supported: bool,
    auto_default_video_gen_filter: bool,
    detached: set[str] | None = None,
    hands_off: bool = False,
) -> bool:
    """Apply video-gen filter default-on flag to `meta_dict["defaultFilterIds"]`."""
    if hands_off:
        return False
    default_ids = _normalize_id_list(meta_dict, "defaultFilterIds")
    changed = False

    kept = [fid for fid in default_ids if fid not in (detached or set())]
    if len(kept) != len(default_ids):
        default_ids = kept
        changed = True

    filter_ids = _normalize_id_list(meta_dict, "filterIds")
    if (auto_default_video_gen_filter and video_gen_filter_function_id
            and video_gen_filter_supported and video_gen_filter_function_id not in default_ids
            and video_gen_filter_function_id in filter_ids):
        default_ids.append(video_gen_filter_function_id)
        changed = True

    if not changed:
        return False
    meta_dict["defaultFilterIds"] = _dedupe_preserve_order(default_ids)
    return True


def _apply_provider_routing_default_filter_ids(
    meta_dict: dict,
    *,
    provider_routing_filter_id: str | None,
    auto_default_provider_routing_filter: bool,
    detached: set[str] | None = None,
    hands_off: bool = False,
) -> bool:
    """Apply provider routing filter default-on flag to `meta_dict["defaultFilterIds"]`."""
    if hands_off:
        return False
    filter_ids = _normalize_id_list(meta_dict, "filterIds")
    default_ids = _normalize_id_list(meta_dict, "defaultFilterIds")
    changed = False

    kept = [fid for fid in default_ids if fid not in (detached or set())]
    if len(kept) != len(default_ids):
        default_ids = kept
        changed = True

    if (
        auto_default_provider_routing_filter
        and provider_routing_filter_id
        and provider_routing_filter_id not in default_ids
        and provider_routing_filter_id in filter_ids
    ):
        default_ids.append(provider_routing_filter_id)
        changed = True

    if not changed:
        return False
    meta_dict["defaultFilterIds"] = _dedupe_preserve_order(default_ids)
    return True


def _answers_in_text(spec: Any) -> bool:
    if not isinstance(spec, dict):
        return False
    architecture = spec.get("architecture")
    if not isinstance(architecture, dict):
        return False
    modalities = architecture.get("output_modalities")
    return isinstance(modalities, list) and "text" in modalities


def media_capability_defaults(
    valves: Any,
    pipe_capabilities: dict[str, bool],
    answers_in_text: bool = False,
    rules_out_tool_use: bool = False,
) -> dict[str, Any]:
    """Capability defaults for a model, applied only where the model has no setting yet.

    A model that answers with an image or a clip cannot use a tool call, so Open WebUI's
    built-in tools are unticked rather than withheld at request time: the operator can see
    the box, and a box they tick themselves is left alone from then on.
    """
    if not getattr(valves, "UPDATE_MODEL_CAPABILITIES", False):
        return {}
    if not (
        pipe_capabilities.get("video_generation") or pipe_capabilities.get("image_output")
    ):
        return {}
    exempt = answers_in_text and not pipe_capabilities.get("video_generation")
    defaults: dict[str, Any] = {}
    if not exempt:
        defaults["file_context"] = False
    if getattr(valves, "DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS", False) and not (
        exempt and not rules_out_tool_use
    ):
        defaults["builtin_tools"] = False
    return defaults


def file_context_builtin_tool_defaults(capabilities: dict[str, Any]) -> dict[str, Any]:
    if capabilities.get("file_context") is not False:
        return {}
    return {"files": False}


def _merged_capabilities(
    base_caps: dict[str, Any] | None,
    capabilities: dict[str, Any] | None,
    capability_defaults: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base_caps) if isinstance(base_caps, dict) else {}
    for key, value in (capabilities or {}).items():
        merged[key] = value
    for key, value in (capability_defaults or {}).items():
        merged.setdefault(key, value)
    return merged


def needs_frontend_catalog(valves: Any, provider_routing_enabled: bool) -> bool:
    """Whether any enabled feature reads something only the frontend catalog carries.

    Two gates decide whether the fetch happens -- an early return in the sync routine and
    the fetch itself -- and a term added to one and not the other leaves the feature
    silently unserved. One predicate so they cannot drift.
    """
    return bool(
        provider_routing_enabled
        or valves.UPDATE_MODEL_IMAGES
        or valves.UPDATE_MODEL_CAPABILITIES
        or valves.UPDATE_MODEL_DESCRIPTIONS
        or valves.AUTO_ATTACH_WEB_TOOLS_FILTER
        or valves.ENABLE_VIDEO_GENERATION
        or valves.AUTO_INSTALL_VIDEO_FILTERS
        or valves.AUTO_INSTALL_IMAGE_FILTERS
        or valves.AUTO_INSTALL_IMAGE_GEN_FILTER
    )


def syncs_owui_models(valves: Any, provider_routing_enabled: bool) -> bool:
    return bool(
        valves.UPDATE_MODEL_CAPABILITIES
        or valves.UPDATE_MODEL_IMAGES
        or valves.UPDATE_MODEL_DESCRIPTIONS
        or valves.AUTO_ATTACH_WEB_TOOLS_FILTER
        or valves.AUTO_INSTALL_WEB_TOOLS_FILTER
        or valves.AUTO_DEFAULT_WEB_TOOLS_FILTER
        or valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER
        or valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER
        or valves.AUTO_INSTALL_IMAGE_GEN_FILTER
        or valves.AUTO_ATTACH_IMAGE_GEN_FILTER
        or valves.AUTO_INSTALL_VIDEO_FILTERS
        or valves.AUTO_ATTACH_VIDEO_FILTERS
        or valves.AUTO_INSTALL_IMAGE_FILTERS
        or valves.AUTO_ATTACH_IMAGE_FILTERS
        or valves.AUTO_INSTALL_FUSION_FILTER
        or valves.AUTO_ATTACH_FUSION_FILTER
        or valves.AUTO_DEFAULT_PROVIDER_ROUTING_FILTERS
        or provider_routing_enabled
    )


def schedules_owui_model_sync(valves: Any, provider_routing_enabled: bool) -> bool:
    return syncs_owui_models(valves, provider_routing_enabled)


WEB_TOOL_SWITCHES = (
    ("ENABLE_WEB_SEARCH", "WEB_SEARCH", "enable_web_search"),
    ("ENABLE_WEB_FETCH", "WEB_FETCH", "enable_web_fetch"),
    ("ENABLE_DATETIME", "DATETIME", "enable_datetime"),
    ("ENABLE_ADVISOR", "ADVISOR", "enable_advisor"),
    ("ENABLE_SUBAGENT", "SUBAGENT", "enable_subagent"),
    ("ENABLE_SEARCH_MODELS", "SEARCH_MODELS", "enable_search_models"),
)


def every_web_tool_is_off(valves: Any) -> bool:
    return not any(getattr(valves, switch) for switch, _, _ in WEB_TOOL_SWITCHES)


class ModelCatalogManager:
    """Manages model metadata synchronization from OpenRouter to Open WebUI."""

    def __init__(
        self,
        *,
        pipe: Pipe,
        multimodal_handler: Any,
        logger: logging.Logger,
        task_done_callback: Callable[[asyncio.Task], None] | None = None,
    ):
        self._pipe = pipe
        self._multimodal_handler = multimodal_handler
        self.logger = logger
        self._task_done_callback = task_done_callback

        # State for sync scheduling
        self._model_metadata_sync_task: asyncio.Task | None = None
        self._model_metadata_sync_key: tuple[Any, ...] | None = None
        self._model_metadata_sync_retry_after: float = 0.0

        self._cached_provider_map: dict[str, dict[str, Any]] = {}
        self._provider_overlay_failed_slugs: frozenset[str] = frozenset()
        self._provider_overlay_skipped_slugs: frozenset[str] = frozenset()

    def get_provider_overlay_skipped_slugs(self) -> frozenset[str]:
        return self._provider_overlay_skipped_slugs

    def get_cached_provider_map(self) -> dict[str, dict[str, Any]]:
        """Return the cached provider map from the last frontend catalog fetch.

        Returns a dict with structure:
        {
            "openai/gpt-4o": {
                "providers": ["openai", "azure", "together"],
                "quantizations": ["fp16", "bf16"],
                "short_name": "GPT-4o",
                "provider_names": {"openai": "OpenAI", "azure": "Azure", ...}
            },
            ...
        }

        This allows immediate access to provider data in pipes() without waiting
        for the background sync task. The map is populated by _sync_model_metadata_to_owui().
        """
        return self._cached_provider_map

    @staticmethod
    def _model_form_supports_access_control(model_form_cls: Any) -> bool:
        """Return True when ModelForm uses ``access_control`` rather than legacy grants."""
        fields = getattr(model_form_cls, "model_fields", None)
        return isinstance(fields, dict) and "access_control" in fields

    @staticmethod
    def _normalize_access_grants(value: Any) -> list[dict[str, Any]]:
        """Normalize legacy grant objects/dicts into plain dictionaries."""
        if not isinstance(value, list):
            return []
        normalized: list[dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, dict):
                normalized.append(dict(entry))
                continue
            model_dump = getattr(entry, "model_dump", None)
            if callable(model_dump):
                with contextlib.suppress(Exception):
                    dumped = model_dump()
                    if isinstance(dumped, dict):
                        normalized.append(dumped)
        return normalized

    @staticmethod
    def _legacy_grants_imply_public(grants: list[dict[str, Any]]) -> bool:
        """Detect wildcard read grants used by legacy public model overlays."""
        for grant in grants:
            principal_id = grant.get("principal_id")
            principal_type = grant.get("principal_type")
            permission = grant.get("permission")
            if (
                isinstance(principal_id, str)
                and principal_id == "*"
                and isinstance(principal_type, str)
                and principal_type.lower() == "user"
                and isinstance(permission, str)
                and permission.lower() == "read"
            ):
                return True
        return False

    @classmethod
    def _legacy_grants_to_access_control(cls, grants_value: Any) -> dict[str, Any] | None:
        """Convert legacy grants into OWUI ``access_control`` shape."""
        grants = cls._normalize_access_grants(grants_value)
        if cls._legacy_grants_imply_public(grants):
            return None

        access_control: dict[str, dict[str, list[str]]] = {}
        for grant in grants:
            principal_id = grant.get("principal_id")
            principal_type = grant.get("principal_type")
            permission = grant.get("permission")
            if not isinstance(principal_id, str) or not principal_id:
                continue
            if not isinstance(principal_type, str) or not isinstance(permission, str):
                continue

            principal_type_lower = principal_type.lower()
            permission_lower = permission.lower()
            if permission_lower not in {"read", "write"}:
                continue

            bucket = access_control.setdefault(
                permission_lower, {"user_ids": [], "group_ids": []}
            )
            if principal_type_lower == "user":
                target = bucket["user_ids"]
            elif principal_type_lower == "group":
                target = bucket["group_ids"]
            else:
                continue

            if principal_id not in target:
                target.append(principal_id)

        return access_control or {}

    def _resolve_model_access_payload(
        self,
        *,
        model_obj: Any,
        supports_access_control: bool,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Return access payload preserving existing model visibility semantics."""
        if supports_access_control:
            access_control = getattr(model_obj, "access_control", None)
            if isinstance(access_control, dict):
                return dict(access_control)
            return self._legacy_grants_to_access_control(
                getattr(model_obj, "access_grants", None)
            )

        return self._normalize_access_grants(getattr(model_obj, "access_grants", None))

    @staticmethod
    def _default_new_model_access_payload(
        *,
        access_mode: str,
        supports_access_control: bool,
    ) -> dict[str, Any] | list[dict[str, str]] | None:
        """Build default access payload for newly inserted overlays."""
        if supports_access_control:
            return None if access_mode == "public" else {}

        if access_mode == "public":
            return [
                {
                    "principal_type": "user",
                    "principal_id": "*",
                    "permission": "read",
                }
            ]
        return []

    @staticmethod
    def _build_model_form(
        *,
        model_form_cls: Any,
        supports_access_control: bool,
        id: str,
        base_model_id: str | None,
        name: str,
        meta: Any,
        params: Any,
        access_payload: dict[str, Any] | list[dict[str, Any]] | None,
        is_active: bool,
    ) -> Any:
        """Instantiate ModelForm with whichever access schema the runtime supports."""
        payload: dict[str, Any] = {
            "id": id,
            "base_model_id": base_model_id,
            "name": name,
            "meta": meta,
            "params": params,
            "is_active": is_active,
        }
        if supports_access_control:
            payload["access_control"] = (
                access_payload
                if access_payload is None or isinstance(access_payload, dict)
                else {}
            )
        else:
            payload["access_grants"] = (
                access_payload if isinstance(access_payload, list) else []
            )

        return model_form_cls(**payload)

    @timed
    def maybe_schedule_model_metadata_sync(
        self,
        selected_models: list[dict[str, Any]],
        *,
        pipe_identifier: str,
        image_gen_filter_model: str = "",
    ) -> None:
        """Schedule a background task to sync model metadata to OWUI if needed.

        Only schedules if:
        - At least one UPDATE_MODEL_* or AUTO_*_FILTER valve is enabled
        - selected_models is non-empty
        - Sync key has changed (valves, models, or registry state changed)
        - No sync task is already running

        Args:
            selected_models: List of model dicts from the pipe's model list
            pipe_identifier: The pipe's identifier (e.g., "openrouter")
        """
        valves = self._pipe.valves

        admin_routing_models = valves.ADMIN_PROVIDER_ROUTING_MODELS
        user_routing_models = valves.USER_PROVIDER_ROUTING_MODELS
        provider_routing_enabled = bool(admin_routing_models or user_routing_models)

        if not schedules_owui_model_sync(valves, provider_routing_enabled):
            return
        if not selected_models:
            return
        sync_key = (
            pipe_identifier,
            OpenRouterModelRegistry.content_stamp(),
            valves.MODEL_ID,
            valves.FREE_MODEL_FILTER,
            valves.TOOL_CALLING_FILTER,
            valves.ZDR_MODELS_ONLY,
            valves.VARIANT_MODELS,
            valves.UPDATE_MODEL_IMAGES,
            valves.UPDATE_MODEL_CAPABILITIES,
            valves.DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS,
            valves.UPDATE_MODEL_DESCRIPTIONS,
            valves.NEW_MODEL_ACCESS_CONTROL,
            valves.AUTO_ATTACH_WEB_TOOLS_FILTER,
            valves.AUTO_INSTALL_WEB_TOOLS_FILTER,
            valves.AUTO_DEFAULT_WEB_TOOLS_FILTER,
            valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER,
            valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER,
            valves.AUTO_INSTALL_IMAGE_GEN_FILTER,
            valves.AUTO_ATTACH_IMAGE_GEN_FILTER,
            valves.AUTO_INSTALL_VIDEO_FILTERS,
            valves.AUTO_ATTACH_VIDEO_FILTERS,
            valves.AUTO_DEFAULT_VIDEO_FILTERS,
            valves.ENABLE_VIDEO_GENERATION,
            valves.ENABLE_OPENROUTER_IMAGE_GENERATION,
            valves.AUTO_INSTALL_IMAGE_FILTERS,
            valves.AUTO_ATTACH_IMAGE_FILTERS,
            valves.AUTO_DEFAULT_IMAGE_FILTERS,
            valves.ENABLE_OPENROUTER_FUSION,
            valves.AUTO_INSTALL_FUSION_FILTER,
            valves.AUTO_ATTACH_FUSION_FILTER,
            valves.AUTO_DEFAULT_FUSION_FILTER,
            valves.ENABLE_WEB_SEARCH,
            valves.ENABLE_WEB_FETCH,
            valves.ENABLE_DATETIME,
            valves.ENABLE_ADVISOR,
            valves.ENABLE_SUBAGENT,
            valves.ENABLE_SEARCH_MODELS,
            valves.ENABLE_IMAGE_GENERATION,
            valves.VIDEO_INTENT_ENABLED,
            valves.VIDEO_INTENT_MAX_CLARIFICATIONS,
            valves.VIDEO_INTENT_FRAME_EXTRACTION_INDEX,
            valves.VIDEO_INTENT_CONFIRM_MODE,
            image_gen_filter_model,
            admin_routing_models,
            user_routing_models,
            valves.AUTO_DEFAULT_PROVIDER_ROUTING_FILTERS,
        )
        if sync_key == self._model_metadata_sync_key:
            return
        if time.monotonic() < self._model_metadata_sync_retry_after:
            return
        if self._model_metadata_sync_task and not self._model_metadata_sync_task.done():
            return

        models_copy = [dict(model) for model in selected_models]
        self._model_metadata_sync_key = sync_key
        self._model_metadata_sync_task = asyncio.create_task(
            self._sync_model_metadata_to_owui(
                models_copy,
                pipe_identifier=pipe_identifier,
            )
        )
        if self._task_done_callback:
            self._model_metadata_sync_task.add_done_callback(self._task_done_callback)
        self._model_metadata_sync_task.add_done_callback(self._on_model_metadata_sync_done)
        self._model_metadata_sync_task.add_done_callback(self._model_metadata_sync_floor_cleared)

    def _on_model_metadata_sync_done(self, task: asyncio.Task) -> None:
        if self._model_metadata_sync_task is not task and self._model_metadata_sync_task is not None:
            return
        if task.cancelled():
            self._model_metadata_sync_key = None
            return
        exc = task.exception()
        if exc is None:
            return
        try:
            self.logger.error(
                "Model metadata sync failed (%s); the key is released so the next "
                "model-list refresh retries it",
                exc,
                exc_info=exc,
            )
        except Exception:
            self.logger.exception("Model metadata sync failed and could not be logged")
        self._model_metadata_sync_key = None
        self._model_metadata_sync_retry_after = time.monotonic() + _SYNC_RETRY_FLOOR_SECONDS

    def _model_metadata_sync_floor_cleared(self, task: asyncio.Task) -> None:
        if task.cancelled() or self._model_metadata_sync_task is not task:
            return
        if task.exception() is None:
            self._model_metadata_sync_retry_after = 0.0

    @timed
    def _build_icon_mapping(self, frontend_data: dict[str, Any] | None) -> dict[str, str]:
        """Build a slug -> icon URL mapping from the frontend catalog."""
        if not _frontend_catalog_answered(frontend_data):
            return {}
        raw_items = frontend_data["data"]

        icon_mapping: dict[str, str] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug:
                continue
            if slug in icon_mapping:
                continue

            def _favicon_url(source_url: str) -> str | None:
                source_url = (source_url or "").strip()
                if not source_url.startswith(("http://", "https://")):
                    return None
                encoded = quote(source_url, safe="")
                return (
                    "https://t0.gstatic.com/faviconV2"
                    f"?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url={encoded}&size=256"
                )

            icon_url: str | None = None
            provider_hint_url: str | None = None

            endpoint = item.get("endpoint")
            if isinstance(endpoint, dict):
                provider_info = endpoint.get("provider_info")
                if isinstance(provider_info, dict):
                    icon_data = provider_info.get("icon")
                    if isinstance(icon_data, dict):
                        candidate = icon_data.get("url")
                        if isinstance(candidate, str):
                            icon_url = candidate
                    elif isinstance(icon_data, str):
                        icon_url = icon_data
                    candidate_urls: list[str] = []
                    base_url = provider_info.get("baseUrl")
                    if isinstance(base_url, str) and base_url.startswith(("http://", "https://")):
                        candidate_urls.append(base_url)
                    status_page_url = provider_info.get("statusPageUrl")
                    if isinstance(status_page_url, str) and status_page_url.startswith(("http://", "https://")):
                        candidate_urls.append(status_page_url)
                    data_policy = provider_info.get("dataPolicy")
                    if isinstance(data_policy, dict):
                        terms_url = data_policy.get("termsOfServiceURL")
                        if isinstance(terms_url, str) and terms_url.startswith(("http://", "https://")):
                            candidate_urls.append(terms_url)
                        privacy_url = data_policy.get("privacyPolicyURL")
                        if isinstance(privacy_url, str) and privacy_url.startswith(("http://", "https://")):
                            candidate_urls.append(privacy_url)
                    provider_hint_url = candidate_urls[0] if candidate_urls else None

            if not icon_url:
                icon_data = item.get("icon")
                if isinstance(icon_data, dict):
                    candidate = icon_data.get("url")
                    if isinstance(candidate, str):
                        icon_url = candidate
                elif isinstance(icon_data, str):
                    icon_url = icon_data

            icon_url = (icon_url or "").strip()
            if icon_url:
                if icon_url.startswith("//"):
                    icon_url = f"https:{icon_url}"
                elif icon_url.startswith("/"):
                    icon_url = f"{_OPENROUTER_SITE_URL}{icon_url}"
                elif not (icon_url.startswith(("http://", "https://")) or is_inline_data_url(icon_url)):
                    icon_url = f"{_OPENROUTER_SITE_URL}/{icon_url.lstrip('/')}"

            elif provider_hint_url:
                favicon = _favicon_url(provider_hint_url)
                if favicon:
                    icon_url = favicon
                else:
                    continue
            else:
                continue
            icon_mapping[slug] = icon_url

        return icon_mapping

    @timed
    def _build_model_provider_map(
        self,
        frontend_data: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        """Build a model slug -> provider info mapping from frontend catalog.

        Returns a dict with structure:
        {
            "openai/gpt-4o": {
                "providers": ["openai", "azure", "together"],
                "quantizations": ["fp16", "bf16"],
                "short_name": "GPT-4o",
                "provider_names": {"openai": "OpenAI", "azure": "Azure", ...}
            },
            ...
        }

        The frontend catalog returns one row per model with a single featured
        endpoint, so this map carries at most ONE provider per model. It is only
        the degraded fallback; the full provider list for routed models comes
        from the per-model endpoints API via _build_routed_provider_overlay.

        Note: Filters out variant-only endpoints (where model_variant_slug is set) to avoid
        advertising providers that only serve :free/:thinking variants. This prevents users
        from selecting providers that aren't available for the base model.
        """
        if not isinstance(frontend_data, dict):
            return {}
        raw_items = frontend_data.get("data")
        if not isinstance(raw_items, list):
            return {}

        model_providers: dict[str, set[str]] = {}
        model_quantizations: dict[str, set[str]] = {}
        model_short_names: dict[str, str] = {}
        model_provider_names: dict[str, dict[str, str]] = {}

        for item in raw_items:
            if not isinstance(item, dict):
                continue

            model_slug = item.get("slug")
            if not isinstance(model_slug, str) or not model_slug:
                continue

            if model_slug not in model_short_names:
                short_name = item.get("short_name")
                if isinstance(short_name, str) and short_name:
                    model_short_names[model_slug] = short_name

            endpoint = item.get("endpoint")
            if not isinstance(endpoint, dict):
                continue

            model_variant_slug = endpoint.get("model_variant_slug")
            if isinstance(model_variant_slug, str) and model_variant_slug and model_variant_slug != model_slug:
                continue

            provider_info = endpoint.get("provider_info")
            if isinstance(provider_info, dict):
                raw_provider_slug = provider_info.get("slug")
                provider_slug = (
                    raw_provider_slug.strip() if isinstance(raw_provider_slug, str) else ""
                )
                if provider_slug:
                    if model_slug not in model_providers:
                        model_providers[model_slug] = set()
                    model_providers[model_slug].add(provider_slug)

                    if model_slug not in model_provider_names:
                        model_provider_names[model_slug] = {}
                    if provider_slug not in model_provider_names[model_slug]:
                        display_name = provider_info.get("displayName") or provider_info.get("name")
                        if isinstance(display_name, str) and display_name:
                            model_provider_names[model_slug][provider_slug] = display_name

            # Extract quantization
            quantization = endpoint.get("quantization")
            if isinstance(quantization, str) and quantization:
                if model_slug not in model_quantizations:
                    model_quantizations[model_slug] = set()
                model_quantizations[model_slug].add(quantization)

        result: dict[str, dict[str, Any]] = {}
        all_slugs = set(model_providers.keys()) | set(model_quantizations.keys())

        for slug in all_slugs:
            result[slug] = {
                "providers": sorted(model_providers.get(slug, set())),
                "quantizations": sorted(model_quantizations.get(slug, set())),
                "short_name": model_short_names.get(slug, ""),
                "provider_names": model_provider_names.get(slug, {}),
            }

        return result

    @timed
    async def _fetch_frontend_model_catalog(
        self,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any] | None:
        """Fetch OpenRouter's public frontend model catalog (no auth)."""
        try:
            async with session.get(
                _OPENROUTER_FRONTEND_MODELS_URL,
                headers=openrouter_attribution_headers(self._pipe.valves),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as exc:
            self.logger.warning(
                "OpenRouter frontend catalog fetch failed: %s", exc, exc_info=True
            )
            return None

        if isinstance(payload, dict):
            return payload
        self.logger.warning(
            "OpenRouter frontend catalog returned invalid payload type '%s'; expected dict. Remote corruption or schema change detected.",
            type(payload).__name__,
        )
        return None

    async def _fetch_model_endpoints(
        self,
        session: aiohttp.ClientSession,
        model_slug: str,
    ) -> dict[str, Any] | None:
        """Fetch OpenRouter's public per-model endpoint list (no auth)."""
        url = _OPENROUTER_MODEL_ENDPOINTS_URL_TEMPLATE.format(slug=model_slug)
        try:
            async with session.get(
                url,
                headers=openrouter_attribution_headers(self._pipe.valves),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as exc:
            self.logger.debug(
                "OpenRouter endpoints fetch failed for %s: %s", model_slug, exc, exc_info=True
            )
            return None
        if isinstance(payload, dict):
            return payload
        self.logger.warning(
            "OpenRouter endpoints for %s returned invalid payload type '%s'; expected dict.",
            model_slug,
            type(payload).__name__,
        )
        return None

    @timed
    async def _build_routed_provider_overlay(
        self,
        session: aiohttp.ClientSession,
        model_slugs: Iterable[str],
    ) -> dict[str, dict[str, Any]]:
        """Build full provider info for routed models from the per-model endpoints API.

        The frontend catalog only carries one featured endpoint per model, so the
        provider-routing dropdowns would collapse to a single provider without this
        overlay. Only the models named in the routing valves are fetched.
        """
        unique = sorted({(s or "").strip() for s in model_slugs if (s or "").strip()})
        skipped: frozenset[str] = frozenset()
        if len(unique) > _PROVIDER_ROUTING_OVERLAY_MAX_MODELS:
            skipped = frozenset(unique[_PROVIDER_ROUTING_OVERLAY_MAX_MODELS:])
            named = ", ".join(sorted(skipped)[:5])
            if len(skipped) > 5:
                named = f"{named} and {len(skipped) - 5} more"
            self.logger.warning(
                "Provider routing valve lists %d models; only the first %d (sorted) get endpoint data. "
                "Skipped for that cap, so they keep the provider data from the previous cycle "
                "and their routing entry offers fewer providers: %s",
                len(unique),
                _PROVIDER_ROUTING_OVERLAY_MAX_MODELS,
                named,
            )
            unique = unique[:_PROVIDER_ROUTING_OVERLAY_MAX_MODELS]
        self._provider_overlay_skipped_slugs = skipped
        if not unique:
            return {}

        semaphore = asyncio.Semaphore(10)
        results: dict[str, dict[str, Any]] = {}

        async def _fetch_one(slug: str) -> None:
            async with semaphore:
                payload = await self._fetch_model_endpoints(session, slug)
            if payload is None:
                return
            data = payload.get("data")
            if not isinstance(data, dict):
                return
            endpoints = data.get("endpoints")
            if not isinstance(endpoints, list):
                return

            providers: set[str] = set()
            provider_names: dict[str, str] = {}
            quantizations: set[str] = set()
            for endpoint in endpoints:
                if not isinstance(endpoint, dict):
                    continue
                tag = endpoint.get("tag")
                if not isinstance(tag, str):
                    continue
                base_slug = options_key(tag)
                if not base_slug:
                    continue
                providers.add(base_slug)
                display = endpoint.get("provider_name")
                if base_slug not in provider_names and isinstance(display, str) and display.strip():
                    provider_names[base_slug] = display.strip()
                quantization = endpoint.get("quantization")
                if isinstance(quantization, str) and quantization.strip():
                    quantizations.add(quantization.strip())
            if not providers:
                return

            sorted_providers = sorted(providers)
            if len(sorted_providers) > _PROVIDER_ROUTING_MAX_PROVIDERS:
                self.logger.warning(
                    "Model %s reports %d providers; clamping to %d.",
                    slug,
                    len(sorted_providers),
                    _PROVIDER_ROUTING_MAX_PROVIDERS,
                )
                sorted_providers = sorted_providers[:_PROVIDER_ROUTING_MAX_PROVIDERS]
            short_name = data.get("name")
            results[slug] = {
                "providers": sorted_providers,
                "quantizations": sorted(quantizations)[:_PROVIDER_ROUTING_MAX_PROVIDERS],
                "short_name": short_name.strip() if isinstance(short_name, str) else "",
                "provider_names": {
                    s: provider_names[s] for s in sorted_providers if s in provider_names
                },
            }

        await asyncio.gather(*(_fetch_one(slug) for slug in unique), return_exceptions=True)
        return results

    def _merge_provider_overlay(
        self,
        frontend_map: dict[str, dict[str, Any]],
        overlay: dict[str, dict[str, Any]],
        routed_slugs: Iterable[str],
    ) -> dict[str, dict[str, Any]]:
        """Merge endpoint-API provider data over the frontend-derived fallback map.

        Overlay data replaces providers/quantizations for routed models; display
        names keep the frontend spelling when available so previously saved
        dropdown selections stay valid. On fetch failure the previous cycle's
        cached entry is retained when it is richer than the degraded frontend row.
        """
        merged = {slug: dict(entry) for slug, entry in frontend_map.items()}
        failed: list[str] = []
        for slug in sorted({(s or "").strip() for s in routed_slugs if (s or "").strip()}):
            enriched = overlay.get(slug)
            if enriched is None:
                fallback = merged.get(slug)
                cached = self._cached_provider_map.get(slug)
                fallback_count = len((fallback or {}).get("providers") or [])
                cached_count = len((cached or {}).get("providers") or [])
                if cached is not None and cached_count > fallback_count:
                    merged[slug] = dict(cached)
                if slug not in self._provider_overlay_skipped_slugs:
                    failed.append(slug)
                continue

            fallback = merged.get(slug) or {}
            fallback_names = fallback.get("provider_names")
            fallback_names = fallback_names if isinstance(fallback_names, dict) else {}
            overlay_names = enriched.get("provider_names") or {}
            providers = list(enriched.get("providers") or [])
            provider_names: dict[str, str] = {}
            for provider_slug in providers:
                frontend_display = fallback_names.get(provider_slug)
                if isinstance(frontend_display, str) and frontend_display:
                    provider_names[provider_slug] = frontend_display
                elif provider_slug in overlay_names:
                    provider_names[provider_slug] = overlay_names[provider_slug]
            short_name = fallback.get("short_name") or enriched.get("short_name") or ""
            merged[slug] = {
                "providers": providers,
                "quantizations": list(enriched.get("quantizations") or []),
                "short_name": short_name,
                "provider_names": provider_names,
            }

        failed_set = frozenset(failed)
        if failed:
            message = (
                "Provider endpoints data unavailable for %d routed model(s): %s "
                "(dropdowns fall back to degraded catalog data; check slug spelling)"
            )
            if failed_set != self._provider_overlay_failed_slugs:
                self.logger.warning(message, len(failed), ", ".join(failed))
            else:
                self.logger.debug(message, len(failed), ", ".join(failed))
        self._provider_overlay_failed_slugs = failed_set
        return merged

    async def _build_provider_map_with_overlay(
        self,
        session: aiohttp.ClientSession,
        frontend_data: dict[str, Any] | None,
        admin_models_csv: str,
        user_models_csv: str,
    ) -> dict[str, dict[str, Any]]:
        """Build the provider map: frontend fallback plus endpoints-API overlay."""
        frontend_map = self._build_model_provider_map(frontend_data)
        routed = [
            entry.strip()
            for entry in f"{admin_models_csv},{user_models_csv}".split(",")
            if entry.strip()
        ]
        if not routed:
            return frontend_map
        overlay = await self._build_routed_provider_overlay(session, routed)
        return self._merge_provider_overlay(frontend_map, overlay, routed)

    @timed
    async def _build_maker_profile_image_mapping(
        self,
        maker_ids: Iterable[str],
    ) -> dict[str, str]:
        unique = sorted({(m or "").strip() for m in maker_ids if (m or "").strip()})
        if not unique:
            return {}

        semaphore = asyncio.Semaphore(10)
        results: dict[str, str] = {}
        completed: set[str] = set()

        async def _fetch_maker_profile_image(maker_id: str) -> None:
            async with semaphore:
                image_url = await self._pipe._multimodal_handler._fetch_maker_profile_image_url(maker_id)
            if image_url:
                results[maker_id] = image_url
            completed.add(maker_id)

        try:
            async with asyncio.timeout(_ICON_SWEEP_BUDGET_SECONDS):
                await asyncio.gather(
                    *(_fetch_maker_profile_image(maker_id) for maker_id in unique),
                    return_exceptions=True,
                )
        except TimeoutError:
            abandoned = frozenset(m for m in unique if m not in completed)
            self.logger.warning(
                "The maker-profile sweep ran past %ds with %d of %d maker(s) still "
                "unfinished; those keep whatever icon they already had and the next pass "
                "retries.",
                _ICON_SWEEP_BUDGET_SECONDS,
                len(abandoned),
                len(unique),
            )
        return results

    @timed
    async def _sync_model_metadata_to_owui(
        self,
        models: list[dict[str, Any]],
        *,
        pipe_identifier: str,
    ) -> None:
        """Sync model metadata (capabilities, profile images, descriptions) into OWUI's Models table."""
        valves = self._pipe.valves

        admin_routing_models = valves.ADMIN_PROVIDER_ROUTING_MODELS
        user_routing_models = valves.USER_PROVIDER_ROUTING_MODELS
        provider_routing_enabled = bool(admin_routing_models or user_routing_models)

        if not syncs_owui_models(valves, provider_routing_enabled):
            return
        if not models:
            return
        if not pipe_identifier:
            return

        session = self._pipe._create_http_session()
        try:
            frontend_data = None
            if needs_frontend_catalog(valves, provider_routing_enabled):
                frontend_data = await self._fetch_frontend_model_catalog(session)

            if frontend_data is None:
                self.logger.debug("Frontend catalog fetch returned None")
            elif isinstance(frontend_data, dict):
                data_items = frontend_data.get("data")
                if isinstance(data_items, list):
                    self.logger.info(
                        "Frontend catalog fetched successfully: %d items in 'data' array",
                        len(data_items),
                    )
                else:
                    self.logger.warning(
                        "Frontend catalog fetched but 'data' key is not a list: %s",
                        type(data_items).__name__ if data_items is not None else "missing",
                    )
            else:
                self.logger.warning(
                    "Frontend catalog fetch returned unexpected type: %s",
                    type(frontend_data).__name__,
                )

            icon_mapping: dict[str, str] = {}
            if valves.UPDATE_MODEL_IMAGES:
                icon_mapping = self._build_icon_mapping(frontend_data)

            description_mapping: dict[str, str] = {}
            if valves.UPDATE_MODEL_DESCRIPTIONS and isinstance(frontend_data, dict):
                fd_items = frontend_data.get("data")
                if isinstance(fd_items, list):
                    for item in fd_items:
                        if not isinstance(item, dict):
                            continue
                        slug = item.get("slug")
                        desc = item.get("description")
                        if isinstance(slug, str) and slug and isinstance(desc, str) and desc.strip():
                            description_mapping[slug] = desc.strip()

            provider_map: dict[str, dict[str, Any]] = {}
            if provider_routing_enabled:
                provider_map = await self._build_provider_map_with_overlay(
                    session,
                    frontend_data,
                    admin_routing_models,
                    user_routing_models,
                )
            else:
                provider_map = self._build_model_provider_map(frontend_data)
            if not provider_map and self._cached_provider_map:
                self.logger.warning(
                    "Provider map rebuild returned 0 models; keeping the previous map of %d. "
                    "Provider routing options and video provider slugs are now stale.",
                    len(self._cached_provider_map),
                )
                provider_map = dict(self._cached_provider_map)
            else:
                known_ids = {str(m.get("original_id") or "").strip() for m in models} - {""}
                for slug, previous in self._cached_provider_map.items():
                    if slug in provider_map or slug not in known_ids:
                        continue
                    provider_map[slug] = dict(previous)
            self._cached_provider_map = provider_map
            self.logger.info(
                "Provider map built: %d models have provider info. Sample keys: %s",
                len(provider_map),
                list(provider_map.keys())[:5] if provider_map else "[]",
            )

            maker_mapping: dict[str, str] = {}
            stored_icons: dict[str, tuple[str | None, str | None, bool, bool, str | None]] = {}
            frontend_answered = _frontend_catalog_answered(frontend_data)
            model_row_ids = [
                f"{pipe_identifier}.{model['id']}"
                for model in models
                if isinstance(model.get("id"), str) and model["id"].strip()
            ]
            model_rows = await _read_model_rows(model_row_ids, self.logger)
            if valves.UPDATE_MODEL_IMAGES:
                stored_icons = _stored_profile_images(
                    model_row_ids,
                    None if model_rows is _ROWS_UNREADABLE else model_rows,
                    self.logger,
                )
                missing_makers, seeded_makers = makers_needing_a_page(
                    models, icon_mapping, stored_icons, pipe_identifier, frontend_answered
                )
                maker_mapping = dict(seeded_makers)
                if missing_makers:
                    maker_mapping.update(
                        await self._build_maker_profile_image_mapping(missing_makers)
                    )

            icon_data_mapping: dict[str, str] = {}
            maker_data_mapping: dict[str, str] = {}
            slug_to_icon_url: dict[str, str] = {}
            maker_to_image_url: dict[str, str] = {}
            if valves.UPDATE_MODEL_IMAGES:
                slug_to_icon_url = {}
                for model in models:
                    original_id = model.get("original_id")
                    if not isinstance(original_id, str) or not original_id:
                        continue
                    icon_url = icon_mapping.get(original_id)
                    if icon_url:
                        slug_to_icon_url[original_id] = icon_url

                maker_to_image_url = {k: v for k, v in maker_mapping.items() if isinstance(v, str) and v}

                stored_data_by_url: dict[str, str] = {}
                fetchable: dict[str, str] = {}
                needs_maker: set[str] = set()
                for model in models:
                    original_id = model.get("original_id")
                    if not isinstance(original_id, str) or not original_id:
                        continue
                    row = stored_icons.get(f"{pipe_identifier}.{model.get('id')}")
                    if row is not None and (row[2] or row[3]):
                        continue
                    if not frontend_answered and row is not None and row[4] == _FRONTEND_SOURCE_KIND and row[0]:
                        stored_data_by_url[original_id] = row[0]
                        continue
                    icon_url = slug_to_icon_url.get(original_id)
                    if icon_url:
                        covered = _covered_icon(row, icon_url)
                        if covered is not None:
                            stored_data_by_url[original_id] = covered
                            continue
                        fetchable[original_id] = icon_url
                        continue
                    maker_id = original_id.split("/", 1)[0]
                    if maker_id in maker_to_image_url:
                        covered = _covered_icon(row, maker_to_image_url[maker_id])
                        if covered is not None:
                            stored_data_by_url[original_id] = covered
                        else:
                            needs_maker.add(maker_id)

                for maker in needs_maker:
                    fetchable.setdefault(maker, maker_to_image_url[maker])

                unique_urls = sorted(set(fetchable.values()))
                url_to_data: dict[str, str] = {}
                if unique_urls:
                    fetch_semaphore = asyncio.Semaphore(10)
                    completed_urls: set[str] = set()

                    async def _fetch_image_data_url(url: str) -> None:
                        async with fetch_semaphore:
                            data_url = await self._pipe._multimodal_handler._fetch_image_as_data_url(url)
                        if data_url:
                            url_to_data[url] = data_url
                        completed_urls.add(url)

                    try:
                        async with asyncio.timeout(_ICON_SWEEP_BUDGET_SECONDS):
                            await asyncio.gather(
                                *(_fetch_image_data_url(url) for url in unique_urls),
                                return_exceptions=True,
                            )
                    except TimeoutError:
                        abandoned_urls = frozenset(u for u in unique_urls if u not in completed_urls)
                        self.logger.warning(
                            "The icon sweep ran past %ds with %d of %d read(s) still "
                            "unfinished; those keep whatever icon they already had and the "
                            "next pass retries.",
                            _ICON_SWEEP_BUDGET_SECONDS,
                            len(abandoned_urls),
                            len(unique_urls),
                        )

                icon_data_mapping = {
                    slug: data_url
                    for slug, data_url in stored_data_by_url.items()
                    if isinstance(slug, str) and "/" in slug and data_url
                }
                icon_data_mapping.update(
                    {
                        slug: url_to_data[url]
                        for slug, url in fetchable.items()
                        if isinstance(slug, str) and "/" in slug and url_to_data.get(url)
                    }
                )
                maker_data_mapping.update(
                    {
                        maker: url_to_data[url]
                        for maker, url in fetchable.items()
                        if isinstance(maker, str) and "/" not in maker and url_to_data.get(url)
                    }
                )

            semaphore = asyncio.Semaphore(10)
            from ..filters.filter_manager import (
                _REFUSED_FILTER_WRITES,
                a_filter_write_was_refused,
            )

            _REFUSED_FILTER_WRITES.clear()
            install_rows = await self._pipe._read_filter_rows()
            if install_rows is not None:
                install_rows = await self._pipe._with_active_rows(install_rows)
            web_valves = self._pipe.valves
            web_tools_filter_function_id: str | None = None
            web_tools_panel_withheld = every_web_tool_is_off(web_valves)
            web_tools_family_off = not (
                web_valves.AUTO_ATTACH_WEB_TOOLS_FILTER or web_valves.AUTO_INSTALL_WEB_TOOLS_FILTER
            )
            if (
                web_valves.AUTO_ATTACH_WEB_TOOLS_FILTER or web_valves.AUTO_INSTALL_WEB_TOOLS_FILTER
            ) and not web_tools_panel_withheld:
                try:
                    web_tools_filter_function_id = await self._pipe._ensure_filter_manager().ensure_openrouter_web_tools_filter_function_id(
                        enable_web_search=web_valves.ENABLE_WEB_SEARCH,
                        enable_web_fetch=web_valves.ENABLE_WEB_FETCH,
                        enable_datetime=web_valves.ENABLE_DATETIME,
                        enable_advisor=web_valves.ENABLE_ADVISOR,
                        enable_subagent=web_valves.ENABLE_SUBAGENT,
                        enable_search_models=web_valves.ENABLE_SEARCH_MODELS,
                        rows=install_rows,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "OpenRouter Web Tools filter ensure failed: %s", exc, exc_info=True
                    )
                    web_tools_filter_function_id = None
                    web_tools_family_off = False

            image_gen_filter_function_id: str | None = None
            image_gen_family_off = not (
                (valves.AUTO_INSTALL_IMAGE_GEN_FILTER or valves.AUTO_ATTACH_IMAGE_GEN_FILTER)
                and valves.ENABLE_IMAGE_GENERATION
            )
            if (
                valves.AUTO_INSTALL_IMAGE_GEN_FILTER or valves.AUTO_ATTACH_IMAGE_GEN_FILTER
            ) and valves.ENABLE_IMAGE_GENERATION:
                try:
                    image_gen_filter_function_id = await self._pipe._ensure_filter_manager().ensure_openrouter_image_gen_filter_function_id(rows=install_rows)
                except Exception as exc:
                    self.logger.warning(
                        "OpenRouter Image Gen filter ensure failed: %s", exc, exc_info=True
                    )
                    image_gen_filter_function_id = None
                    image_gen_family_off = False

            video_gen_filter_function_ids: dict[str, str] = {}
            video_family_off = not (
                (valves.AUTO_INSTALL_VIDEO_FILTERS or valves.AUTO_ATTACH_VIDEO_FILTERS)
                and valves.ENABLE_VIDEO_GENERATION
            )
            video_filter_ids_unresolved: frozenset[str] = frozenset()
            if (
                (valves.AUTO_INSTALL_VIDEO_FILTERS or valves.AUTO_ATTACH_VIDEO_FILTERS)
                and valves.ENABLE_VIDEO_GENERATION
            ):
                _video_filter_manager = self._pipe._ensure_filter_manager()
                try:
                    video_gen_filter_function_ids, video_filter_ids_unresolved = (
                        await _video_filter_manager.ensure_openrouter_video_gen_filter_function_ids(models, rows=install_rows)
                    )
                except Exception as exc:
                    self.logger.log(
                        warn_level(_warned_video_gen_filter_ensure, f"video_gen:{type(exc).__name__}"),
                        "OpenRouter Video Gen filter ensure failed: %s", exc, exc_info=True
                    )
                    video_gen_filter_function_ids = {}
                    video_family_off = False
            elif not valves.ENABLE_VIDEO_GENERATION:
                try:
                    await self._pipe._ensure_filter_manager()._retire_variant_video_filters(rows=install_rows)
                except Exception as exc:
                    self.logger.debug(
                        "Retiring superseded video filters failed: %s", exc, exc_info=True
                    )

            image_filter_function_ids: dict[str, list[str]] = {}
            image_filter_ids_known = True
            image_filter_ids_unresolved: frozenset[str] = frozenset()
            retired_image_filter_ids: frozenset[str] = frozenset()
            if (
                (valves.AUTO_INSTALL_IMAGE_FILTERS or valves.AUTO_ATTACH_IMAGE_FILTERS)
                and valves.ENABLE_OPENROUTER_IMAGE_GENERATION
            ):
                _image_filter_manager = self._pipe._ensure_filter_manager()
                try:
                    image_filter_function_ids, image_filter_ids_unresolved = (
                        await _image_filter_manager.ensure_openrouter_image_filter_function_ids(models, rows=install_rows)
                    )
                except Exception as exc:
                    self.logger.warning(
                        "OpenRouter Image filter ensure failed: %s", exc, exc_info=True
                    )
                    image_filter_ids_known = False
            else:
                # Retirement is not installation: it deactivates rows a previous design
                # left behind. It has to run with the valves off too, because that is
                # exactly the upgrade where nothing supersedes them -- the old ungated
                # filter would otherwise stay attached and keep writing its invented
                # values into every request.
                try:
                    retired_image_filter_ids = frozenset(
                        await self._pipe._ensure_filter_manager()._retire_variant_image_filters(rows=install_rows)
                        or ()
                    )
                except Exception as exc:
                    self.logger.debug(
                        "Retiring superseded image filters failed: %s", exc, exc_info=True
                    )

            fusion_filter_function_id: str | None = None
            fusion_filter_unresolved = False
            fusion_ids_known = True
            if valves.ENABLE_OPENROUTER_FUSION and (
                valves.AUTO_INSTALL_FUSION_FILTER or valves.AUTO_ATTACH_FUSION_FILTER
            ):
                _fusion_filter_manager = self._pipe._ensure_filter_manager()
                try:
                    fusion_filter_function_id, fusion_filter_unresolved = (
                        await _fusion_filter_manager.ensure_openrouter_fusion_filter_function_id(rows=install_rows)
                    )
                except Exception as exc:
                    self.logger.warning(
                        "OpenRouter Fusion filter ensure failed: %s", exc, exc_info=True
                    )
                    fusion_ids_known = False
                if fusion_filter_unresolved:
                    fusion_ids_known = False

            direct_uploads_filter_function_id: str | None = None
            direct_uploads_family_off = not (
                valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER
                or valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER
            )
            if (
                valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER
                or valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER
            ):
                try:
                    direct_uploads_filter_function_id = await self._pipe._ensure_filter_manager().ensure_direct_uploads_filter_function_id(rows=install_rows)
                except Exception as exc:
                    self.logger.warning(
                        "OpenRouter Direct Uploads filter ensure failed: %s", exc, exc_info=True
                    )
                    direct_uploads_filter_function_id = None
                    direct_uploads_family_off = False

            if valves.AUTO_ATTACH_WEB_TOOLS_FILTER and not every_web_tool_is_off(valves):
                if not web_tools_filter_function_id:
                    if a_filter_write_was_refused(_OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID):
                        if valves.AUTO_INSTALL_WEB_TOOLS_FILTER:
                            self.logger.warning(
                                "AUTO_ATTACH_WEB_TOOLS_FILTER is enabled but the OpenRouter Web Tools filter is "
                                "not installed: Open WebUI refused the write to %r that installs it. The install "
                                "valve is already doing its job, so this is a database fault rather than a "
                                "setting to change; the pipe retries on every model-list build.",
                                _OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
                            )
                        else:
                            self.logger.warning(
                                "AUTO_ATTACH_WEB_TOOLS_FILTER is enabled but the OpenRouter Web Tools filter is "
                                "not installed: Open WebUI refused a write to the %r row while "
                                "AUTO_INSTALL_WEB_TOOLS_FILTER is off, so the pipe is not installing it itself. "
                                "Switch that row on in Workspace > Functions, or turn the install valve on; the "
                                "pipe retries on every model-list build.",
                                _OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
                            )
                    else:
                        self.logger.warning(
                            "AUTO_ATTACH_WEB_TOOLS_FILTER is enabled but the OpenRouter Web Tools filter is not installed. "
                            "Enable AUTO_INSTALL_WEB_TOOLS_FILTER (or install the filter manually) to show the Web Tools toggle in the UI."
                        )
                else:
                    self.logger.info(
                        "Auto-attaching OpenRouter Web Tools filter '%s' to %d model(s).",
                        web_tools_filter_function_id,
                        len(models),
                    )

            if valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER:
                if not direct_uploads_filter_function_id:
                    if a_filter_write_was_refused(_DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID):
                        if valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER:
                            self.logger.warning(
                                "AUTO_ATTACH_DIRECT_UPLOADS_FILTER is enabled but the OpenRouter Direct Uploads filter is "
                                "not installed: Open WebUI refused the write to %r that installs it. The install "
                                "valve is already doing its job, so this is a database fault rather than a "
                                "setting to change; the pipe retries on every model-list build.",
                                _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
                            )
                        else:
                            self.logger.warning(
                                "AUTO_ATTACH_DIRECT_UPLOADS_FILTER is enabled but the OpenRouter Direct Uploads filter is "
                                "not installed: Open WebUI refused a write to the %r row while "
                                "AUTO_INSTALL_DIRECT_UPLOADS_FILTER is off, so the pipe is not installing it "
                                "itself. Switch that row on in Workspace > Functions, or turn the install valve "
                                "on; the pipe retries on every model-list build.",
                                _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
                            )
                    else:
                        self.logger.warning(
                            "AUTO_ATTACH_DIRECT_UPLOADS_FILTER is enabled but the OpenRouter Direct Uploads filter is not installed. "
                            "Enable AUTO_INSTALL_DIRECT_UPLOADS_FILTER (or install the filter manually) to show the toggle in the UI."
                        )
                else:
                    supported_models = 0
                    for model in models:
                        model_id = model.get("id")
                        if isinstance(model_id, str) and model_id:
                            try:
                                supported = bool(
                                    ModelFamily.supports("file_input", model_id)
                                    or ModelFamily.supports("audio_input", model_id)
                                    or ModelFamily.supports("video_input", model_id)
                                )
                            except (AttributeError, TypeError):
                                supported = False
                            if supported:
                                supported_models += 1
                    self.logger.info(
                        "Auto-attaching OpenRouter Direct Uploads filter '%s' to %d/%d model(s) that support direct uploads.",
                        direct_uploads_filter_function_id,
                        supported_models,
                        len(models),
                    )

            if valves.AUTO_ATTACH_VIDEO_FILTERS and valves.ENABLE_VIDEO_GENERATION:
                if not video_gen_filter_function_ids:
                    self.logger.warning(
                        "AUTO_ATTACH_VIDEO_FILTERS is enabled but no OpenRouter Video Generation filters are installed. "
                        "Enable AUTO_INSTALL_VIDEO_FILTERS (or install the filters manually) to show model-specific toggles in the UI."
                    )
                else:
                    supported_models = 0
                    for model in models:
                        model_id = model.get("id")
                        if isinstance(model_id, str) and model_id:
                            try:
                                supported = bool(ModelFamily.supports("video_generation", model_id))
                            except (AttributeError, TypeError):
                                supported = False
                            if supported:
                                supported_models += 1
                    self.logger.info(
                        "Auto-attaching %d OpenRouter Video Generation filter(s) to %d/%d video model(s).",
                        len(set(video_gen_filter_function_ids.values())),
                        supported_models,
                        len(models),
                    )

            # Provider routing filter generation
            provider_routing_filter_map: dict[str, str] = {}
            pr_ids_known = True
            if provider_routing_enabled:
                admin_list = [m.strip() for m in admin_routing_models.split(",") if m.strip()]
                user_list = [m.strip() for m in user_routing_models.split(",") if m.strip()]
                self.logger.info(
                    "Provider routing enabled: admin_models=%d, user_models=%d, provider_map_size=%d",
                    len(admin_list),
                    len(user_list),
                    len(provider_map),
                )
                if provider_map:
                    try:
                        provider_routing_filter_map = await self._pipe._ensure_filter_manager().ensure_provider_routing_filters(
                            admin_routing_models,
                            user_routing_models,
                            provider_map,
                            models,
                            pipe_identifier,
                            not_fetched_slugs=self._provider_overlay_skipped_slugs,
                        )
                        if not isinstance(provider_routing_filter_map, dict):
                            provider_routing_filter_map = {}
                    except Exception as exc:
                        pr_ids_known = False
                        self.logger.warning("Provider routing filter generation failed: %s", exc, exc_info=True)
                        provider_routing_filter_map = {}
                    pr_ids_known = pr_ids_known and bool(
                        self._pipe._ensure_filter_manager()._provider_routing_ids_known
                    )
                else:
                    pr_ids_known = False
                    self.logger.warning(
                        "Provider routing enabled but provider_map is empty (frontend catalog may have failed to load)"
                    )

                if provider_routing_filter_map:
                    self.logger.info(
                        "Auto-attaching provider routing filters to %d model(s): %s",
                        len(provider_routing_filter_map),
                        ", ".join(sorted(provider_routing_filter_map.keys())),
                    )

            _valid_openrouter_filter_ids: frozenset[str] = frozenset()
            try:
                from open_webui.models.functions import Functions as _FunctionsTable

                _all_filter_functions = await _FunctionsTable.get_functions_by_type("filter")
                _valid_openrouter_filter_ids = frozenset(
                    f.id for f in _all_filter_functions if f.id.startswith("openrouter_")
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to fetch valid filter IDs for stale pruning; stale openrouter_* "
                    "filter references will not be cleaned this cycle: %s",
                    exc,
                    exc_info=True,
                )

            sync_failures: list[str] = []

            async def _apply(model: dict[str, Any]) -> None:
                openrouter_id = model.get("id")
                name = model.get("name")
                if not isinstance(openrouter_id, str) or not openrouter_id.strip():
                    return
                if not isinstance(name, str) or not name:
                    name = openrouter_id

                openwebui_model_id = f"{pipe_identifier}.{openrouter_id}"

                def _safe_supports(feature: str) -> bool:
                    try:
                        return bool(ModelFamily.supports(feature, openrouter_id))
                    except (AttributeError, TypeError):
                        return False

                pipe_capabilities = {
                    "file_input": _safe_supports("file_input"),
                    "audio_input": _safe_supports("audio_input"),
                    "video_input": _safe_supports("video_input"),
                    "video_generation": _safe_supports("video_generation"),
                    "image_output": _safe_supports("image_output"),
                    "vision": _safe_supports("vision"),
                }

                capabilities = None
                media_spec = ModelFamily._lookup_spec(str(model.get("norm_id") or ""))
                capability_defaults = media_capability_defaults(
                    valves,
                    pipe_capabilities,
                    _answers_in_text(media_spec),
                    ModelFamily.rules_out_tool_use(str(model.get("norm_id") or "")),
                )
                if valves.UPDATE_MODEL_CAPABILITIES:
                    raw_caps = model.get("capabilities")
                    if isinstance(raw_caps, dict):
                        capabilities = dict(raw_caps)
                        if "web_search" in raw_caps:
                            capability_defaults["web_search"] = bool(raw_caps["web_search"])
                        capabilities.pop("web_search", None)
                        if "citations" in raw_caps:
                            capability_defaults["citations"] = bool(raw_caps["citations"])
                        capabilities.pop("citations", None)

                description = None
                if valves.UPDATE_MODEL_DESCRIPTIONS:
                    original_id_for_desc = model.get("original_id")
                    if isinstance(original_id_for_desc, str) and original_id_for_desc:
                        frontend_desc = description_mapping.get(original_id_for_desc)
                        if frontend_desc:
                            description = frontend_desc
                    if description is None:
                        norm_id = model.get("norm_id")
                        spec = ModelFamily._lookup_spec(str(norm_id or ""))
                        raw_desc = spec.get("description")
                        if isinstance(raw_desc, str):
                            cleaned = raw_desc.strip()
                            if cleaned:
                                description = cleaned

                original_id = model.get("original_id")

                profile_image_url = None
                image_source_url = None
                image_source_kind = None
                if (
                    valves.UPDATE_MODEL_IMAGES
                    and isinstance(original_id, str) and original_id
                ):
                    profile_image_url = icon_data_mapping.get(original_id)
                    if profile_image_url:
                        image_source_url = _stampable_icon_source(
                            slug_to_icon_url.get(original_id)
                        )
                    if not profile_image_url:
                        maker_id = original_id.split("/", 1)[0]
                        profile_image_url = maker_data_mapping.get(maker_id)
                        if profile_image_url:
                            image_source_url = _stampable_icon_source(
                                maker_to_image_url.get(maker_id)
                            )
                    if image_source_url:
                        image_source_kind = (
                            _FRONTEND_SOURCE_KIND
                            if image_source_url == slug_to_icon_url.get(original_id)
                            else _MAKER_SOURCE_KIND
                        )

                from ..filters.fusion_filter_renderer import (
                    is_fusion_model as _is_fusion,
                )
                norm_id = model.get("norm_id") or ""
                spec_lookup_id = (
                    (model.get("variant_base_norm_id") or norm_id.rsplit(":", 1)[0])
                    if model.get("variant_is_virtual")
                    else (norm_id or openrouter_id)
                )

                def _safe_rules_out(mid: str) -> bool:
                    try:
                        return bool(ModelFamily.rules_out_tool_use(mid))
                    except (AttributeError, TypeError):
                        return False

                tool_use_ruled_out = _safe_rules_out(spec_lookup_id)
                picture_only = uses_dedicated_image_api(ModelFamily._lookup_spec(str(norm_id or "")))
                fusion_model = bool(_is_fusion(openrouter_id) or _is_fusion(str(original_id or "")))
                image_gen_filter_supported = not (
                    tool_use_ruled_out
                    or picture_only
                    or pipe_capabilities.get("video_generation")
                    or fusion_model
                )
                if not image_gen_filter_supported and SessionLogger.debug_enabled(self.logger):
                    self.logger.debug(
                        "Image Generation switch withheld for %s: %s.",
                        openrouter_id,
                        "tool use is ruled out" if tool_use_ruled_out else (
                            "picture-only model" if picture_only else (
                                "video model" if pipe_capabilities.get("video_generation") else "fusion model"
                            )
                        ),
                    )
                web_tools_supported = bool(
                    web_tools_filter_function_id
                    and (
                        valves.AUTO_ATTACH_WEB_TOOLS_FILTER
                        or valves.AUTO_DEFAULT_WEB_TOOLS_FILTER
                    )
                    and not tool_use_ruled_out
                    and not picture_only
                    and not pipe_capabilities.get("video_generation")
                    and not _is_fusion(openrouter_id)
                )

                native_supported = bool(
                    pipe_capabilities.get("file_input")
                    or pipe_capabilities.get("audio_input")
                    or pipe_capabilities.get("video_input")
                )
                auto_attach_direct_uploads = bool(valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER)
                video_gen_filter_function_id = ""
                if pipe_capabilities.get("video_generation"):
                    video_gen_filter_function_id = (
                        video_gen_filter_function_ids.get(openrouter_id)
                        or video_gen_filter_function_ids.get(str(original_id or ""))
                        or ""
                    )
                auto_attach_video_gen = bool(
                    valves.AUTO_ATTACH_VIDEO_FILTERS
                    and valves.ENABLE_VIDEO_GENERATION
                    and pipe_capabilities.get("video_generation")
                )

                image_filter_ids_for_model: list[str] = []
                image_ids_unresolved = (
                    openrouter_id in image_filter_ids_unresolved
                    or str(original_id or "") in image_filter_ids_unresolved
                )
                video_ids_unresolved = (
                    openrouter_id in video_filter_ids_unresolved
                    or str(original_id or "") in video_filter_ids_unresolved
                )
                if pipe_capabilities.get("image_output"):
                    image_filter_ids_for_model = list(
                        image_filter_function_ids.get(openrouter_id)
                        or image_filter_function_ids.get(str(original_id or ""))
                        or []
                    )
                # Not gated on there being ids: a model that loses its filter still has
                # to have the old one detached, and that is this pass's job.
                auto_attach_image_filter = bool(
                    valves.AUTO_ATTACH_IMAGE_FILTERS
                    and valves.ENABLE_OPENROUTER_IMAGE_GENERATION
                )

                from ..filters.fusion_filter_renderer import is_fusion_model
                fusion_filter_ids_for_model: list[str] = []
                if fusion_filter_function_id and (
                    is_fusion_model(openrouter_id) or is_fusion_model(str(original_id or ""))
                ):
                    fusion_filter_ids_for_model = [fusion_filter_function_id]
                auto_attach_fusion = bool(
                    fusion_filter_ids_for_model
                    and valves.ENABLE_OPENROUTER_FUSION
                    and valves.AUTO_ATTACH_FUSION_FILTER
                )

                pr_filter_id = provider_routing_filter_map.get(original_id) if original_id else None

                if provider_routing_filter_map and SessionLogger.debug_enabled(self.logger):
                    self.logger.debug(
                        "PR lookup: original_id=%r, map_keys=%d, pr_filter_id=%r",
                        original_id,
                        len(provider_routing_filter_map),
                        pr_filter_id,
                    )

                async with semaphore:
                    try:
                        await self._update_or_insert_model_with_metadata(
                            openwebui_model_id,
                            name,
                            capabilities,
                            profile_image_url,
                            valves.UPDATE_MODEL_CAPABILITIES,
                            valves.UPDATE_MODEL_IMAGES,
                            image_source_url=image_source_url,
                            image_source_kind=image_source_kind,
                            capability_defaults=capability_defaults,
                            filter_function_id=web_tools_filter_function_id,
                            filter_supported=web_tools_supported,
                            auto_attach_filter=bool(
                                web_tools_supported and valves.AUTO_ATTACH_WEB_TOOLS_FILTER
                            ),
                            auto_default_filter=valves.AUTO_DEFAULT_WEB_TOOLS_FILTER,
                            web_tools_panel_withheld=web_tools_panel_withheld,
                            web_tools_family_off=web_tools_family_off,
                            direct_uploads_filter_function_id=direct_uploads_filter_function_id,
                            direct_uploads_filter_supported=native_supported,
                            auto_attach_direct_uploads_filter=auto_attach_direct_uploads,
                            direct_uploads_family_off=direct_uploads_family_off,
                            image_gen_filter_function_id=image_gen_filter_function_id,
                            image_gen_filter_supported=image_gen_filter_supported,
                            auto_attach_image_gen_filter=bool(valves.AUTO_ATTACH_IMAGE_GEN_FILTER),
                            image_gen_family_off=image_gen_family_off,
                            video_gen_filter_function_id=video_gen_filter_function_id,
                            video_gen_filter_supported=bool(pipe_capabilities.get("video_generation")),
                            auto_attach_video_gen_filter=auto_attach_video_gen,
                            video_ids_unresolved=video_ids_unresolved,
                            auto_default_video_gen_filter=bool(
                                auto_attach_video_gen
                                and valves.AUTO_DEFAULT_VIDEO_FILTERS
                            ),
                            video_family_off=video_family_off,
                            image_filter_function_ids=image_filter_ids_for_model,
                            image_filter_supported=bool(pipe_capabilities.get("image_output")),
                            auto_attach_image_filter=auto_attach_image_filter,
                            auto_default_image_filter=bool(
                                auto_attach_image_filter
                                and valves.AUTO_DEFAULT_IMAGE_FILTERS
                            ),
                            image_filter_ids_known=image_filter_ids_known,
                            image_ids_unresolved=image_ids_unresolved,
                            retired_image_filter_ids=retired_image_filter_ids,
                            fusion_filter_function_ids=fusion_filter_ids_for_model,
                            fusion_filter_supported=bool(fusion_filter_ids_for_model),
                            auto_attach_fusion_filter=auto_attach_fusion,
                            auto_default_fusion_filter=bool(
                                auto_attach_fusion
                                and valves.AUTO_DEFAULT_FUSION_FILTER
                            ),
                            fusion_ids_known=fusion_ids_known,
                            provider_routing_filter_id=pr_filter_id,
                            provider_routing_ids_known=pr_ids_known,
                            auto_default_provider_routing_filter=bool(
                                valves.AUTO_DEFAULT_PROVIDER_ROUTING_FILTERS
                            ),
                            valid_openrouter_filter_ids=_valid_openrouter_filter_ids,
                            openrouter_pipe_capabilities=pipe_capabilities,
                            description=description,
                            update_descriptions=valves.UPDATE_MODEL_DESCRIPTIONS,
                            new_model_access_control=valves.NEW_MODEL_ACCESS_CONTROL,
                            existing=(
                                _ROWS_UNREADABLE
                                if model_rows is _ROWS_UNREADABLE
                                else (
                                    model_rows.get(openwebui_model_id, _ROW_NOT_FETCHED)
                                    if model_rows is not None
                                    else _ROW_NOT_FETCHED
                                )
                            ),
                        )
                    except Exception as exc:
                        sync_failures.append(openwebui_model_id)
                        self.logger.debug(
                            "Model metadata sync failed (model=%s): %s",
                            openwebui_model_id,
                            exc,
                            exc_info=True,
                        )

            apply_results = await _gather_in_chunks(
                _apply, models, _APPLY_YIELD_EVERY
            )
            for model, outcome in zip(models, apply_results):
                if isinstance(outcome, BaseException):
                    model_ref = model.get("id") if isinstance(model, dict) else None
                    sync_failures.append(str(model_ref or "<unknown>"))
                    self.logger.debug(
                        "Model metadata apply failed before the sync call (model=%s)",
                        model_ref,
                        exc_info=outcome,
                    )
            if sync_failures:
                self.logger.warning(
                    "Model metadata sync failed for %d/%d model(s); their capabilities, "
                    "descriptions and filter attachments are unchanged. First failures: %s. "
                    "Enable DEBUG logging for per-model tracebacks.",
                    len(sync_failures),
                    len(models),
                    ", ".join(sync_failures[:5]),
                )
        finally:
            with contextlib.suppress(Exception):
                await session.close()

    async def prune_stale_openrouter_filter_ids(self) -> int:
        """Remove stale ``openrouter_*`` filter IDs from all model metadata.

        Queries the ``function`` table for existing ``openrouter_*`` filters,
        then sweeps the ``model`` table and removes any ``filterIds`` entries
        that match the ``openrouter_`` prefix but are not in the valid set.

        Returns the number of models that were updated.
        """
        from open_webui.models.functions import Functions as _FunctionsTable
        from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models

        supports_access_control = self._model_form_supports_access_control(ModelForm)

        all_filters = await _FunctionsTable.get_functions_by_type("filter")
        valid_ids = frozenset(
            f.id for f in all_filters if f.id.startswith("openrouter_")
        )

        updated = 0
        all_models = await Models.get_all_models()
        own_prefix = f"{self._pipe.id}." if getattr(self._pipe, "id", None) else None
        for model in all_models:
            meta = model.meta
            if not meta:
                continue
            if own_prefix is not None and not str(getattr(model, "id", "") or "").startswith(own_prefix):
                stored_ids = getattr(meta, "filterIds", None)
                if isinstance(stored_ids, list) and not any(
                    isinstance(fid, str) and fid.startswith("openrouter_") for fid in stored_ids
                ):
                    continue
            meta_dict = meta.model_dump()
            filter_ids = meta_dict.get("filterIds", [])
            if not isinstance(filter_ids, list) or not filter_ids:
                continue

            pruned = [
                fid for fid in filter_ids
                if not isinstance(fid, str)
                or not fid.startswith("openrouter_")
                or fid in valid_ids
            ]
            if len(pruned) == len(filter_ids):
                continue

            removed = set(filter_ids) - set(pruned)
            meta_dict["filterIds"] = pruned
            try:
                meta_obj = ModelMeta(**meta_dict)
                form = self._build_model_form(
                    model_form_cls=ModelForm,
                    supports_access_control=supports_access_control,
                    id=model.id,
                    base_model_id=model.base_model_id,
                    name=model.name,
                    meta=meta_obj,
                    params=_params_without_tag_scanning(ModelParams, model.params),
                    access_payload=self._resolve_model_access_payload(
                        model_obj=model,
                        supports_access_control=supports_access_control,
                    ),
                    is_active=model.is_active,
                )
                if await Models.update_model_by_id(model.id, form) is None:
                    raise _ModelWriteRefused(
                        f"Open WebUI did not report the write to {model.id} as landed"
                    )
            except Exception as exc:
                self.logger.warning(
                    "Startup prune: model '%s' keeps its stale filter IDs (%s): Open WebUI would not save it: %s",
                    model.id,
                    ", ".join(sorted(str(r) for r in removed)),
                    exc,
                    exc_info=True,
                )
                continue
            self.logger.warning(
                "Startup prune: removed stale filter IDs from model '%s': %s",
                model.id,
                ", ".join(sorted(str(r) for r in removed)),
            )
            updated += 1

        return updated

    @timed
    async def _update_or_insert_model_with_metadata(
        self,
        openwebui_model_id: str,
        name: str,
        capabilities: dict | None,
        profile_image_url: str | None,
        update_capabilities: bool,
        update_images: bool,
        *,
        capability_defaults: dict[str, Any] | None = None,
        filter_function_id: str | None = None,
        filter_supported: bool = False,
        auto_attach_filter: bool = False,
        auto_default_filter: bool = False,
        web_tools_panel_withheld: bool = False,
        web_tools_family_off: bool = False,
        direct_uploads_filter_function_id: str | None = None,
        direct_uploads_filter_supported: bool = False,
        auto_attach_direct_uploads_filter: bool = False,
        direct_uploads_family_off: bool = False,
        image_gen_filter_function_id: str | None = None,
        image_gen_filter_supported: bool = True,
        auto_attach_image_gen_filter: bool = False,
        image_gen_family_off: bool = False,
        video_gen_filter_function_id: str | None = None,
        video_gen_filter_supported: bool = False,
        auto_attach_video_gen_filter: bool = False,
        auto_default_video_gen_filter: bool = False,
        video_family_off: bool = False,
        video_ids_unresolved: bool = False,
        image_filter_function_ids: list[str] | None = None,
        image_filter_supported: bool = False,
        auto_attach_image_filter: bool = False,
        auto_default_image_filter: bool = False,
        image_filter_ids_known: bool = True,
        image_ids_unresolved: bool = False,
        retired_image_filter_ids: frozenset[str] = frozenset(),
        fusion_filter_function_ids: list[str] | None = None,
        fusion_filter_supported: bool = False,
        auto_attach_fusion_filter: bool = False,
        auto_default_fusion_filter: bool = False,
        fusion_ids_known: bool = True,
        provider_routing_filter_id: str | None = None,
        provider_routing_ids_known: bool = True,
        auto_default_provider_routing_filter: bool = False,
        valid_openrouter_filter_ids: frozenset[str] = frozenset(),
        openrouter_pipe_capabilities: dict[str, bool] | None = None,
        description: str | None = None,
        image_source_url: str | None = None,
        image_source_kind: str | None = None,
        update_descriptions: bool = False,
        new_model_access_control: str,
        existing: Any = _ROW_NOT_FETCHED,
    ):
        """Safely update existing model or insert new overlay with metadata, never touching owner."""
        from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models
        supports_access_control = self._model_form_supports_access_control(ModelForm)

        openwebui_model_id = (openwebui_model_id or "").strip()
        if not openwebui_model_id:
            return
        name = (name or "").strip() or openwebui_model_id

        if existing is _ROWS_UNREADABLE:
            raise _ModelWriteRefused(
                f"the models table could not be read, so {openwebui_model_id} is left "
                "exactly as it is"
            )
        if existing is _ROW_NOT_FETCHED:
            existing = await Models.get_model_by_id(openwebui_model_id)

        disable_model_metadata_sync = False
        disable_capability_updates = False
        disable_image_updates = False
        disable_web_tools_auto_attach = False
        disable_web_tools_default_on = False
        disable_direct_uploads_auto_attach = False
        disable_video_gen_auto_attach = False
        disable_image_filter_auto_attach = False
        disable_description_updates = False

        if existing is not None:
            from ..api.transforms import _get_disable_param

            params = getattr(existing, "params", None)
            disable_model_metadata_sync = _get_disable_param(params, "disable_model_metadata_sync")
            disable_capability_updates = _get_disable_param(params, "disable_capability_updates")
            disable_image_updates = _get_disable_param(params, "disable_image_updates")
            disable_web_tools_auto_attach = _get_disable_param(params, "disable_web_tools_auto_attach")
            disable_web_tools_default_on = _get_disable_param(params, "disable_web_tools_default_on")
            disable_direct_uploads_auto_attach = _get_disable_param(params, "disable_direct_uploads_auto_attach")
            disable_video_gen_auto_attach = _get_disable_param(params, "disable_video_gen_auto_attach")
            disable_image_filter_auto_attach = _get_disable_param(params, "disable_image_filter_auto_attach")
            disable_description_updates = _get_disable_param(params, "disable_description_updates")

        if disable_model_metadata_sync:
            return

        if disable_capability_updates:
            update_capabilities = False
        if disable_image_updates:
            update_images = False
        hands_off: set[str] = set()
        if disable_web_tools_auto_attach:
            auto_attach_filter = False
            hands_off.add("web_tools_attached_id")
        if disable_web_tools_default_on:
            auto_default_filter = False
        if disable_direct_uploads_auto_attach:
            auto_attach_direct_uploads_filter = False
            hands_off.add("direct_uploads_filter_id")
        if disable_video_gen_auto_attach:
            auto_attach_video_gen_filter = False
            hands_off.add("video_gen_filter_id")
        if disable_image_filter_auto_attach:
            auto_attach_image_filter = False
            hands_off.add("image_filter_ids")
        if disable_description_updates:
            update_descriptions = False

        id_from_record = False
        default_filter_id = filter_function_id
        if not filter_function_id and existing is not None and not disable_web_tools_default_on:
            recorded_pipe_meta = getattr(existing.meta, "model_dump", None)
            recorded_meta = recorded_pipe_meta() if callable(recorded_pipe_meta) else {}
            recorded_pipe_meta = (
                recorded_meta.get(_PIPE_METADATA_KEY)
                if isinstance(recorded_meta, dict)
                else None
            )
            if isinstance(recorded_pipe_meta, dict):
                for key in ("web_tools_attached_id", "web_tools_filter_id"):
                    candidate = recorded_pipe_meta.get(key)
                    if isinstance(candidate, str) and candidate:
                        default_filter_id = candidate
                        id_from_record = True
                        break

        def _ensure_pipe_meta(meta_dict: dict) -> dict:
            pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
            if isinstance(pipe_meta, dict):
                return pipe_meta
            pipe_meta = {}
            meta_dict[_PIPE_METADATA_KEY] = pipe_meta
            return pipe_meta

        def _normalize_id_list(meta_dict: dict, key: str) -> list[str]:
            current = meta_dict.get(key, [])
            if not isinstance(current, list):
                return []
            normalized: list[str] = []
            for entry in current:
                if isinstance(entry, str) and entry:
                    normalized.append(entry)
            return normalized

        def _dedupe_preserve_order(entries: list[str]) -> list[str]:
            seen: set[str] = set()
            deduped: list[str] = []
            for entry in entries:
                if entry in seen:
                    continue
                seen.add(entry)
                deduped.append(entry)
            return deduped

        def _prune_stale_openrouter_filter_ids(
            meta_dict: dict, keep_ids: frozenset[str] = frozenset()
        ) -> bool:
            """Remove openrouter_* filter IDs that no longer exist in the function table.

            Only prunes IDs with the ``openrouter_`` prefix — filter IDs belonging
            to other plugins are left untouched.  If *valid_openrouter_filter_ids*
            is empty (e.g. the batch query failed), pruning is skipped to avoid
            accidentally removing valid references.
            """
            if not valid_openrouter_filter_ids:
                return False
            normalized = _normalize_id_list(meta_dict, "filterIds")
            if not normalized:
                return False
            pruned = [
                fid for fid in normalized
                if not fid.startswith("openrouter_")
                or fid in valid_openrouter_filter_ids
                or fid in keep_ids
            ]
            if len(pruned) == len(normalized):
                return False
            removed = set(normalized) - set(pruned)
            self.logger.warning(
                "Pruned stale openrouter_* filter IDs from model '%s': %s",
                openwebui_model_id,
                ", ".join(sorted(removed)),
            )
            meta_dict["filterIds"] = pruned
            return True

        def _apply_filter_ids(meta_dict: dict) -> bool:
            return _apply_single_id_filter_ids(
                meta_dict,
                filter_function_id=filter_function_id,
                supported=filter_supported,
                auto_attach=auto_attach_filter,
                record_key="web_tools_attached_id",
                hands_off="web_tools_attached_id" in hands_off,
                family_off=web_tools_family_off,
            )

        def _apply_direct_uploads_filter_ids(meta_dict: dict) -> bool:
            return _apply_single_id_filter_ids(
                meta_dict,
                filter_function_id=direct_uploads_filter_function_id,
                supported=direct_uploads_filter_supported,
                auto_attach=auto_attach_direct_uploads_filter,
                record_key="direct_uploads_filter_id",
                hands_off="direct_uploads_filter_id" in hands_off,
                blank_is_a_decision=True,
                family_off=direct_uploads_family_off,
            )

        def _apply_image_gen_filter_ids(meta_dict: dict) -> bool:
            return _apply_single_id_filter_ids(
                meta_dict,
                filter_function_id=image_gen_filter_function_id,
                supported=image_gen_filter_supported,
                auto_attach=auto_attach_image_gen_filter,
                record_key="image_gen_filter_id",
                blank_is_a_decision=True,
                family_off=image_gen_family_off,
            )

        def _recorded_ids_kept_this_pass(meta_dict: dict) -> frozenset[str]:
            keep: set[str] = set()
            if video_ids_unresolved:
                keep.add(_recorded_filter_id(meta_dict, "video_gen_filter_id"))
            if image_ids_unresolved or not image_filter_ids_known:
                keep |= _recorded_ids_any_shape(meta_dict, prune_key="image_filter_ids")
            if not fusion_ids_known:
                keep |= _recorded_ids_any_shape(meta_dict, prune_key="fusion_filter_ids")
            if not provider_routing_ids_known:
                keep.add(_recorded_filter_id(meta_dict, "provider_routing_filter_id"))
            if _single_id_is_transient(
                filter_function_id, filter_supported, auto_attach_filter,
                web_tools_family_off, blank_is_a_decision=False,
            ):
                keep.add(_recorded_filter_id(meta_dict, "web_tools_attached_id"))
            if _single_id_is_transient(
                image_gen_filter_function_id, image_gen_filter_supported,
                auto_attach_image_gen_filter, image_gen_family_off,
                blank_is_a_decision=True,
            ):
                keep.add(_recorded_filter_id(meta_dict, "image_gen_filter_id"))
            if _single_id_is_transient(
                direct_uploads_filter_function_id, direct_uploads_filter_supported,
                auto_attach_direct_uploads_filter, direct_uploads_family_off,
                blank_is_a_decision=True,
            ):
                keep.add(_recorded_filter_id(meta_dict, "direct_uploads_filter_id"))
            return frozenset(keep)

        def _apply_default_filter_ids(
            meta_dict: dict,
            *,
            detached: set[str] | None = None,
            attach_detached: set[str] | None = None,
            hands_off: bool = False,
        ) -> bool:
            if hands_off:
                return False

            seeded_key = "web_tools_default_seeded"
            pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
            if not isinstance(pipe_meta, dict):
                pipe_meta = {}
            previous_id = pipe_meta.get("web_tools_filter_id")
            previous_id_str = previous_id if isinstance(previous_id, str) else ""
            recorded_id_str = previous_id_str
            owned_id = default_filter_id
            owned_id_str = (
                owned_id
                if (owned_id and _web_tools_owned(
                    pipe_meta, owned_id, previous_id_str, id_from_record=id_from_record
                ))
                else recorded_id_str
            )

            default_ids = _normalize_id_list(meta_dict, "defaultFilterIds")
            changed = False

            seeded_by_pipe = bool(pipe_meta.get(seeded_key, False))
            release = set(detached or ())
            blank_id_release = bool(
                id_from_record
                and not filter_function_id
                and (not auto_default_filter or web_tools_panel_withheld)
                and (
                    seeded_by_pipe
                    or pipe_meta.get("web_tools_seeded_id") == owned_id_str
                )
            )
            if owned_id_str and (
                blank_id_release
                or (
                    filter_function_id
                    and seeded_by_pipe
                    and (
                        owned_id in release
                        or filter_function_id in release
                        or (detached is not None and not auto_default_filter)
                        or (detached is not None and disable_web_tools_default_on)
                    )
                )
            ):
                release.add(owned_id_str)
            if previous_id_str and previous_id_str != owned_id_str and not auto_default_filter:
                stale_ids, released = _release_stale_seed(
                    pipe_meta,
                    default_ids,
                    previous_id_str=previous_id_str,
                    owned_id_str=owned_id_str,
                    owned_id=owned_id,
                    filter_function_id=filter_function_id,
                    seeded_key=seeded_key,
                )
                if released:
                    default_ids = stale_ids
                    changed = True
            kept = [fid for fid in default_ids if fid not in release or fid != owned_id_str]
            if len(kept) != len(default_ids):
                default_ids = kept
                changed = True
                pipe_meta = _ensure_pipe_meta(meta_dict)
                pipe_meta[seeded_key] = False
                pipe_meta.pop("web_tools_filter_id", None)

            seeding = bool(auto_default_filter and owned_id and filter_supported)
            if not seeding:
                superseded = [
                    fid
                    for fid in default_ids
                    if fid in (attach_detached or set())
                    and (
                        bool(pipe_meta.get(seeded_key, False))
                        or _web_tools_seeded_entry(pipe_meta, fid)
                    )
                ]
                if superseded:
                    default_ids = [fid for fid in default_ids if fid not in superseded]
                    changed = True
                    pipe_meta = _ensure_pipe_meta(meta_dict)
                    pipe_meta[seeded_key] = False

            if not auto_default_filter or not owned_id or not filter_supported:
                if changed:
                    meta_dict["defaultFilterIds"] = _dedupe_preserve_order(default_ids)
                    meta_dict[_PIPE_METADATA_KEY] = pipe_meta
                return changed

            filter_ids = _normalize_id_list(meta_dict, "filterIds")
            if owned_id not in filter_ids:
                if changed:
                    meta_dict["defaultFilterIds"] = _dedupe_preserve_order(default_ids)
                    meta_dict[_PIPE_METADATA_KEY] = pipe_meta
                return changed

            pipe_meta = _ensure_pipe_meta(meta_dict)
            if previous_id_str and previous_id_str != owned_id and previous_id_str in default_ids:
                default_ids = [owned_id if fid == previous_id_str else fid for fid in default_ids]
                changed = True

            if owned_id not in default_ids:
                if not seeded_by_pipe:
                    default_ids.append(owned_id)
                    pipe_meta[seeded_key] = True
                    pipe_meta["web_tools_seeded_id"] = owned_id
                    changed = True
            elif (
                pipe_meta.get("web_tools_seeded_id") == owned_id
                and not seeded_by_pipe
            ):
                pipe_meta[seeded_key] = True
                changed = True
            elif not seeded_by_pipe and pipe_meta.get(seeded_key) is not False:
                pipe_meta[seeded_key] = False
                changed = True

            if previous_id_str != owned_id and pipe_meta.get(seeded_key, False):
                pipe_meta["web_tools_filter_id"] = owned_id
                changed = True

            if not changed:
                return False

            meta_dict["defaultFilterIds"] = _dedupe_preserve_order(default_ids)
            meta_dict[_PIPE_METADATA_KEY] = pipe_meta
            return True

        def _apply_provider_routing_filter_ids(meta_dict: dict) -> bool:
            """Attach provider routing filter to model if configured."""
            if not provider_routing_ids_known:
                return False
            pr_debug = SessionLogger.debug_enabled(self.logger)
            if pr_debug:
                self.logger.debug(
                    "PR attach attempt: model=%r, filter_id=%r",
                    openwebui_model_id,
                    provider_routing_filter_id,
                )

            normalized = _normalize_id_list(meta_dict, "filterIds")
            pipe_meta = meta_dict.get(_PIPE_METADATA_KEY)
            previous_id = None
            recorded_id = None
            if isinstance(pipe_meta, dict):
                prev = pipe_meta.get("provider_routing_filter_id")
                if isinstance(prev, str) and prev:
                    recorded_id = prev
                    if prev != provider_routing_filter_id:
                        previous_id = prev

            if not provider_routing_filter_id:
                if recorded_id and recorded_id in normalized:
                    if pr_debug:
                        self.logger.debug(
                            "PR retire: detaching '%s' from model '%s'",
                            recorded_id,
                            openwebui_model_id,
                        )
                    meta_dict["filterIds"] = _dedupe_preserve_order(
                        [fid for fid in normalized if fid != recorded_id]
                    )
                    pipe_meta = _ensure_pipe_meta(meta_dict)
                    pipe_meta.pop("provider_routing_filter_id", None)
                    meta_dict[_PIPE_METADATA_KEY] = pipe_meta
                    return True
                if pr_debug:
                    self.logger.debug("PR attach: filter_id is None/empty, skipping")
                return False

            had = set(normalized)
            wanted = set(had)
            wanted.add(provider_routing_filter_id)
            if previous_id:
                wanted.discard(previous_id)

            if pr_debug:
                self.logger.debug(
                    "PR attach: current_filterIds=%r, had=%r, wanted=%r, previous_id=%r",
                    normalized,
                    had,
                    wanted,
                    previous_id,
                )

            if wanted == had:
                if _record_needs_repair(
                    recorded_id, [provider_routing_filter_id], attaching=True
                ):
                    pipe_meta = _ensure_pipe_meta(meta_dict)
                    pipe_meta["provider_routing_filter_id"] = provider_routing_filter_id
                    meta_dict[_PIPE_METADATA_KEY] = pipe_meta
                    return True
                if pr_debug:
                    self.logger.debug("PR attach: wanted==had, no change needed")
                return False

            if provider_routing_filter_id not in normalized:
                normalized.append(provider_routing_filter_id)
            normalized = [fid for fid in normalized if fid in wanted]
            meta_dict["filterIds"] = _dedupe_preserve_order(normalized)
            pipe_meta = _ensure_pipe_meta(meta_dict)
            pipe_meta["provider_routing_filter_id"] = provider_routing_filter_id
            meta_dict[_PIPE_METADATA_KEY] = pipe_meta

            if pr_debug:
                self.logger.debug(
                    "PR attach: SUCCESS - new filterIds=%r",
                    meta_dict["filterIds"],
                )
            return True

        if existing:
            meta_dict = {}
            if existing.meta:
                meta_dict.update(existing.meta.model_dump())

            meta_updated = False

            if update_capabilities and (capabilities is not None or capability_defaults):
                existing_caps = meta_dict.get("capabilities")
                merged_caps = _merged_capabilities(existing_caps, capabilities, capability_defaults)
                if merged_caps != existing_caps:
                    meta_dict["capabilities"] = merged_caps
                    meta_updated = True

                file_context_tool_defaults = file_context_builtin_tool_defaults(merged_caps)
                if file_context_tool_defaults:
                    existing_builtin = meta_dict.get("builtinTools")
                    merged_builtin: dict[str, Any] = (
                        dict(existing_builtin) if isinstance(existing_builtin, dict) else {}
                    )
                    for key, value in file_context_tool_defaults.items():
                        merged_builtin.setdefault(key, value)
                    if merged_builtin != existing_builtin:
                        meta_dict["builtinTools"] = merged_builtin
                        meta_updated = True

            if (
                update_images and profile_image_url
                and meta_dict.get("profile_image_url") != profile_image_url
            ):
                meta_dict["profile_image_url"] = profile_image_url
                meta_updated = True

            if update_images and image_source_url:
                pipe_meta = _ensure_pipe_meta(meta_dict)
                if pipe_meta.get("image_source_url") != image_source_url:
                    pipe_meta["image_source_url"] = image_source_url
                    meta_updated = True
                if image_source_kind and pipe_meta.get("image_source_kind") != image_source_kind:
                    pipe_meta["image_source_kind"] = image_source_kind
                    meta_updated = True

            if (
                update_descriptions and description
                and meta_dict.get("description") != description
            ):
                meta_dict["description"] = description
                meta_updated = True

            if _prune_stale_openrouter_filter_ids(
                meta_dict, _recorded_ids_kept_this_pass(meta_dict)
            ):
                meta_updated = True

            web_tools_hands_off = "web_tools_attached_id" in hands_off
            web_tools_ids_now = [filter_function_id] if (
                filter_supported and auto_attach_filter and filter_function_id
            ) else []
            web_tools_attach_detached = _detached_by_this_pass(
                meta_dict, prune_key="web_tools_attached_id",
                filter_function_ids=web_tools_ids_now,
            )
            web_tools_detached = web_tools_attach_detached | _detached_by_this_pass(
                meta_dict, prune_key="web_tools_filter_id",
                filter_function_ids=web_tools_ids_now,
            )
            if not filter_function_id:
                web_tools_attach_detached = set()
                web_tools_detached = set()

            if _apply_filter_ids(meta_dict):
                meta_updated = True

            if _apply_default_filter_ids(
                meta_dict, detached=web_tools_detached,
                attach_detached=web_tools_attach_detached,
                hands_off=web_tools_hands_off,
            ):
                meta_updated = True

            direct_uploads_detached_id = _recorded_filter_id(meta_dict, "direct_uploads_filter_id")
            direct_uploads_ids_now = [direct_uploads_filter_function_id] if (
                direct_uploads_filter_supported and auto_attach_direct_uploads_filter
                and direct_uploads_filter_function_id
            ) else []
            direct_uploads_detached = _detached_by_this_pass(
                meta_dict, prune_key="direct_uploads_filter_id",
                filter_function_ids=direct_uploads_ids_now,
            )
            if direct_uploads_filter_supported and not direct_uploads_filter_function_id and not direct_uploads_family_off and auto_attach_direct_uploads_filter:
                direct_uploads_detached = set()
            if _apply_direct_uploads_filter_ids(meta_dict):
                meta_updated = True

            if _apply_single_id_default_filter_ids(
                meta_dict, owned_id=direct_uploads_detached_id, detached=direct_uploads_detached,
                hands_off="direct_uploads_filter_id" in hands_off,
            ):
                meta_updated = True

            image_gen_detached_id = _recorded_filter_id(meta_dict, "image_gen_filter_id")
            image_gen_ids_now = [image_gen_filter_function_id] if (
                image_gen_filter_supported and auto_attach_image_gen_filter
                and image_gen_filter_function_id
            ) else []
            image_gen_detached = _detached_by_this_pass(
                meta_dict, prune_key="image_gen_filter_id",
                filter_function_ids=image_gen_ids_now,
            )
            if image_gen_filter_supported and not image_gen_filter_function_id and not image_gen_family_off and auto_attach_image_gen_filter:
                image_gen_detached = set()
            if _apply_image_gen_filter_ids(meta_dict):
                meta_updated = True

            if _apply_single_id_default_filter_ids(
                meta_dict, owned_id=image_gen_detached_id, detached=image_gen_detached,
                hands_off="image_gen_filter_id" in hands_off,
            ):
                meta_updated = True

            video_hands_off = "video_gen_filter_id" in hands_off
            video_ids_now = [video_gen_filter_function_id] if (
                video_gen_filter_supported and auto_attach_video_gen_filter and video_gen_filter_function_id
            ) else []
            video_detached = _detached_with_default_off(
                meta_dict,
                prune_key="video_gen_filter_id",
                filter_function_ids=video_ids_now,
                auto_default=auto_default_video_gen_filter,
            )
            if not video_gen_filter_function_id and video_gen_filter_supported and not video_family_off and auto_attach_video_gen_filter:
                video_detached = set()
            if _apply_video_gen_filter_ids(
                meta_dict,
                video_gen_filter_function_id=video_gen_filter_function_id,
                video_gen_filter_supported=video_gen_filter_supported,
                auto_attach_video_gen_filter=auto_attach_video_gen_filter,
                hands_off=video_hands_off,
                family_off=video_family_off,
            ):
                meta_updated = True

            if _apply_video_default_filter_ids(
                meta_dict,
                video_gen_filter_function_id=video_gen_filter_function_id,
                video_gen_filter_supported=video_gen_filter_supported,
                auto_default_video_gen_filter=auto_default_video_gen_filter,
                detached=video_detached,
                hands_off=video_hands_off,
            ):
                meta_updated = True

            image_hands_off = (
                "image_filter_ids" in hands_off
                or not image_filter_ids_known
                or image_ids_unresolved
            )
            image_ids_now = image_filter_function_ids if (
                image_filter_supported and auto_attach_image_filter
            ) else []
            image_detached = _detached_with_default_off(
                meta_dict,
                prune_key="image_filter_ids",
                filter_function_ids=image_ids_now,
                auto_default=auto_default_image_filter,
            )
            if _apply_list_filter_ids(
                meta_dict,
                filter_function_ids=image_filter_function_ids,
                filter_supported=image_filter_supported,
                auto_attach=auto_attach_image_filter,
                prune_key="image_filter_ids",
                hands_off=image_hands_off,
                retired_ids=retired_image_filter_ids,
            ):
                meta_updated = True

            if _apply_list_default_filter_ids(
                meta_dict,
                detached=image_detached,
                filter_function_ids=image_filter_function_ids,
                filter_supported=image_filter_supported,
                auto_default=auto_default_image_filter,
                hands_off=image_hands_off,
            ):
                meta_updated = True

            fusion_hands_off = "fusion_filter_ids" in hands_off or not fusion_ids_known
            fusion_ids_now = fusion_filter_function_ids if (
                fusion_filter_supported and auto_attach_fusion_filter
            ) else []
            fusion_detached = _detached_with_default_off(
                meta_dict,
                prune_key="fusion_filter_ids",
                filter_function_ids=fusion_ids_now,
                auto_default=auto_default_fusion_filter,
            )
            if _apply_list_filter_ids(
                meta_dict,
                filter_function_ids=fusion_filter_function_ids,
                filter_supported=fusion_filter_supported,
                auto_attach=auto_attach_fusion_filter,
                prune_key="fusion_filter_ids",
                hands_off=fusion_hands_off,
            ):
                meta_updated = True

            if _apply_list_default_filter_ids(
                meta_dict,
                detached=fusion_detached,
                filter_function_ids=fusion_filter_function_ids,
                filter_supported=fusion_filter_supported,
                auto_default=auto_default_fusion_filter,
                hands_off=fusion_hands_off,
            ):
                meta_updated = True

            pr_hands_off = not provider_routing_ids_known
            pr_ids_now = [provider_routing_filter_id] if provider_routing_filter_id else []
            pr_detached = set() if pr_hands_off else _detached_with_default_off(
                meta_dict,
                prune_key="provider_routing_filter_id",
                filter_function_ids=pr_ids_now,
                auto_default=auto_default_provider_routing_filter,
            )
            if _apply_provider_routing_filter_ids(meta_dict):
                meta_updated = True
                self.logger.debug(
                    "Attached provider routing filter '%s' to model '%s'",
                    provider_routing_filter_id,
                    openwebui_model_id,
                )
            if _apply_provider_routing_default_filter_ids(
                meta_dict,
                detached=pr_detached,
                provider_routing_filter_id=provider_routing_filter_id,
                auto_default_provider_routing_filter=auto_default_provider_routing_filter,
                hands_off=pr_hands_off,
            ):
                meta_updated = True

            if openrouter_pipe_capabilities is not None:
                pipe_meta = _ensure_pipe_meta(meta_dict)
                if pipe_meta.get("capabilities") != openrouter_pipe_capabilities:
                    pipe_meta["capabilities"] = dict(openrouter_pipe_capabilities)
                    meta_dict[_PIPE_METADATA_KEY] = pipe_meta
                    meta_updated = True

            existing_tags = _params_as_mapping(
                getattr(existing, "params", None)
            ).get("reasoning_tags")
            if not meta_updated and existing_tags is False:
                return

            meta_obj = ModelMeta(**meta_dict)
            if meta_dict.get("profile_image_url") and not _icon_was_stored(meta_obj):
                _drop_unsaved_icon_source(meta_dict)
                meta_obj = ModelMeta(**meta_dict)
            model_form = self._build_model_form(
                model_form_cls=ModelForm,
                supports_access_control=supports_access_control,
                id=existing.id,
                base_model_id=existing.base_model_id,
                name=existing.name,
                meta=meta_obj,
                params=_params_without_tag_scanning(ModelParams, existing.params),
                access_payload=self._resolve_model_access_payload(
                    model_obj=existing,
                    supports_access_control=supports_access_control,
                ),
                is_active=existing.is_active,
            )
            if await Models.update_model_by_id(openwebui_model_id, model_form) is None:
                raise _ModelWriteRefused(
                    f"Open WebUI did not report the write to {openwebui_model_id} as "
                    "landed, so it is counted as not written"
                )

        else:
            meta_dict = {}
            if update_capabilities and (capabilities is not None or capability_defaults):
                merged_caps = _merged_capabilities(None, capabilities, capability_defaults)
                if merged_caps:
                    meta_dict["capabilities"] = merged_caps
                builtin_tool_defaults = file_context_builtin_tool_defaults(merged_caps)
                if builtin_tool_defaults:
                    meta_dict["builtinTools"] = {**builtin_tool_defaults}
            if update_images and profile_image_url:
                meta_dict["profile_image_url"] = profile_image_url
            if update_images and image_source_url:
                _ensure_pipe_meta(meta_dict)["image_source_url"] = image_source_url
                if image_source_kind:
                    _ensure_pipe_meta(meta_dict)["image_source_kind"] = image_source_kind
            if update_descriptions and description:
                meta_dict["description"] = description

            _apply_filter_ids(meta_dict)
            _apply_default_filter_ids(meta_dict)
            _apply_direct_uploads_filter_ids(meta_dict)
            _apply_image_gen_filter_ids(meta_dict)
            _apply_video_gen_filter_ids(
                meta_dict,
                video_gen_filter_function_id=video_gen_filter_function_id,
                video_gen_filter_supported=video_gen_filter_supported,
                auto_attach_video_gen_filter=auto_attach_video_gen_filter,
            )
            video_detached = _detached_with_default_off(
                meta_dict,
                prune_key="video_gen_filter_id",
                filter_function_ids=[video_gen_filter_function_id] if video_gen_filter_function_id else [],
                auto_default=auto_default_video_gen_filter,
            )
            _apply_video_default_filter_ids(
                meta_dict,
                video_gen_filter_function_id=video_gen_filter_function_id,
                video_gen_filter_supported=video_gen_filter_supported,
                auto_default_video_gen_filter=auto_default_video_gen_filter,
                detached=video_detached,
            )
            image_detached = _detached_with_default_off(
                meta_dict,
                prune_key="image_filter_ids",
                filter_function_ids=image_filter_function_ids,
                auto_default=auto_default_image_filter,
            )
            image_hands_off = (
                "image_filter_ids" in hands_off
                or not image_filter_ids_known
                or image_ids_unresolved
            )
            _apply_list_filter_ids(
                meta_dict,
                filter_function_ids=image_filter_function_ids,
                filter_supported=image_filter_supported,
                auto_attach=auto_attach_image_filter,
                prune_key="image_filter_ids",
                hands_off=image_hands_off,
                retired_ids=retired_image_filter_ids,
            )
            _apply_list_default_filter_ids(
                meta_dict,
                detached=image_detached,
                filter_function_ids=image_filter_function_ids,
                filter_supported=image_filter_supported,
                auto_default=auto_default_image_filter,
                hands_off=image_hands_off,
            )
            fusion_detached = _detached_with_default_off(
                meta_dict,
                prune_key="fusion_filter_ids",
                filter_function_ids=fusion_filter_function_ids,
                auto_default=auto_default_fusion_filter,
            )
            fusion_hands_off = "fusion_filter_ids" in hands_off or not fusion_ids_known
            _apply_list_filter_ids(
                meta_dict,
                filter_function_ids=fusion_filter_function_ids,
                filter_supported=fusion_filter_supported,
                auto_attach=auto_attach_fusion_filter,
                prune_key="fusion_filter_ids",
                hands_off=fusion_hands_off,
            )
            _apply_list_default_filter_ids(
                meta_dict,
                detached=fusion_detached,
                filter_function_ids=fusion_filter_function_ids,
                filter_supported=fusion_filter_supported,
                auto_default=auto_default_fusion_filter,
                hands_off=fusion_hands_off,
            )
            pr_hands_off = not provider_routing_ids_known
            pr_detached = set() if pr_hands_off else _detached_with_default_off(
                meta_dict,
                prune_key="provider_routing_filter_id",
                filter_function_ids=[provider_routing_filter_id] if provider_routing_filter_id else [],
                auto_default=auto_default_provider_routing_filter,
            )
            if _apply_provider_routing_filter_ids(meta_dict):
                self.logger.debug(
                    "Attached provider routing filter '%s' to new model '%s'",
                    provider_routing_filter_id,
                    openwebui_model_id,
                )
            _apply_provider_routing_default_filter_ids(
                meta_dict,
                detached=pr_detached,
                provider_routing_filter_id=provider_routing_filter_id,
                auto_default_provider_routing_filter=auto_default_provider_routing_filter,
                hands_off=pr_hands_off,
            )

            if openrouter_pipe_capabilities is not None:
                pipe_meta = _ensure_pipe_meta(meta_dict)
                pipe_meta["capabilities"] = dict(openrouter_pipe_capabilities)
                meta_dict[_PIPE_METADATA_KEY] = pipe_meta

            if not meta_dict:
                return

            meta_obj = ModelMeta(**meta_dict)
            if meta_dict.get("profile_image_url") and not _icon_was_stored(meta_obj):
                _drop_unsaved_icon_source(meta_dict)
                meta_obj = ModelMeta(**meta_dict)
            params_obj = _params_without_tag_scanning(ModelParams, None)

            access_mode = new_model_access_control
            access_payload = self._default_new_model_access_payload(
                access_mode=access_mode,
                supports_access_control=supports_access_control,
            )

            model_form = self._build_model_form(
                model_form_cls=ModelForm,
                supports_access_control=supports_access_control,
                id=openwebui_model_id,
                base_model_id=None,
                name=name,
                meta=meta_obj,
                params=params_obj,
                access_payload=access_payload,
                is_active=True,
            )
            if await Models.insert_new_model(model_form, user_id="") is None:
                raise _ModelWriteRefused(
                    f"Open WebUI did not report the insert of {openwebui_model_id} as "
                    "landed, so it is counted as not written"
                )
