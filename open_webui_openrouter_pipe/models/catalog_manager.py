"""Model catalog management subsystem.

This module handles synchronization of model metadata (capabilities, icons, descriptions)
from OpenRouter's frontend catalog to Open WebUI's model database.

Key responsibilities:
- Schedule metadata sync tasks on pipe invocation
- Fetch frontend catalog and extract icon/web-search mappings
- Download and convert profile images to data URLs
- Update or insert model records with metadata in OWUI database
- Auto-attach companion filters (OpenRouter Search, Direct Uploads)
- Respect per-model advanced params (disable_model_metadata_sync, etc.)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Iterable, Optional, TYPE_CHECKING
from urllib.parse import quote

import aiohttp
from fastapi.concurrency import run_in_threadpool

from ..core.timing_logger import timed

# Open WebUI internals (imported locally in methods for test compatibility)
try:
    from open_webui.models.models import ModelForm
except ImportError:
    ModelForm = None  # type: ignore

from ..core.config import _OPENROUTER_SITE_URL, _OPENROUTER_FRONTEND_MODELS_URL
# Lazy import to avoid circular dependency.
from .registry import OpenRouterModelRegistry, ModelFamily

if TYPE_CHECKING:
    from ..pipe import Pipe


class ModelCatalogManager:
    """Manages model metadata synchronization from OpenRouter to Open WebUI."""

    @timed
    def __init__(
        self,
        *,
        pipe: "Pipe",
        multimodal_handler: Any,
        logger: logging.Logger,
    ):
        """Initialize the catalog manager.

        Args:
            pipe: Reference to parent Pipe instance (for _create_http_session, _ensure_*_filter_function_id)
            multimodal_handler: Reference to MultimodalHandler (for _fetch_image_as_data_url, _fetch_maker_profile_image_url)
            logger: Logger instance
        """
        self._pipe = pipe
        self._multimodal_handler = multimodal_handler
        self.logger = logger

        # State for sync scheduling
        self._model_metadata_sync_task: asyncio.Task | None = None
        self._model_metadata_sync_key: tuple[Any, ...] | None = None

    @timed
    def maybe_schedule_model_metadata_sync(
        self,
        selected_models: list[dict[str, Any]],
        *,
        pipe_identifier: str,
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
        if not (
            valves.UPDATE_MODEL_CAPABILITIES
            or valves.UPDATE_MODEL_IMAGES
            or valves.UPDATE_MODEL_DESCRIPTIONS
            or valves.AUTO_ATTACH_ORS_FILTER
            or valves.AUTO_INSTALL_ORS_FILTER
            or valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER
            or valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER
            or valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER
        ):
            return
        if not selected_models:
            return
        last_fetch = getattr(OpenRouterModelRegistry, "_last_fetch", 0.0)
        sync_key = (
            pipe_identifier,
            float(last_fetch or 0.0),
            str(valves.MODEL_ID or ""),
            bool(valves.UPDATE_MODEL_IMAGES),
            bool(valves.UPDATE_MODEL_CAPABILITIES),
            bool(valves.UPDATE_MODEL_DESCRIPTIONS),
            bool(valves.AUTO_ATTACH_ORS_FILTER),
            bool(valves.AUTO_INSTALL_ORS_FILTER),
            bool(valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER),
            bool(valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER),
            bool(valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER),
        )
        if sync_key == self._model_metadata_sync_key:
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

    @timed
    def _build_icon_mapping(self, frontend_data: dict[str, Any] | None) -> dict[str, str]:
        """Build a slug -> icon URL mapping from the frontend catalog."""
        if not isinstance(frontend_data, dict):
            return {}
        raw_items = frontend_data.get("data")
        if not isinstance(raw_items, list):
            return {}

        icon_mapping: dict[str, str] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug:
                continue
            if slug in icon_mapping:
                continue

            @timed
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
                elif not icon_url.startswith(("http://", "https://", "data:image")):
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
    def _build_web_search_support_mapping(
        self,
        frontend_data: dict[str, Any] | None,
    ) -> dict[str, bool]:
        """Return a slug -> True mapping when the frontend catalog signals web-search support."""
        if not isinstance(frontend_data, dict):
            return {}
        raw_items = frontend_data.get("data")
        if not isinstance(raw_items, list):
            return {}

        mapping: dict[str, bool] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            slug = item.get("slug")
            if not isinstance(slug, str) or not slug:
                continue

            endpoint = item.get("endpoint")
            if not isinstance(endpoint, dict):
                continue

            supported_parameters = endpoint.get("supported_parameters")
            has_web_search_options = False
            if isinstance(supported_parameters, list):
                for entry in supported_parameters:
                    if isinstance(entry, str) and entry.strip() == "web_search_options":
                        has_web_search_options = True
                        break

            supports_native_web_search = False
            features = endpoint.get("features")
            if isinstance(features, dict):
                supports_native_web_search = features.get("supports_native_web_search") is True

            supports_priced_web_search = False
            pricing = endpoint.get("pricing")
            if isinstance(pricing, dict):
                supports_priced_web_search = OpenRouterModelRegistry._supports_web_search(pricing)

            if supports_native_web_search or has_web_search_options or supports_priced_web_search:
                mapping[slug] = True

        return mapping

    @timed
    async def _fetch_frontend_model_catalog(
        self,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any] | None:
        """Fetch OpenRouter's public frontend model catalog (no auth)."""
        try:
            async with session.get(_OPENROUTER_FRONTEND_MODELS_URL) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except Exception as exc:
            self.logger.debug("OpenRouter frontend catalog fetch failed: %s", exc)
            return None

        # Protect against corrupt or malicious JSON from remote source.
        # Ensure it's a dict before returning, logging invalid responses.
        if isinstance(payload, dict):
            return payload
        self.logger.warning(
            "OpenRouter frontend catalog returned invalid payload type '%s'; expected dict. Remote corruption or schema change detected.",
            type(payload).__name__,
        )
        return None

    @timed
    async def _build_maker_profile_image_mapping(
        self,
        session: aiohttp.ClientSession,
        maker_ids: Iterable[str],
    ) -> dict[str, str]:
        unique = sorted({(m or "").strip() for m in maker_ids if (m or "").strip()})
        if not unique:
            return {}

        semaphore = asyncio.Semaphore(10)
        results: dict[str, str] = {}

        @timed
        async def _fetch_maker_profile_image(maker_id: str) -> None:
            async with semaphore:
                image_url = await self._pipe._fetch_maker_profile_image_url(session, maker_id)
                if image_url:
                    results[maker_id] = image_url

        await asyncio.gather(
            *(_fetch_maker_profile_image(maker_id) for maker_id in unique),
            return_exceptions=True,
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
        if not (
            valves.UPDATE_MODEL_CAPABILITIES
            or valves.UPDATE_MODEL_IMAGES
            or valves.UPDATE_MODEL_DESCRIPTIONS
            or valves.AUTO_ATTACH_ORS_FILTER
            or valves.AUTO_INSTALL_ORS_FILTER
            or valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER
            or valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER
            or valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER
        ):
            return
        if not models:
            return
        if not pipe_identifier:
            return

        session = self._pipe._create_http_session()
        try:
            frontend_data = None
            if (
                valves.UPDATE_MODEL_IMAGES
                or valves.UPDATE_MODEL_CAPABILITIES
                or valves.AUTO_ATTACH_ORS_FILTER
            ):
                frontend_data = await self._fetch_frontend_model_catalog(session)

            icon_mapping: dict[str, str] = {}
            if valves.UPDATE_MODEL_IMAGES:
                icon_mapping = self._build_icon_mapping(frontend_data)

            web_search_mapping: dict[str, bool] = {}
            if valves.UPDATE_MODEL_CAPABILITIES or valves.AUTO_ATTACH_ORS_FILTER:
                web_search_mapping = self._build_web_search_support_mapping(frontend_data)

            maker_mapping: dict[str, str] = {}
            if valves.UPDATE_MODEL_IMAGES:
                missing_makers: set[str] = set()
                for model in models:
                    original_id = model.get("original_id")
                    if not isinstance(original_id, str) or not original_id:
                        continue
                    if original_id in icon_mapping:
                        continue
                    maker_id = original_id.split("/", 1)[0]
                    if maker_id:
                        missing_makers.add(maker_id)
                if missing_makers:
                    maker_mapping = await self._build_maker_profile_image_mapping(
                        session,
                        missing_makers,
                    )

            icon_data_mapping: dict[str, str] = {}
            maker_data_mapping: dict[str, str] = {}
            if valves.UPDATE_MODEL_IMAGES:
                slug_to_icon_url: dict[str, str] = {}
                for model in models:
                    original_id = model.get("original_id")
                    if not isinstance(original_id, str) or not original_id:
                        continue
                    icon_url = icon_mapping.get(original_id)
                    if icon_url:
                        slug_to_icon_url[original_id] = icon_url

                maker_to_image_url = {k: v for k, v in maker_mapping.items() if isinstance(v, str) and v}

                unique_urls = sorted(set(slug_to_icon_url.values()) | set(maker_to_image_url.values()))
                if unique_urls:
                    url_to_data: dict[str, str] = {}
                    fetch_semaphore = asyncio.Semaphore(10)

                    @timed
                    async def _fetch_image_data_url(url: str) -> None:
                        async with fetch_semaphore:
                            data_url = await self._pipe._fetch_image_as_data_url(session, url)
                            if data_url:
                                url_to_data[url] = data_url

                    await asyncio.gather(
                        *(_fetch_image_data_url(url) for url in unique_urls),
                        return_exceptions=True,
                    )

                    icon_data_mapping = {
                        slug: url_to_data.get(url, "")
                        for slug, url in slug_to_icon_url.items()
                        if url_to_data.get(url)
                    }
                    maker_data_mapping = {
                        maker: url_to_data.get(url, "")
                        for maker, url in maker_to_image_url.items()
                        if url_to_data.get(url)
                    }

            # DB writes are performed via OWUI helper functions in a threadpool.
            semaphore = asyncio.Semaphore(10)
            ors_filter_function_id: str | None = None
            if valves.AUTO_ATTACH_ORS_FILTER or valves.AUTO_INSTALL_ORS_FILTER:
                try:
                    ors_filter_function_id = await run_in_threadpool(self._pipe._ensure_ors_filter_function_id)
                except Exception as exc:
                    self.logger.debug("OpenRouter Search filter ensure failed: %s", exc)
                    ors_filter_function_id = None

            direct_uploads_filter_function_id: str | None = None
            if (
                valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER
                or valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER
            ):
                try:
                    direct_uploads_filter_function_id = await run_in_threadpool(
                        self._pipe._ensure_direct_uploads_filter_function_id
                    )
                except Exception as exc:
                    self.logger.debug("OpenRouter Direct Uploads filter ensure failed: %s", exc)
                    direct_uploads_filter_function_id = None

            if valves.AUTO_ATTACH_ORS_FILTER:
                if not ors_filter_function_id:
                    self.logger.warning(
                        "AUTO_ATTACH_ORS_FILTER is enabled but the OpenRouter Search filter is not installed. "
                        "Enable AUTO_INSTALL_ORS_FILTER (or install the filter manually) to show the OpenRouter Search toggle in the UI."
                    )
                else:
                    supported_models = 0
                    for model in models:
                        original_id = model.get("original_id")
                        if isinstance(original_id, str) and original_id and web_search_mapping.get(original_id):
                            supported_models += 1
                    self.logger.info(
                        "Auto-attaching OpenRouter Search filter '%s' to %d/%d model(s) that support OpenRouter web search.",
                        ors_filter_function_id,
                        supported_models,
                        len(models),
                    )

            if valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER:
                if not direct_uploads_filter_function_id:
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
                            except Exception:
                                supported = False
                            if supported:
                                supported_models += 1
                    self.logger.info(
                        "Auto-attaching OpenRouter Direct Uploads filter '%s' to %d/%d model(s) that support direct uploads.",
                        direct_uploads_filter_function_id,
                        supported_models,
                        len(models),
                    )

            @timed
            async def _apply(model: dict[str, Any]) -> None:
                openrouter_id = model.get("id")
                name = model.get("name")
                if not isinstance(openrouter_id, str) or not openrouter_id:
                    return
                if not isinstance(name, str) or not name:
                    name = openrouter_id

                openwebui_model_id = f"{pipe_identifier}.{openrouter_id}"

                @timed
                def _safe_supports(feature: str) -> bool:
                    try:
                        return bool(ModelFamily.supports(feature, openrouter_id))
                    except Exception:
                        return False

                pipe_capabilities = {
                    "file_input": _safe_supports("file_input"),
                    "audio_input": _safe_supports("audio_input"),
                    "video_input": _safe_supports("video_input"),
                    "vision": _safe_supports("vision"),
                }

                capabilities = None
                if valves.UPDATE_MODEL_CAPABILITIES:
                    raw_caps = model.get("capabilities")
                    if isinstance(raw_caps, dict):
                        capabilities = dict(raw_caps)
                        # OWUI Web Search is OWUI-native and works with any model. Keep the
                        # Integrations "Web Search" toggle available for all OpenRouter-pipe models.
                        capabilities["web_search"] = True

                description = None
                if valves.UPDATE_MODEL_DESCRIPTIONS:
                    norm_id = model.get("norm_id")
                    spec = ModelFamily._lookup_spec(str(norm_id or ""))
                    raw_desc = spec.get("description")
                    if isinstance(raw_desc, str):
                        cleaned = raw_desc.strip()
                        if cleaned:
                            description = cleaned

                profile_image_url = None
                if valves.UPDATE_MODEL_IMAGES:
                    original_id = model.get("original_id")
                    if isinstance(original_id, str) and original_id:
                        profile_image_url = icon_data_mapping.get(original_id)
                        if not profile_image_url:
                            maker_id = original_id.split("/", 1)[0]
                            profile_image_url = maker_data_mapping.get(maker_id)

                ors_supported = False
                auto_attach_or_default = bool(
                    ors_filter_function_id
                    and (
                        valves.AUTO_ATTACH_ORS_FILTER
                        or valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER
                    )
                )
                if auto_attach_or_default:
                    original_id = model.get("original_id")
                    if isinstance(original_id, str) and original_id and web_search_mapping.get(original_id):
                        ors_supported = True

                native_supported = bool(
                    pipe_capabilities.get("file_input")
                    or pipe_capabilities.get("audio_input")
                    or pipe_capabilities.get("video_input")
                )
                auto_attach_direct_uploads = bool(
                    direct_uploads_filter_function_id and valves.AUTO_ATTACH_DIRECT_UPLOADS_FILTER
                )

                if (
                    not capabilities
                    and not description
                    and not profile_image_url
                    and not auto_attach_or_default
                    and not pipe_capabilities
                    and not auto_attach_direct_uploads
                ):
                    return

                async with semaphore:
                    try:
                        await run_in_threadpool(
                            self._update_or_insert_model_with_metadata,
                            openwebui_model_id,
                            name,
                            capabilities,
                            profile_image_url,
                            valves.UPDATE_MODEL_CAPABILITIES,
                            valves.UPDATE_MODEL_IMAGES,
                            filter_function_id=ors_filter_function_id,
                            filter_supported=ors_supported,
                            auto_attach_filter=auto_attach_or_default,
                            auto_default_filter=valves.AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER,
                            direct_uploads_filter_function_id=direct_uploads_filter_function_id,
                            direct_uploads_filter_supported=native_supported,
                            auto_attach_direct_uploads_filter=auto_attach_direct_uploads,
                            openrouter_pipe_capabilities=pipe_capabilities,
                            description=description,
                            update_descriptions=valves.UPDATE_MODEL_DESCRIPTIONS,
                        )
                    except Exception as exc:
                        self.logger.debug(
                            "Model metadata sync failed (model=%s): %s",
                            openwebui_model_id,
                            exc,
                        )

            await asyncio.gather(*(_apply(model) for model in models), return_exceptions=True)
        finally:
            with contextlib.suppress(Exception):
                await session.close()

    @timed
    def _update_or_insert_model_with_metadata(
        self,
        openwebui_model_id: str,
        name: str,
        capabilities: Optional[dict],
        profile_image_url: Optional[str],
        update_capabilities: bool,
        update_images: bool,
        *,
        filter_function_id: str | None = None,
        filter_supported: bool = False,
        auto_attach_filter: bool = False,
        auto_default_filter: bool = False,
        direct_uploads_filter_function_id: str | None = None,
        direct_uploads_filter_supported: bool = False,
        auto_attach_direct_uploads_filter: bool = False,
        openrouter_pipe_capabilities: dict[str, bool] | None = None,
        description: str | None = None,
        update_descriptions: bool = False,
    ):
        """Safely update existing model or insert new overlay with metadata, never touching owner."""
        from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models

        openwebui_model_id = (openwebui_model_id or "").strip()
        if not openwebui_model_id:
            return
        name = (name or "").strip() or openwebui_model_id

        existing = Models.get_model_by_id(openwebui_model_id)

        # Per-model advanced params: allow operators to prevent the pipe from overwriting manually-edited
        # model settings (capability checkboxes, icons, filter attachments/defaults, etc).
        disable_model_metadata_sync = False
        disable_capability_updates = False
        disable_image_updates = False
        disable_openrouter_search_auto_attach = False
        disable_openrouter_search_default_on = False
        disable_direct_uploads_auto_attach = False
        disable_description_updates = False

        if existing is not None:
            # Lazy import to avoid circular dependency
            from ..api.transforms import _get_disable_param

            params = getattr(existing, "params", None)
            disable_model_metadata_sync = _get_disable_param(params, "disable_model_metadata_sync")
            disable_capability_updates = _get_disable_param(params, "disable_capability_updates")
            disable_image_updates = _get_disable_param(params, "disable_image_updates")
            disable_openrouter_search_auto_attach = _get_disable_param(params, "disable_openrouter_search_auto_attach")
            disable_openrouter_search_default_on = _get_disable_param(params, "disable_openrouter_search_default_on")
            disable_direct_uploads_auto_attach = _get_disable_param(params, "disable_direct_uploads_auto_attach")
            disable_description_updates = _get_disable_param(params, "disable_description_updates")

        if disable_model_metadata_sync:
            return

        if disable_capability_updates:
            update_capabilities = False
        if disable_image_updates:
            update_images = False
        if disable_openrouter_search_auto_attach:
            auto_attach_filter = False
        if disable_openrouter_search_default_on:
            auto_default_filter = False
        if disable_direct_uploads_auto_attach:
            auto_attach_direct_uploads_filter = False
        if disable_description_updates:
            update_descriptions = False

        @timed
        def _ensure_pipe_meta(meta_dict: dict) -> dict:
            pipe_meta = meta_dict.get("openrouter_pipe")
            if isinstance(pipe_meta, dict):
                return pipe_meta
            pipe_meta = {}
            meta_dict["openrouter_pipe"] = pipe_meta
            return pipe_meta

        @timed
        def _normalize_id_list(meta_dict: dict, key: str) -> list[str]:
            current = meta_dict.get(key, [])
            if not isinstance(current, list):
                return []
            normalized: list[str] = []
            for entry in current:
                if isinstance(entry, str) and entry:
                    normalized.append(entry)
            return normalized

        @timed
        def _dedupe_preserve_order(entries: list[str]) -> list[str]:
            seen: set[str] = set()
            deduped: list[str] = []
            for entry in entries:
                if entry in seen:
                    continue
                seen.add(entry)
                deduped.append(entry)
            return deduped

        @timed
        def _apply_filter_ids(meta_dict: dict) -> bool:
            if not auto_attach_filter or not filter_function_id:
                return False
            normalized = _normalize_id_list(meta_dict, "filterIds")
            pipe_meta = meta_dict.get("openrouter_pipe")
            previous_id = None
            if isinstance(pipe_meta, dict):
                prev = pipe_meta.get("openrouter_search_filter_id")
                if isinstance(prev, str) and prev and prev != filter_function_id:
                    previous_id = prev
            had = set(normalized)
            wanted = set(had)
            if filter_supported:
                wanted.add(filter_function_id)
            else:
                wanted.discard(filter_function_id)
            if previous_id:
                wanted.discard(previous_id)
            if wanted == had:
                return False
            # Preserve order as much as possible; append new id at the end.
            if filter_supported and filter_function_id not in normalized:
                normalized.append(filter_function_id)
            normalized = [fid for fid in normalized if fid in wanted]
            meta_dict["filterIds"] = _dedupe_preserve_order(normalized)
            return True

        @timed
        def _apply_direct_uploads_filter_ids(meta_dict: dict) -> bool:
            if not auto_attach_direct_uploads_filter or not direct_uploads_filter_function_id:
                return False
            normalized = _normalize_id_list(meta_dict, "filterIds")
            pipe_meta = meta_dict.get("openrouter_pipe")
            previous_id = None
            if isinstance(pipe_meta, dict):
                prev = pipe_meta.get("direct_uploads_filter_id")
                if isinstance(prev, str) and prev and prev != direct_uploads_filter_function_id:
                    previous_id = prev
            had = set(normalized)
            wanted = set(had)
            if direct_uploads_filter_supported:
                wanted.add(direct_uploads_filter_function_id)
            else:
                wanted.discard(direct_uploads_filter_function_id)
            if previous_id:
                wanted.discard(previous_id)
            if wanted == had:
                return False
            # Preserve order as much as possible; append new id at the end.
            if direct_uploads_filter_supported and direct_uploads_filter_function_id not in normalized:
                normalized.append(direct_uploads_filter_function_id)
            normalized = [fid for fid in normalized if fid in wanted]
            meta_dict["filterIds"] = _dedupe_preserve_order(normalized)
            pipe_meta = _ensure_pipe_meta(meta_dict)
            pipe_meta["direct_uploads_filter_id"] = direct_uploads_filter_function_id
            meta_dict["openrouter_pipe"] = pipe_meta
            return True

        @timed
        def _apply_default_filter_ids(meta_dict: dict) -> bool:
            if not auto_default_filter or not filter_function_id or not filter_supported:
                return False

            # Never set a default filter unless the filter is actually attached. This matters when
            # operators disable auto-attach (or when the filter is removed/unsupported) so we don't
            # leave the model in a "default on" state for a filter that isn't present.
            filter_ids = _normalize_id_list(meta_dict, "filterIds")
            if filter_function_id not in filter_ids:
                return False

            pipe_meta = _ensure_pipe_meta(meta_dict)
            seeded_key = "openrouter_search_default_seeded"
            previous_id = pipe_meta.get("openrouter_search_filter_id")
            previous_id_str = previous_id if isinstance(previous_id, str) else ""

            default_ids = _normalize_id_list(meta_dict, "defaultFilterIds")
            changed = False

            # If the filter id changed (rare), migrate defaults while preserving operator intent.
            if previous_id_str and previous_id_str != filter_function_id and previous_id_str in default_ids:
                default_ids = [filter_function_id if fid == previous_id_str else fid for fid in default_ids]
                changed = True

            seeded = bool(pipe_meta.get(seeded_key, False))
            if filter_function_id in default_ids:
                if not seeded:
                    pipe_meta[seeded_key] = True
                    changed = True
            else:
                if not seeded:
                    default_ids.append(filter_function_id)
                    pipe_meta[seeded_key] = True
                    changed = True

            if previous_id_str != filter_function_id:
                pipe_meta["openrouter_search_filter_id"] = filter_function_id
                changed = True

            if not changed:
                return False

            meta_dict["defaultFilterIds"] = _dedupe_preserve_order(default_ids)
            meta_dict["openrouter_pipe"] = pipe_meta
            return True

        if existing:
            # Update existing model - preserve ALL existing fields including owner
            meta_dict = {}
            if existing.meta:
                meta_dict.update(existing.meta.model_dump())

            meta_updated = False

            if update_capabilities and capabilities is not None:
                existing_caps = meta_dict.get("capabilities")
                merged_caps: dict[str, Any] = dict(existing_caps) if isinstance(existing_caps, dict) else {}
                for key, value in capabilities.items():
                    merged_caps[key] = value
                if merged_caps != existing_caps:
                    meta_dict["capabilities"] = merged_caps
                    meta_updated = True

            if update_images and profile_image_url:
                if meta_dict.get("profile_image_url") != profile_image_url:
                    meta_dict["profile_image_url"] = profile_image_url
                    meta_updated = True

            if update_descriptions and description:
                if meta_dict.get("description") != description:
                    meta_dict["description"] = description
                    meta_updated = True

            if _apply_filter_ids(meta_dict):
                meta_updated = True

            if _apply_default_filter_ids(meta_dict):
                meta_updated = True

            if _apply_direct_uploads_filter_ids(meta_dict):
                meta_updated = True

            if openrouter_pipe_capabilities is not None:
                pipe_meta = _ensure_pipe_meta(meta_dict)
                if pipe_meta.get("capabilities") != openrouter_pipe_capabilities:
                    pipe_meta["capabilities"] = dict(openrouter_pipe_capabilities)
                    meta_dict["openrouter_pipe"] = pipe_meta
                    meta_updated = True

            if not meta_updated:
                return

            meta_obj = ModelMeta(**meta_dict)
            model_form = ModelForm(
                id=existing.id,
                base_model_id=existing.base_model_id,
                name=existing.name,
                meta=meta_obj,
                params=existing.params if existing.params else ModelParams(),
                access_control=existing.access_control,
                is_active=existing.is_active,
            )
            Models.update_model_by_id(openwebui_model_id, model_form)

        else:
            # Insert new overlay model - do NOT set user_id/owner
            meta_dict = {}
            if update_capabilities and capabilities is not None:
                meta_dict["capabilities"] = capabilities
            if update_images and profile_image_url:
                meta_dict["profile_image_url"] = profile_image_url
            if update_descriptions and description:
                meta_dict["description"] = description

            _apply_filter_ids(meta_dict)
            _apply_default_filter_ids(meta_dict)
            _apply_direct_uploads_filter_ids(meta_dict)

            if openrouter_pipe_capabilities is not None:
                pipe_meta = _ensure_pipe_meta(meta_dict)
                pipe_meta["capabilities"] = dict(openrouter_pipe_capabilities)
                meta_dict["openrouter_pipe"] = pipe_meta

            if not meta_dict:
                # Nothing to insert, skip
                return

            # Create proper ModelMeta and ModelParams objects
            meta_obj = ModelMeta(**meta_dict)
            params_obj = ModelParams()

            model_form = ModelForm(
                id=openwebui_model_id,
                base_model_id=None,
                name=name,
                meta=meta_obj,
                params=params_obj,
                access_control=None,
                is_active=True,
            )
            # Use empty user_id to let OWUI handle ownership defaults
            Models.insert_new_model(model_form, user_id="")
