"""HTTP client for OpenRouter image generation and image model catalogs."""

from __future__ import annotations

import base64
import binascii
from typing import Any

import aiohttp

from ..core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_REFERER,
    _OPENROUTER_TITLE,
    _apply_owui_forward_user_headers,
)
from ..core.costs import chat_usage_to_responses_usage
from ..core.errors import _build_openrouter_api_error
from ..requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
    _debug_print_response,
)
from ..storage.multimodal import _guess_image_mime_type, canonical_image_mime

_CATALOG_TIMEOUT_SECONDS = 15
"""Cap on one catalog or published-contract read.

The shared session leaves its total timeout unset so a long streaming reply is never cut
off. These are small GETs on the path that builds Open WebUI's model list, and the
contract read now runs once per model, so an unbounded wait here stalls the dropdown.
"""
from .image_types import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationResult,
    clamp_text,
    summarise_names,
)


class OpenRouterImageClient:

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        logger: Any,
        http_referer: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> None:
        self._session = session
        self._base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self._api_key = api_key
        self._logger = logger
        self._http_referer = http_referer or _OPENROUTER_REFERER
        self._user = user
        self._owui_chat_id = owui_chat_id

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise RuntimeError("OpenRouter API key is required for image catalog fetch.")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            "HTTP-Referer": self._http_referer,
        }
        return _apply_owui_forward_user_headers(headers, self._user, self._owui_chat_id)

    async def list_models(self) -> list[dict[str, Any]]:
        url = f"{self._base_url}/models?output_modalities=image"
        headers = self._headers()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=self._logger)
        async with self._session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=_CATALOG_TIMEOUT_SECONDS)
        ) as resp:
            if resp.status >= 400:
                await _debug_print_error_response(resp, logger=self._logger)
            resp.raise_for_status()
            payload = await resp.json()
        _debug_print_response(payload, logger=self._logger)
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    async def endpoints(self, model_id: str) -> list[dict[str, Any]]:
        safe_id = (model_id or "").strip()
        if not safe_id:
            raise ImageGenerationError("Image model id is missing.")
        url = f"{self._base_url}/images/models/{safe_id}/endpoints"
        headers = self._headers()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=self._logger)
        async with self._session.get(
            url, headers=headers, timeout=aiohttp.ClientTimeout(total=_CATALOG_TIMEOUT_SECONDS)
        ) as resp:
            if resp.status >= 400:
                await _debug_print_error_response(resp, logger=self._logger)
            resp.raise_for_status()
            payload = await resp.json()
        _debug_print_response(payload, logger=self._logger)
        if not isinstance(payload, dict):
            return []
        records = payload.get("endpoints")
        return [item for item in records if isinstance(item, dict)] if isinstance(records, list) else []

    async def generate(
        self, payload: dict[str, Any], *, max_decoded_bytes: int = 0
    ) -> ImageGenerationResult:
        url = f"{self._base_url}/images"
        headers = self._headers()
        _debug_print_request(headers, {"method": "POST", "url": url, "json": payload}, logger=self._logger)
        async with self._session.post(url, headers=headers, json=payload) as resp:
            if resp.status >= 400:
                body = await _debug_print_error_response(resp, logger=self._logger)
                raise _build_openrouter_api_error(
                    resp.status,
                    resp.reason or "",
                    body,
                    requested_model=payload.get("model") if isinstance(payload, dict) else None,
                )
            data = await resp.json()
        _debug_print_response(data, logger=self._logger)
        if not isinstance(data, dict):
            raise ImageGenerationError("OpenRouter image generation returned an invalid response.")

        billed = chat_usage_to_responses_usage(data.get("usage"))

        entries = data.get("data")
        if not isinstance(entries, list) or not entries:
            raise ImageGenerationError(
                "OpenRouter image generation returned no images.", usage=billed
            )

        images: list[GeneratedImage] = []
        rejected: list[str] = []
        decoded_total = 0
        for entry in entries:
            if not isinstance(entry, dict):
                rejected.append(clamp_text(f"an entry of type {type(entry).__name__} carried no image"))
                continue
            blob = entry.get("b64_json")
            if not isinstance(blob, str) or not blob:
                rejected.append(clamp_text(f"an entry with keys {sorted(entry)} carried no inline base64"))
                continue
            decoded_total += (len(blob) * 3) // 4
            if 0 < max_decoded_bytes < decoded_total:
                raise ImageGenerationError(
                    f"OpenRouter returned more image data than the {max_decoded_bytes // (1024 * 1024)} MB "
                    "BASE64_MAX_SIZE_MB ceiling allows; raise the valve, ask for fewer images, "
                    "or ask for a smaller resolution.",
                    usage=billed,
                )
            try:
                raw = base64.b64decode(blob, validate=True)
            except (binascii.Error, ValueError):
                rejected.append(clamp_text(f"an entry starting {blob[:24]!r} was not decodable base64"))
                continue
            raw_media_type = entry.get("media_type")
            declared = raw_media_type if isinstance(raw_media_type, str) else ""
            normalised = declared.split(";", 1)[0].strip().lower()
            mime_type = _guess_image_mime_type("", "", raw)
            if mime_type is None:
                mime_type = canonical_image_mime(normalised)
                if mime_type is None:
                    rejected.append(
                        clamp_text(
                            f"{len(raw)} byte(s) declared "
                            f"{clamp_text(declared or 'none', 40)!r} are not a recognised "
                            f"image (starts {raw[:16]!r})"
                        )
                    )
                    continue
            images.append(GeneratedImage(data=raw, mime_type=mime_type))

        if not images:
            detail = f" ({summarise_names(rejected)})" if rejected else ""
            raise ImageGenerationError(
                f"OpenRouter image generation returned no usable images{detail}.", usage=billed
            )

        return ImageGenerationResult(images=images, usage=billed, rejected=rejected)
