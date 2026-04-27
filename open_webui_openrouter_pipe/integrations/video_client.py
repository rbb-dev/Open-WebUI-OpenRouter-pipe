from __future__ import annotations

from typing import Any

import aiohttp

from ..core.config import _OPENROUTER_CATEGORIES, _OPENROUTER_REFERER, _OPENROUTER_TITLE
from ..requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
    _debug_print_response,
    _extract_error_message_from_body,
)
from .video_types import VideoGenerationError


def extension_for_video_mime(mime: str) -> str:
    normalized = (mime or "").split(";", 1)[0].strip().lower()
    if normalized == "video/webm":
        return ".webm"
    return ".mp4"


class OpenRouterVideoClient:

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        logger: Any,
        http_referer: str | None = None,
    ) -> None:
        self._session = session
        self._base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self._api_key = api_key
        self._logger = logger
        self._http_referer = http_referer or _OPENROUTER_REFERER

    def content_url(self, job_id: str) -> str:
        return f"{self._base_url}/videos/{job_id}/content"

    def bearer_header(self) -> dict[str, str]:
        if not self._api_key:
            raise VideoGenerationError("OpenRouter API key is required for video generation.")
        return {"Authorization": f"Bearer {self._api_key}"}

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise VideoGenerationError("OpenRouter API key is required for video generation.")
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            "HTTP-Referer": self._http_referer,
        }

    async def list_models(self) -> list[dict[str, Any]]:
        url = f"{self._base_url}/videos/models"
        headers = self._headers()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=self._logger)
        async with self._session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                await _debug_print_error_response(resp, logger=self._logger)
            resp.raise_for_status()
            payload = await resp.json()
        _debug_print_response(payload, logger=self._logger)
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/videos"
        headers = self._headers()
        _debug_print_request(headers, {"method": "POST", "url": url, "json": payload}, logger=self._logger)
        async with self._session.post(url, headers=headers, json=payload) as resp:
            if resp.status >= 400:
                body = await _debug_print_error_response(resp, logger=self._logger)
                detail = _extract_error_message_from_body(body)
                raise VideoGenerationError(detail or f"OpenRouter video generation failed with HTTP {resp.status}.")
            data = await resp.json()
        _debug_print_response(data, logger=self._logger)
        if not isinstance(data, dict):
            raise VideoGenerationError("OpenRouter video generation returned an invalid response.")
        return data

    async def status(self, job_id: str) -> dict[str, Any]:
        safe_job_id = (job_id or "").strip()
        if not safe_job_id:
            raise VideoGenerationError("Video generation job id is missing.")
        url = f"{self._base_url}/videos/{safe_job_id}"
        headers = self._headers()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=self._logger)
        async with self._session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                body = await _debug_print_error_response(resp, logger=self._logger)
                detail = _extract_error_message_from_body(body)
                raise VideoGenerationError(detail or f"OpenRouter video status failed with HTTP {resp.status}.")
            data = await resp.json()
        _debug_print_response(data, logger=self._logger)
        if not isinstance(data, dict):
            raise VideoGenerationError("OpenRouter video status returned an invalid response.")
        return data
