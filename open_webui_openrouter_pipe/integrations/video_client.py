from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp

from ..core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_REFERER,
    _OPENROUTER_TITLE,
    _apply_owui_forward_user_headers,
)
from ..core.errors import UpstreamBodyUnreadable, _build_openrouter_api_error
from ..core.utils import _DEFAULT_VALVES, http_timeout
from ..requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
    _debug_print_response,
)
from .video_types import VideoGenerationError

_VIDEO_CATALOG_TIMEOUT_SECONDS = 15

_VIDEO_BODY_EXCERPT_CHARS = 200


def _video_body_not_an_object(resp: Any, payload: Any, endpoint: str) -> UpstreamBodyUnreadable:
    return UpstreamBodyUnreadable(
        endpoint=endpoint,
        body_excerpt=repr(payload)[:_VIDEO_BODY_EXCERPT_CHARS],
        content_type=getattr(resp, "content_type", None),
    )


async def _decode_video_json_body(resp: Any, endpoint: str, logger: Any) -> Any:
    try:
        return await resp.json()
    except (aiohttp.ClientPayloadError, aiohttp.ServerDisconnectedError):
        raise
    except Exception:
        logger.debug(
            "OpenRouter video response was not decodable JSON; falling back to text",
            exc_info=True,
        )
        text = ""
        try:
            text = await resp.text()
            return json.loads(text)
        except (aiohttp.ClientPayloadError, aiohttp.ServerDisconnectedError):
            raise
        except Exception as exc:
            raise UpstreamBodyUnreadable(
                endpoint=endpoint,
                body_excerpt=text[:_VIDEO_BODY_EXCERPT_CHARS],
                content_type=getattr(resp, "content_type", None),
            ) from exc


_VIDEO_MIME_EXTENSIONS: dict[str, str] = {
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "video/x-m4v": ".m4v",
    "video/3gpp": ".3gp",
    "video/3gpp2": ".3g2",
    "video/x-matroska": ".mkv",
    "video/ogg": ".ogv",
    "video/mpeg": ".mpeg",
    "video/x-msvideo": ".avi",
}



def extension_for_video_mime(mime: str) -> str:
    normalized = (mime or "").split(";", 1)[0].strip().lower()
    return _VIDEO_MIME_EXTENSIONS.get(normalized, ".mp4")


class OpenRouterVideoClient:

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
        valves: Any = None,
    ) -> None:
        self._session = session
        self._valves = valves
        self._base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self._api_key = api_key
        self._logger = logger
        self._http_referer = http_referer or _OPENROUTER_REFERER
        self._user = user
        self._owui_chat_id = owui_chat_id

    def _timeout(self) -> aiohttp.ClientTimeout:
        return http_timeout(self._valves if self._valves is not None else _DEFAULT_VALVES)

    def _timeout_kwargs(self) -> dict[str, Any]:
        if self._valves is None:
            return {}
        return {"timeout": self._timeout()}

    def content_url(self, job_id: str, index: int = 0) -> str:
        url = f"{self._base_url}/videos/{job_id}/content"
        return url if index <= 0 else f"{url}?index={index}"

    def poll_url(self, job_id: str, polling_url: Any = None) -> str:
        fallback = f"{self._base_url}/videos/{job_id}"
        candidate = polling_url.strip() if isinstance(polling_url, str) else ""
        if not candidate:
            return fallback
        if candidate.startswith("/"):
            parts = self._base_url.split("/", 3)
            return f"{parts[0]}//{parts[2]}{candidate}" if len(parts) >= 3 else fallback
        if candidate == self._base_url or candidate.startswith(f"{self._base_url}/"):
            return candidate
        return fallback

    @staticmethod
    def output_count(payload: Any) -> int:
        urls = payload.get("unsigned_urls") if isinstance(payload, dict) else None
        if not isinstance(urls, list):
            return 1
        return max(1, sum(1 for item in urls if isinstance(item, str) and item.strip()))

    def bearer_header(self) -> dict[str, str]:
        if not self._api_key:
            raise VideoGenerationError("OpenRouter API key is required for video generation.")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        return _apply_owui_forward_user_headers(headers, self._user, self._owui_chat_id)

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise VideoGenerationError("OpenRouter API key is required for video generation.")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            "HTTP-Referer": self._http_referer,
        }
        return _apply_owui_forward_user_headers(headers, self._user, self._owui_chat_id)

    async def list_models(self) -> list[dict[str, Any]]:
        url = f"{self._base_url}/videos/models"
        headers = self._headers()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=self._logger)
        async with self._session.get(
            url, headers=headers,
            timeout=aiohttp.ClientTimeout(total=_VIDEO_CATALOG_TIMEOUT_SECONDS),
        ) as resp:
            if resp.status >= 400:
                await _debug_print_error_response(resp, logger=self._logger)
            resp.raise_for_status()
            payload = await resp.json()
        _debug_print_response(payload, logger=self._logger)
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    async def model_modalities(self, model_id: str) -> list[str]:
        slug = (model_id or "").strip().strip("/")
        if not slug:
            return []
        url = f"{self._base_url}/models/{slug}/endpoints"
        try:
            async with self._session.get(
                url, headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=_VIDEO_CATALOG_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status >= 400:
                    return []
                payload = await resp.json()
        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, TimeoutError, OSError, ValueError):
            return []
        data = payload.get("data") if isinstance(payload, dict) else None
        arch = (data or {}).get("architecture") if isinstance(data, dict) else None
        found = (arch or {}).get("input_modalities") if isinstance(arch, dict) else None
        return [item for item in found if isinstance(item, str)] if isinstance(found, list) else []

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/videos"
        headers = self._headers()
        _debug_print_request(headers, {"method": "POST", "url": url, "json": payload}, logger=self._logger)
        async with self._session.post(url, headers=headers, json=payload, **self._timeout_kwargs()) as resp:
            if resp.status >= 400:
                body = await _debug_print_error_response(resp, logger=self._logger)
                raise _build_openrouter_api_error(
                    resp.status,
                    resp.reason or "",
                    body,
                    requested_model=str(payload.get("model") or "") or None,
                )
            data = await _decode_video_json_body(resp, url, self._logger)
        _debug_print_response(data, logger=self._logger)
        if not isinstance(data, dict):
            raise _video_body_not_an_object(resp, data, url)
        return data

    async def status(self, job_id: str, polling_url: Any = None) -> dict[str, Any]:
        safe_job_id = (job_id or "").strip()
        if not safe_job_id:
            raise VideoGenerationError("Video generation job id is missing.")
        url = self.poll_url(safe_job_id, polling_url)
        headers = self._headers()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=self._logger)
        async with self._session.get(url, headers=headers, **self._timeout_kwargs()) as resp:
            if resp.status >= 400:
                body = await _debug_print_error_response(resp, logger=self._logger)
                raise _build_openrouter_api_error(
                    resp.status,
                    resp.reason or "",
                    body,
                )
            data = await _decode_video_json_body(resp, url, self._logger)
        _debug_print_response(data, logger=self._logger)
        if not isinstance(data, dict):
            raise _video_body_not_an_object(resp, data, url)
        return data
