"""HTTP client for OpenRouter image generation and image model catalogs."""

from __future__ import annotations

import base64
import binascii
import json
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

_IMAGE_SSE_CONTENT_TYPE = "text/event-stream"
_IMAGE_SSE_PREFIX = "data:"
_IMAGE_SSE_DONE = "[DONE]"


def _image_stream_partial(event: dict[str, Any], state: dict[str, Any]) -> str:
    index = event.get("partial_image_index")
    ordinal = (
        index + 1
        if isinstance(index, int) and not isinstance(index, bool) and index >= 0
        else state["previews"] + 1
    )
    state["previews"] = ordinal
    return f"Generating image… preview {ordinal}"


def _image_stream_text(event: dict[str, Any], state: dict[str, Any]) -> str:
    if event.get("phase") != "content" or state["drawing"]:
        return ""
    state["drawing"] = True
    return "Drawing the image…"


def _image_stream_completed(event: dict[str, Any], state: dict[str, Any]) -> str:
    entry: dict[str, Any] = {"b64_json": event.get("b64_json")}
    media_type = event.get("media_type")
    if isinstance(media_type, str) and media_type:
        entry["media_type"] = media_type
    state["data"].append(entry)
    usage = event.get("usage")
    if isinstance(usage, dict):
        state["usage"] = usage
    return ""


def _image_stream_error(event: dict[str, Any], _state: dict[str, Any]) -> str:
    detail = event.get("error")
    message = detail.get("message") if isinstance(detail, dict) else None
    raise ImageGenerationError(
        "OpenRouter stopped generating the image: "
        f"{clamp_text(message if isinstance(message, str) and message else 'no reason was given', 160)}. "
        "Nothing was billed."
    )


_IMAGE_STREAM_HANDLERS = {
    "image_generation.partial_image": _image_stream_partial,
    "image_generation.text_chunk": _image_stream_text,
    "image_generation.completed": _image_stream_completed,
    "error": _image_stream_error,
}


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

    async def _consume_stream_line(
        self, raw_line: str, state: dict[str, Any], on_progress: Any
    ) -> None:
        line = (raw_line or "").strip()
        if not line.startswith(_IMAGE_SSE_PREFIX):
            return
        payload = line[len(_IMAGE_SSE_PREFIX) :].strip()
        if not payload or payload == _IMAGE_SSE_DONE:
            return
        try:
            event = json.loads(payload)
        except ValueError:
            self._logger.debug("Image stream chunk was not readable JSON", exc_info=True)
            return
        if not isinstance(event, dict):
            return
        handler = _IMAGE_STREAM_HANDLERS.get(str(event.get("type")))
        if handler is None:
            return
        message = handler(event, state)
        if message and on_progress is not None:
            await on_progress(message)

    async def _read_stream_as_buffered_response(
        self, resp: Any, on_progress: Any
    ) -> dict[str, Any]:
        state: dict[str, Any] = {"data": [], "usage": None, "previews": 0, "drawing": False}
        buffer = ""
        async for chunk in resp.content.iter_any():
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                await self._consume_stream_line(line, state, on_progress)
        await self._consume_stream_line(buffer, state, on_progress)
        if not state["data"]:
            raise ImageGenerationError(
                "OpenRouter's image stream ended before the finished image arrived. "
                "Nothing was billed for it."
            )
        return {"data": state["data"], "usage": state["usage"]}

    async def generate(
        self, payload: dict[str, Any], *, max_decoded_bytes: int = 0, on_progress: Any = None
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
            if resp.content_type == _IMAGE_SSE_CONTENT_TYPE:
                data = await self._read_stream_as_buffered_response(resp, on_progress)
            else:
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
