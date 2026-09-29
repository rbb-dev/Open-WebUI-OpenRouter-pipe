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
from ..core.utils import (
    _DEFAULT_VALVES,
    IMAGE_NO_IMAGES_REASON,
    clamp_text,
    http_timeout,
    image_failure_billing_suffix,
    summarise_names,
    utf8_stream_decoder,
)
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
)

_IMAGE_SSE_CONTENT_TYPE = "text/event-stream"
_IMAGE_SSE_PREFIX = "data:"
_IMAGE_SSE_DONE = "[DONE]"


def _over_ceiling_reason(what: str, max_decoded_bytes: int) -> str:
    return (
        f"{what} pushed this reply past the {max_decoded_bytes // (1024 * 1024)} MB "
        "BASE64_MAX_SIZE_MB ceiling for one generated-image reply"
    )


class _ProgressCallbackFailed(BaseException):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


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


def _image_stream_error(event: dict[str, Any], state: dict[str, Any]) -> str:
    detail = event.get("error")
    message = detail.get("message") if isinstance(detail, dict) else None
    reason = clamp_text(
        message if isinstance(message, str) and message else "no reason was given", 160
    )
    if state["data"]:
        sentence = f"OpenRouter reported a problem after delivering the image: {reason}."
        state["warning"] = sentence
        return sentence
    billed = chat_usage_to_responses_usage(state.get("usage"))
    raise ImageGenerationError(
        "OpenRouter stopped generating the image: "
        f"{reason}. "
        + image_failure_billing_suffix(billed),
        usage=billed or None,
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
            try:
                await on_progress(message)
            except Exception as exc:
                raise _ProgressCallbackFailed(exc) from exc

    async def _read_stream_as_buffered_response(
        self, resp: Any, on_progress: Any, state: dict[str, Any]
    ) -> dict[str, Any]:
        buffer = ""
        _utf8 = utf8_stream_decoder()
        try:
            async for chunk in resp.content.iter_any():
                buffer += _utf8.decode(chunk)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    await self._consume_stream_line(line, state, on_progress)
            buffer += _utf8.decode(b"", True)
            await self._consume_stream_line(buffer, state, on_progress)
        except _ProgressCallbackFailed as failed:
            raise failed.cause from failed.cause
        except ImageGenerationError:
            raise
        except Exception as exc:
            billed = chat_usage_to_responses_usage(state.get("usage"))
            if state["data"]:
                self._logger.warning(
                    "OpenRouter's image stream for this request stopped after the finished "
                    "image arrived: %s", clamp_text(str(exc) or type(exc).__name__, 160)
                )
                if not state["warning"]:
                    state["warning"] = (
                        "The image stream failed after the finished image had "
                        f"already arrived: {clamp_text(str(exc), 160)}."
                    )
            elif not billed:
                raise
            else:
                raise ImageGenerationError(
                    "OpenRouter's connection dropped before the image stream finished: "
                    f"{clamp_text(str(exc) or type(exc).__name__, 160)}. "
                    + image_failure_billing_suffix(billed),
                    usage=billed,
                ) from exc
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
        async with self._session.post(url, headers=headers, json=payload, **self._timeout_kwargs()) as resp:
            if resp.status >= 400:
                body = await _debug_print_error_response(resp, logger=self._logger)
                raise _build_openrouter_api_error(
                    resp.status,
                    resp.reason or "",
                    body,
                    requested_model=payload.get("model") if isinstance(payload, dict) else None,
                )
            state: dict[str, Any] = {
                "data": [], "usage": None, "previews": 0, "drawing": False, "warning": "",
            }
            if resp.content_type == _IMAGE_SSE_CONTENT_TYPE:
                try:
                    data = await self._read_stream_as_buffered_response(
                        resp, on_progress, state
                    )
                except (ImageGenerationError, aiohttp.ClientError, TimeoutError) as exc:
                    if not state["data"]:
                        if isinstance(exc, ImageGenerationError):
                            raise
                        raise ImageGenerationError(
                            "OpenRouter's image stream ended before the finished image "
                            "arrived. Nothing was billed for it."
                        ) from exc
                    self._logger.warning(
                        "Image stream failed after the finished image arrived; "
                        "delivering it anyway: %s", exc
                    )
                    if not state["warning"]:
                        state["warning"] = (
                            "The image stream failed after the finished image had "
                            f"already arrived: {clamp_text(str(exc), 160)}."
                        )
                    data = {"data": state["data"], "usage": state["usage"]}
            else:
                data = await resp.json()
        _debug_print_response(data, logger=self._logger)
        if not isinstance(data, dict):
            raise ImageGenerationError("OpenRouter image generation returned an invalid response.")

        billed = chat_usage_to_responses_usage(data.get("usage"))

        entries = data.get("data")
        if not isinstance(entries, list) or not entries:
            raise ImageGenerationError(IMAGE_NO_IMAGES_REASON, usage=billed)

        images: list[GeneratedImage] = []
        rejected: list[str] = []
        decoded_total = 0
        over_ceiling = 0
        over_ceiling_own = 0
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                rejected.append(clamp_text(f"an entry of type {type(entry).__name__} carried no image"))
                continue
            blob = entry.get("b64_json")
            if not isinstance(blob, str) or not blob:
                rejected.append(clamp_text(f"an entry with keys {sorted(entry)} carried no inline base64"))
                continue
            own_decoded = (len(blob) * 3) // 4
            decoded_total += own_decoded
            if 0 < max_decoded_bytes < decoded_total:
                over_ceiling += 1
                if 0 < max_decoded_bytes < own_decoded:
                    over_ceiling_own += 1
                rejected.append(
                    clamp_text(
                        _over_ceiling_reason(f"entry {index + 1} of {len(entries)}", max_decoded_bytes)
                    )
                )
                continue
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

        return ImageGenerationResult(
            images=images, usage=billed, rejected=rejected, warning=state.get("warning", ""),
            over_ceiling=over_ceiling, over_ceiling_own=over_ceiling_own,
        )
