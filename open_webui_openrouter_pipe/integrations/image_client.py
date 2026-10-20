from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Any

import aiohttp

from ..core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_REFERER,
    _OPENROUTER_TITLE,
    _apply_owui_forward_user_headers,
)
from ..core.costs import chat_usage_to_responses_usage
from ..core.errors import UpstreamBodyUnreadable, _build_openrouter_api_error
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
from ..storage.multimodal import (
    _decode_base64_in_quanta,
    _guess_image_mime_type,
    canonical_image_mime,
    head_is_not_a_picture,
)
from ..storage.receive_spool import PartRef, ReceiveSpool, attach_parts

_CATALOG_TIMEOUT_SECONDS = 15
from .image_types import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationResult,
)

_IMAGE_SSE_CONTENT_TYPE = "text/event-stream"
_IMAGE_SSE_PREFIX = "data:"
_IMAGE_SSE_DONE = "[DONE]"

_IMAGE_DECODE_PREFIX_BYTES = 512
_IMAGE_DECODE_PREFIX_CHARS = (_IMAGE_DECODE_PREFIX_BYTES // 3 + 1) * 4


def _decoded_prefix(blob: str) -> bytes | None:
    try:
        return base64.b64decode(blob[:_IMAGE_DECODE_PREFIX_CHARS], validate=True)
    except (binascii.Error, ValueError):
        return None


def _classify(head: bytes, declared: str) -> str | None:
    mime_type = _guess_image_mime_type("", "", head)
    if mime_type is not None:
        return mime_type
    if head_is_not_a_picture(head):
        return None
    return canonical_image_mime(declared.split(";", 1)[0].strip().lower())


def _not_a_picture_reason(byte_count: int, declared: str, head: bytes) -> str:
    return clamp_text(
        f"{byte_count} byte(s) declared "
        f"{clamp_text(declared or 'none', 40)!r} are not a recognised "
        f"image (starts {head[:16]!r})"
    )


def _own_decoded(blob: str) -> int:
    length = len(blob)
    if length < 4:
        return 0
    pad = blob[-4:].count("=")
    return (length // 4) * 3 - pad


_IMAGE_BODY_EXCERPT_CHARS = 200


def _image_body_not_an_object(resp: Any, payload: Any) -> UpstreamBodyUnreadable:
    return UpstreamBodyUnreadable(
        endpoint="/images",
        body_excerpt=repr(payload)[:_IMAGE_BODY_EXCERPT_CHARS],
        content_type=getattr(resp, "content_type", None),
    )


_IMAGE_BODY_CHUNK_BYTES = 1 << 20

_IMAGE_LINE_SCAN_CHARS = 256 * 1024

_IMAGE_LINE_MEDIA_HINTS = ('"b64_json"', '"base64"', "data:image/")


async def _decode_image_json_body(
    resp: Any, logger: Any, spool: ReceiveSpool | None = None
) -> Any:
    scanner = spool.scanner() if spool is not None else None
    body: bytes | bytearray = b""
    parts: dict[str, PartRef] = {}
    try:
        if scanner is None:
            body = await resp.read()
        else:
            async for chunk in resp.content.iter_chunked(_IMAGE_BODY_CHUNK_BYTES):
                scanner.feed(chunk)
            body, parts = scanner.finish()
    except (aiohttp.ClientPayloadError, aiohttp.ServerDisconnectedError):
        raise
    except Exception as exc:
        raise UpstreamBodyUnreadable(
            endpoint="/images",
            body_excerpt="",
            content_type=getattr(resp, "content_type", None),
        ) from exc
    if not body or body.isspace():
        return None
    try:
        parsed = json.loads(body)
    except (ValueError, RecursionError) as exc:
        logger.debug("OpenRouter image response was not decodable JSON", exc_info=True)
        raise UpstreamBodyUnreadable(
            endpoint="/images",
            body_excerpt=bytes(body[:_IMAGE_BODY_EXCERPT_CHARS]).decode("utf-8", "replace"),
            content_type=getattr(resp, "content_type", None),
        ) from exc
    return attach_parts(parsed, parts)


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
        state["stream_broken"] = True
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
        self._spool: ReceiveSpool | None = None

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
            raise ImageGenerationError("Image endpoint response was not an object.")
        records = payload.get("endpoints")
        if not isinstance(records, list):
            raise ImageGenerationError("Image endpoint response carried no endpoint list.")
        return [item for item in records if isinstance(item, dict)]

    def _parse_stream_event(self, payload: str) -> Any:
        scan = (
            len(payload) > _IMAGE_LINE_SCAN_CHARS
            or any(hint in payload for hint in _IMAGE_LINE_MEDIA_HINTS)
        )
        if not scan or self._spool is None:
            return None
        try:
            line_scan = self._spool.scanner()
            line_scan.feed(payload.encode("utf-8"))
            slim, parts = line_scan.finish()
            return attach_parts(json.loads(slim), parts)
        except (ValueError, RecursionError):
            return None

    async def _consume_stream_line(
        self, raw_line: str, state: dict[str, Any], on_progress: Any
    ) -> None:
        line = (raw_line or "").strip()
        if not line.startswith(_IMAGE_SSE_PREFIX):
            return
        payload = line[len(_IMAGE_SSE_PREFIX) :].strip()
        if not payload or payload == _IMAGE_SSE_DONE:
            return
        event = self._parse_stream_event(payload)
        if event is None:
            try:
                event = json.loads(payload)
            except ValueError:
                self._logger.debug("Image stream chunk was not readable JSON", exc_info=True)
                return
        await self._dispatch_stream_event(event, state, on_progress)

    async def _dispatch_stream_event(
        self, event: Any, state: dict[str, Any], on_progress: Any
    ) -> None:
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

    async def _consume_scanned_line(
        self, scan: Any, state: dict[str, Any], on_progress: Any
    ) -> None:
        slim, parts = scan.finish()
        try:
            event = attach_parts(json.loads(slim), parts)
        except (ValueError, RecursionError, UnicodeDecodeError):
            self._logger.debug(
                "Image stream event parse failed after its streamed scan", exc_info=True
            )
            return
        await self._dispatch_stream_event(event, state, on_progress)

    async def _read_stream_as_buffered_response(
        self,
        resp: Any,
        on_progress: Any,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        buffer = ""
        line_scan = None
        _utf8 = utf8_stream_decoder()
        try:
            async for chunk in resp.content.iter_any():
                piece = _utf8.decode(chunk)
                if line_scan is not None:
                    scan = line_scan
                    newline_at = piece.find("\n")
                    if newline_at < 0:
                        scan.feed(piece.encode("utf-8"))
                        continue
                    scan.feed(piece[:newline_at].encode("utf-8"))
                    line_scan = None
                    await self._consume_scanned_line(scan, state, on_progress)
                    piece = piece[newline_at + 1 :]
                    if not piece:
                        continue
                buffer += piece
                if "\n" not in piece:
                    if (
                        line_scan is None
                        and self._spool is not None
                        and len(buffer) > _IMAGE_LINE_SCAN_CHARS
                        and buffer.startswith(_IMAGE_SSE_PREFIX)
                    ):
                        line_scan = self._spool.scanner()
                        line_scan.feed(buffer[len(_IMAGE_SSE_PREFIX) :].lstrip().encode("utf-8"))
                        buffer = ""
                    continue
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    await self._consume_stream_line(line, state, on_progress)
                if (
                    line_scan is None
                    and self._spool is not None
                    and len(buffer) > _IMAGE_LINE_SCAN_CHARS
                    and buffer.startswith(_IMAGE_SSE_PREFIX)
                ):
                    line_scan = self._spool.scanner()
                    line_scan.feed(buffer[len(_IMAGE_SSE_PREFIX) :].lstrip().encode("utf-8"))
                    buffer = ""
            if line_scan is not None:
                await self._consume_scanned_line(line_scan, state, on_progress)
                line_scan = None
            buffer += _utf8.decode(b"", True)
            await self._consume_stream_line(buffer, state, on_progress)
        except _ProgressCallbackFailed as failed:
            if not state["data"]:
                raise failed.cause from failed.cause
            self._logger.warning(
                "The pipe's own image progress callback failed after the finished image "
                "arrived; delivering it anyway: %s",
                clamp_text(str(failed.cause) or type(failed.cause).__name__, 160),
            )
            if not state["warning"]:
                state["warning"] = (
                    "The image progress channel failed after the finished image had "
                    f"already arrived: {clamp_text(str(failed.cause), 160)}."
                )
        except ImageGenerationError:
            raise
        except Exception as exc:
            billed = chat_usage_to_responses_usage(state.get("usage"))
            if state["data"]:
                self._logger.warning(
                    "OpenRouter's image stream for this request stopped after the finished "
                    "image arrived: %s", clamp_text(str(exc) or type(exc).__name__, 160)
                )
                state["stream_broken"] = True
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
        self, payload: dict[str, Any], *, spool: ReceiveSpool | None = None, on_progress: Any = None
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
                "stream_broken": False,
            }
            self._spool = spool
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
                    state["stream_broken"] = True
                    if not state["warning"]:
                        state["warning"] = (
                            "The image stream failed after the finished image had "
                            f"already arrived: {clamp_text(str(exc), 160)}."
                        )
                    data = {"data": state["data"], "usage": state["usage"]}
            else:
                data = await _decode_image_json_body(resp, self._logger, spool)
        _debug_print_response(data, logger=self._logger)
        if not isinstance(data, dict):
            raise _image_body_not_an_object(resp, data)

        billed = chat_usage_to_responses_usage(data.get("usage"))

        entries = data.get("data")
        if not isinstance(entries, list) or not entries:
            raise ImageGenerationError(IMAGE_NO_IMAGES_REASON, usage=billed)

        images: list[GeneratedImage] = []
        rejected: list[str] = []
        over_guard_count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                rejected.append(clamp_text(f"an entry of type {type(entry).__name__} carried no image"))
                continue
            blob = entry.get("b64_json")
            raw_media_type = entry.get("media_type")
            declared = raw_media_type if isinstance(raw_media_type, str) else ""
            if isinstance(blob, PartRef):
                if blob.broken:
                    rejected.append(clamp_text("an entry was not decodable base64"))
                    continue
                if blob.over_guard:
                    over_guard_count += 1
                    rejected.append(clamp_text(blob.refusal()))
                    continue
                if not blob.exists():
                    rejected.append(clamp_text("an entry's received image was no longer available"))
                    continue
                try:
                    head = blob.read_head(_IMAGE_DECODE_PREFIX_BYTES)
                except OSError:
                    rejected.append(clamp_text("an entry's received image could not be read"))
                    continue
                mime_type = _classify(head, declared) if head else None
                if mime_type is None:
                    rejected.append(_not_a_picture_reason(blob.decoded_bytes, declared, head))
                    continue
                images.append(
                    GeneratedImage(data=b"", mime_type=mime_type, source_path=Path(blob.path))
                )
                continue
            if not isinstance(blob, str) or not blob:
                rejected.append(clamp_text(f"an entry with keys {sorted(entry)} carried no inline base64"))
                continue
            probe = _decoded_prefix(blob)
            if probe is not None and _classify(probe, declared) is None:
                rejected.append(_not_a_picture_reason(_own_decoded(blob), declared, probe))
                continue
            try:
                raw = await _decode_base64_in_quanta(blob)
            except (binascii.Error, ValueError):
                rejected.append(clamp_text(f"an entry starting {blob[:24]!r} was not decodable base64"))
                continue
            mime_type = _classify(raw, declared)
            if mime_type is None:
                rejected.append(_not_a_picture_reason(len(raw), declared, raw))
                continue
            images.append(GeneratedImage(data=raw, mime_type=mime_type))

        if not images:
            detail = f" ({summarise_names(rejected)})" if rejected else ""
            raise ImageGenerationError(
                f"OpenRouter image generation returned no usable images{detail}.", usage=billed
            )

        return ImageGenerationResult(
            images=images, usage=billed, rejected=rejected, warning=state.get("warning", ""),
            over_ceiling=over_guard_count, over_ceiling_own=over_guard_count,
            stream_broken=bool(state.get("stream_broken")),
        )
