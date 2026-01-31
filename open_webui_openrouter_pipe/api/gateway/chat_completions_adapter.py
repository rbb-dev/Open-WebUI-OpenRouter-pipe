"""Chat Completions API adapter for OpenRouter.

This module handles Chat Completions API streaming and non-streaming requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Optional

import aiohttp
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...core.config import _OPENROUTER_TITLE, _select_openrouter_http_referer
from ...core.timing_logger import timed, timing_scope, timing_mark
from ...requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
    _debug_print_response,
)
# Imports from core.errors
from ...core.errors import (
    _build_openrouter_api_error,
)
# Imports from api.transforms
from ..transforms import (
    _filter_openrouter_chat_request,
    _filter_openrouter_request,
    _responses_payload_to_chat_completions_payload,
)
# Imports from storage
from ...storage.multimodal import (
    _is_internal_file_url,
)
from ...storage.persistence import generate_item_id
from ...models.registry import normalize_model_id_dotted

if TYPE_CHECKING:
    from ...pipe import Pipe


class ChatCompletionsAdapter:
    """Adapter for OpenRouter /chat/completions API endpoint."""

    @timed
    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize ChatCompletionsAdapter.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging
        """
        self._pipe = pipe
        self.logger = logger

    @timed
    async def send_openai_chat_completions_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        breaker_key: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Send /chat/completions and adapt streaming output into Responses-style events."""
        effective_valves = valves or self._pipe.valves
        chat_payload = _responses_payload_to_chat_completions_payload(
            dict(responses_request_body or {}),
        )
        chat_payload = _filter_openrouter_chat_request(chat_payload)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _select_openrouter_http_referer(effective_valves),
        }
        self._pipe._maybe_apply_anthropic_beta_headers(
            headers,
            chat_payload.get("model"),
            valves=effective_valves,
        )
        _debug_print_request(headers, chat_payload, logger=self.logger)
        url = base_url.rstrip("/") + "/chat/completions"

        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        tool_call_added: set[int] = set()
        tool_calls_completed = False
        assistant_text_parts: list[str] = []
        latest_usage: dict[str, Any] = {}
        seen_citation_urls: set[str] = set()
        latest_message_annotations: list[dict[str, Any]] = []
        reasoning_item_id: str | None = None
        reasoning_text_parts: list[str] = []
        reasoning_summary_text: str | None = None
        reasoning_details_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        reasoning_details_order: list[tuple[str, str]] = []

        @timed
        def _ensure_tool_call_id(index: int, current: dict[str, Any]) -> str:
            tid = current.get("id")
            if isinstance(tid, str) and tid.strip():
                return tid.strip()
            model_val = (chat_payload.get("model") or "model")
            generated = f"toolcall-{normalize_model_id_dotted(str(model_val))}-{index}"
            current["id"] = generated
            return generated

        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            reraise=True,
        )

        @timed
        async def _inline_internal_chat_files() -> None:
            messages = chat_payload.get("messages")
            if not isinstance(messages, list) or not messages:
                return
            chunk_size = getattr(effective_valves, "IMAGE_UPLOAD_CHUNK_BYTES", self._pipe.valves.IMAGE_UPLOAD_CHUNK_BYTES)
            max_bytes = int(getattr(effective_valves, "BASE64_MAX_SIZE_MB", self._pipe.valves.BASE64_MAX_SIZE_MB)) * 1024 * 1024
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list) or not content:
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "file":
                        continue
                    file_obj = block.get("file")
                    if not isinstance(file_obj, dict):
                        continue
                    file_value = file_obj.get("file_data")
                    if not isinstance(file_value, str) or not file_value.strip():
                        file_value = file_obj.get("file_url")
                    if not isinstance(file_value, str) or not file_value.strip():
                        continue
                    file_value = file_value.strip()
                    if not _is_internal_file_url(file_value):
                        continue
                    inlined = await self._pipe._inline_internal_file_url(
                        file_value,
                        chunk_size=chunk_size,
                        max_bytes=max_bytes,
                    )
                    if not inlined:
                        raise ValueError(f"Failed to inline Open WebUI file URL for /chat/completions: {file_value}")
                    file_obj["file_data"] = inlined

        @timed
        def _record_reasoning_detail(detail: dict[str, Any]) -> None:
            dtype = detail.get("type")
            if not isinstance(dtype, str) or not dtype:
                return
            did = detail.get("id")
            if not isinstance(did, str) or not did.strip():
                did = f"{dtype}:{len(reasoning_details_order)}"
            key = (dtype, did)
            existing = reasoning_details_by_key.get(key)
            if existing is None:
                reasoning_details_order.append(key)
                reasoning_details_by_key[key] = dict(detail)
                return
            merged = dict(existing)
            if dtype == "reasoning.text":
                prev_text = merged.get("text")
                next_text = detail.get("text")
                if isinstance(prev_text, str) and isinstance(next_text, str):
                    merged["text"] = f"{prev_text}{next_text}"
                elif isinstance(next_text, str):
                    merged["text"] = next_text
            elif dtype == "reasoning.summary":
                next_summary = detail.get("summary")
                if isinstance(next_summary, str) and next_summary.strip():
                    merged["summary"] = next_summary
            elif dtype == "reasoning.encrypted":
                prev_data = merged.get("data")
                next_data = detail.get("data")
                if isinstance(prev_data, str) and isinstance(next_data, str):
                    merged["data"] = f"{prev_data}{next_data}"
                elif isinstance(next_data, str):
                    merged["data"] = next_data
            for k, v in detail.items():
                if k in merged:
                    continue
                merged[k] = v
            reasoning_details_by_key[key] = merged

        @timed
        def _final_reasoning_details() -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for key in reasoning_details_order:
                detail = reasoning_details_by_key.get(key)
                if isinstance(detail, dict) and detail:
                    out.append(dict(detail))
            return out

        first_chunk_received = False
        async for attempt in retryer:
            with attempt:
                if breaker_key and not self._pipe._breaker_allows(breaker_key):
                    raise RuntimeError("Breaker open for user during stream")

                await _inline_internal_chat_files()

                timing_mark("chat_http_request_start")
                async with session.post(url, json=chat_payload, headers=headers) as resp:
                    timing_mark("chat_http_headers_received")
                    if resp.status >= 400:
                        error_body = await _debug_print_error_response(resp, logger=self.logger)
                        if breaker_key:
                            self._pipe._record_failure(breaker_key)
                        special_statuses = {400, 401, 402, 403, 404, 408, 429}
                        if resp.status in special_statuses:
                            extra_meta: dict[str, Any] = {}
                            retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                            if retry_after:
                                extra_meta["retry_after"] = retry_after
                                extra_meta["retry_after_seconds"] = retry_after
                            rate_scope = (
                                resp.headers.get("X-RateLimit-Scope")
                                or resp.headers.get("x-ratelimit-scope")
                            )
                            if rate_scope:
                                extra_meta["rate_limit_type"] = rate_scope
                            reason_text = resp.reason or "HTTP error"
                            raise _build_openrouter_api_error(
                                resp.status,
                                reason_text,
                                error_body,
                                requested_model=chat_payload.get("model"),
                                extra_metadata=extra_meta or None,
                            )
                        raise RuntimeError(f"OpenRouter request failed ({resp.status}): {resp.reason}")

                    resp.raise_for_status()

                    buf = bytearray()
                    event_data_parts: list[bytes] = []
                    done = False

                    async for chunk in resp.content.iter_any():
                        if not chunk:
                            continue
                        if not first_chunk_received:
                            first_chunk_received = True
                            timing_mark("chat_first_chunk")
                        buf.extend(chunk)
                        start_idx = 0
                        while True:
                            newline_idx = buf.find(b"\n", start_idx)
                            if newline_idx == -1:
                                break
                            line = buf[start_idx:newline_idx]
                            start_idx = newline_idx + 1
                            stripped = line.strip()

                            if not stripped:
                                if not event_data_parts:
                                    continue
                                data_blob = b"\n".join(event_data_parts).strip()
                                event_data_parts.clear()
                                if not data_blob:
                                    continue
                                if data_blob == b"[DONE]":
                                    done = True
                                    timing_mark("chat_stream_done")
                                    break
                                try:
                                    chunk_obj = json.loads(data_blob.decode("utf-8"))
                                except json.JSONDecodeError:
                                    continue

                                if isinstance(chunk_obj, dict) and isinstance(chunk_obj.get("usage"), dict):
                                    latest_usage = dict(chunk_obj["usage"])

                                choices = chunk_obj.get("choices") if isinstance(chunk_obj, dict) else None
                                if not isinstance(choices, list) or not choices:
                                    continue
                                choice0 = choices[0] if isinstance(choices[0], dict) else {}
                                delta = choice0.get("delta") if isinstance(choice0, dict) else None
                                delta_obj = delta if isinstance(delta, dict) else {}

                                delta_reasoning_details = delta_obj.get("reasoning_details")
                                if isinstance(delta_reasoning_details, list) and delta_reasoning_details:
                                    for entry in delta_reasoning_details:
                                        if not isinstance(entry, dict):
                                            continue
                                        _record_reasoning_detail(entry)
                                        rtype = entry.get("type")
                                        if not isinstance(rtype, str) or not rtype:
                                            continue
                                        if reasoning_item_id is None:
                                            candidate_id = entry.get("id")
                                            if isinstance(candidate_id, str) and candidate_id.strip():
                                                reasoning_item_id = candidate_id.strip()
                                            else:
                                                reasoning_item_id = f"reasoning-{generate_item_id()}"
                                            yield {
                                                "type": "response.output_item.added",
                                                "item": {
                                                    "type": "reasoning",
                                                    "id": reasoning_item_id,
                                                    "status": "in_progress",
                                                },
                                            }
                                        if rtype == "reasoning.text":
                                            text = entry.get("text")
                                            if isinstance(text, str) and text:
                                                reasoning_text_parts.append(text)
                                                yield {
                                                    "type": "response.reasoning_text.delta",
                                                    "item_id": reasoning_item_id,
                                                    "delta": text,
                                                }
                                        elif rtype == "reasoning.summary":
                                            summary = entry.get("summary")
                                            if isinstance(summary, str) and summary.strip():
                                                reasoning_summary_text = summary.strip()
                                                yield {
                                                    "type": "response.reasoning_summary_text.done",
                                                    "item_id": reasoning_item_id,
                                                    "text": reasoning_summary_text,
                                                }

                                annotations: list[Any] = []
                                delta_annotations = delta_obj.get("annotations")
                                if isinstance(delta_annotations, list) and delta_annotations:
                                    annotations.extend(delta_annotations)
                                message_obj = choice0.get("message") if isinstance(choice0, dict) else None
                                if isinstance(message_obj, dict):
                                    message_annotations = message_obj.get("annotations")
                                    if isinstance(message_annotations, list) and message_annotations:
                                        annotations.extend(message_annotations)
                                        latest_message_annotations = [
                                            dict(a) for a in message_annotations if isinstance(a, dict)
                                        ]
                                    message_reasoning_details = message_obj.get("reasoning_details")
                                    if isinstance(message_reasoning_details, list) and message_reasoning_details:
                                        for entry in message_reasoning_details:
                                            if isinstance(entry, dict):
                                                _record_reasoning_detail(entry)

                                if annotations:
                                    for raw_ann in annotations:
                                        if not isinstance(raw_ann, dict):
                                            continue
                                        if raw_ann.get("type") != "url_citation":
                                            continue
                                        payload = raw_ann.get("url_citation")
                                        if isinstance(payload, dict):
                                            url = payload.get("url")
                                            title = payload.get("title") or url
                                        else:
                                            url = raw_ann.get("url")
                                            title = raw_ann.get("title") or url
                                        if not isinstance(url, str) or not url.strip():
                                            continue
                                        url = url.strip()
                                        if url in seen_citation_urls:
                                            continue
                                        seen_citation_urls.add(url)
                                        if isinstance(title, str):
                                            title = title.strip() or url
                                        else:
                                            title = url
                                        yield {
                                            "type": "response.output_text.annotation.added",
                                            "annotation": {"type": "url_citation", "url": url, "title": title},
                                        }

                                content_delta = delta_obj.get("content")
                                if isinstance(content_delta, str) and content_delta:
                                    assistant_text_parts.append(content_delta)
                                    yield {"type": "response.output_text.delta", "delta": content_delta}

                                tool_calls = delta_obj.get("tool_calls")
                                if isinstance(tool_calls, list) and tool_calls:
                                    for raw_call in tool_calls:
                                        if not isinstance(raw_call, dict):
                                            continue
                                        index = raw_call.get("index")
                                        if not isinstance(index, int):
                                            index = max(tool_calls_by_index.keys(), default=-1) + 1
                                        current = tool_calls_by_index.setdefault(index, {})
                                        if isinstance(raw_call.get("id"), str):
                                            current["id"] = raw_call["id"]
                                        function = raw_call.get("function")
                                        if isinstance(function, dict):
                                            name = function.get("name")
                                            if isinstance(name, str) and name:
                                                current["name"] = name
                                            args_delta = function.get("arguments")
                                            if isinstance(args_delta, str) and args_delta:
                                                existing = current.get("arguments") or ""
                                                current["arguments"] = f"{existing}{args_delta}"

                                        if index not in tool_call_added:
                                            tool_call_added.add(index)
                                            call_id = _ensure_tool_call_id(index, current)
                                            yield {
                                                "type": "response.output_item.added",
                                                "item": {
                                                    "type": "function_call",
                                                    "id": call_id,
                                                    "call_id": call_id,
                                                    "status": "in_progress",
                                                    "name": current.get("name") or "",
                                                    "arguments": current.get("arguments") or "",
                                                },
                                            }

                                finish_reason = choice0.get("finish_reason") if isinstance(choice0, dict) else None
                                if isinstance(finish_reason, str) and finish_reason:
                                    if finish_reason == "tool_calls":
                                        tool_calls_completed = True
                                    elif finish_reason in {"stop", "length", "content_filter"}:
                                        tool_calls_completed = True

                            if stripped.startswith(b":"):
                                continue
                            if stripped.startswith(b"data:"):
                                event_data_parts.append(bytes(stripped[5:].lstrip()))
                                continue

                        if start_idx > 0:
                            del buf[:start_idx]
                        if done:
                            break
                    break

        if tool_calls_by_index and tool_calls_completed:
            for index in sorted(tool_calls_by_index.keys()):
                current = tool_calls_by_index[index]
                call_id = _ensure_tool_call_id(index, current)
                yield {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "status": "completed",
                        "name": current.get("name") or "",
                        "arguments": current.get("arguments") or "",
                    },
                }

        if reasoning_item_id is not None:
            reasoning_text = "".join(reasoning_text_parts).strip()
            reasoning_item: dict[str, Any] = {
                "type": "reasoning",
                "id": reasoning_item_id,
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": reasoning_text}] if reasoning_text else [],
                "summary": [{"type": "summary_text", "text": reasoning_summary_text}] if reasoning_summary_text else [],
            }
            yield {
                "type": "response.output_item.done",
                "item": reasoning_item,
            }

        assistant_text = "".join(assistant_text_parts)
        message_item: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": assistant_text}],
        }
        if latest_message_annotations:
            message_item["annotations"] = latest_message_annotations
        final_reasoning_details = _final_reasoning_details()
        if final_reasoning_details:
            message_item["reasoning_details"] = final_reasoning_details
        output: list[dict[str, Any]] = [message_item]
        for index in sorted(tool_calls_by_index.keys()):
            current = tool_calls_by_index[index]
            call_id = _ensure_tool_call_id(index, current)
            name = current.get("name")
            if not isinstance(name, str) or not name:
                continue
            output.append(
                {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": current.get("arguments") or "{}",
                }
            )

        yield {
            "type": "response.completed",
            "response": {
                "output": output,
                "usage": ChatCompletionsAdapter._chat_usage_to_responses_usage(latest_usage),
            },
        }


    # ======================================================================
    # send_openai_chat_completions_nonstreaming_request (122 lines)
    # ======================================================================

    @timed
    async def send_openai_chat_completions_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        breaker_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Send /chat/completions with stream=false and return the JSON payload."""
        effective_valves = valves or self._pipe.valves
        chat_payload = _responses_payload_to_chat_completions_payload(
            dict(responses_request_body or {}),
        )
        chat_payload = _filter_openrouter_chat_request(chat_payload)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _select_openrouter_http_referer(effective_valves),
        }
        self._pipe._maybe_apply_anthropic_beta_headers(
            headers,
            chat_payload.get("model"),
            valves=effective_valves,
        )
        _debug_print_request(headers, chat_payload, logger=self.logger)
        url = base_url.rstrip("/") + "/chat/completions"

        @timed
        async def _inline_internal_chat_files() -> None:
            messages = chat_payload.get("messages")
            if not isinstance(messages, list) or not messages:
                return
            chunk_size = getattr(effective_valves, "IMAGE_UPLOAD_CHUNK_BYTES", self._pipe.valves.IMAGE_UPLOAD_CHUNK_BYTES)
            max_bytes = int(getattr(effective_valves, "BASE64_MAX_SIZE_MB", self._pipe.valves.BASE64_MAX_SIZE_MB)) * 1024 * 1024
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list) or not content:
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "file":
                        continue
                    file_obj = block.get("file")
                    if not isinstance(file_obj, dict):
                        continue
                    file_value = file_obj.get("file_data")
                    if not isinstance(file_value, str) or not file_value.strip():
                        file_value = file_obj.get("file_url")
                    if not isinstance(file_value, str) or not file_value.strip():
                        continue
                    file_value = file_value.strip()
                    if not _is_internal_file_url(file_value):
                        continue
                    inlined = await self._pipe._inline_internal_file_url(
                        file_value,
                        chunk_size=chunk_size,
                        max_bytes=max_bytes,
                    )
                    if not inlined:
                        raise ValueError(f"Failed to inline Open WebUI file URL for /chat/completions: {file_value}")
                    file_obj["file_data"] = inlined

        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                if breaker_key and not self._pipe._breaker_allows(breaker_key):
                    raise RuntimeError("Breaker open for user during request")

                await _inline_internal_chat_files()

                timing_mark("chat_nonstream_http_request_start")
                async with session.post(url, json=chat_payload, headers=headers) as resp:
                    timing_mark("chat_nonstream_http_response")
                    if resp.status >= 400:
                        error_body = await _debug_print_error_response(resp, logger=self.logger)
                        if breaker_key:
                            self._pipe._record_failure(breaker_key)
                        special_statuses = {400, 401, 402, 403, 404, 408, 429}
                        if resp.status in special_statuses:
                            extra_meta: dict[str, Any] = {}
                            retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                            if retry_after:
                                extra_meta["retry_after"] = retry_after
                                extra_meta["retry_after_seconds"] = retry_after
                            rate_scope = (
                                resp.headers.get("X-RateLimit-Scope")
                                or resp.headers.get("x-ratelimit-scope")
                            )
                            if rate_scope:
                                extra_meta["rate_limit_type"] = rate_scope
                            reason_text = resp.reason or "HTTP error"
                            raise _build_openrouter_api_error(
                                resp.status,
                                reason_text,
                                error_body,
                                requested_model=chat_payload.get("model"),
                                extra_metadata=extra_meta or None,
                            )
                        raise RuntimeError(f"OpenRouter request failed ({resp.status}): {resp.reason}")
                    resp.raise_for_status()
                    try:
                        data = await resp.json()
                    except Exception:
                        text = await resp.text()
                        try:
                            data = json.loads(text)
                        except Exception as exc:
                            raise RuntimeError("Invalid JSON response from /chat/completions") from exc
                    if isinstance(data, dict):
                        _debug_print_response(data, logger=self.logger)
                        return data
                    _debug_print_response(data, logger=self.logger)
                    return {}

        return {}


    # ======================================================================
    # send_openrouter_nonstreaming_request_as_events (242 lines)
    # ======================================================================

    @timed
    async def send_openrouter_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        workers: int = 4,
        breaker_key: Optional[str] = None,
        delta_char_limit: int = 0,
        idle_flush_ms: int = 0,
        chunk_queue_maxsize: int = 100,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Unified streaming request entrypoint with endpoint routing + fallback."""
        effective_valves = valves or self._pipe.valves
        model_id = (responses_request_body or {}).get("model") or ""
        endpoint = endpoint_override or self._pipe._select_llm_endpoint(str(model_id), valves=effective_valves)

        @timed
        def _responses_event_is_user_visible(event: dict[str, Any]) -> bool:
            etype = event.get("type")
            if not isinstance(etype, str) or not etype:
                return True
            if etype.startswith("response.output_"):
                return True
            if etype.startswith("response.content_part"):
                return True
            if etype.startswith("response.reasoning"):
                return True
            if etype in {"response.completed", "response.failed", "response.error", "error"}:
                return True
            return False

        responses_emitted_user_visible = False
        responses_buffer: list[dict[str, Any]] = []

        @timed
        async def _run_responses() -> AsyncGenerator[dict[str, Any], None]:
            nonlocal responses_emitted_user_visible
            request_payload = _filter_openrouter_request(dict(responses_request_body or {}))
            async for event in self._pipe.send_openai_responses_streaming_request(
                session,
                request_payload,
                api_key=api_key,
                base_url=base_url,
                valves=effective_valves,
                workers=workers,
                breaker_key=breaker_key,
                delta_char_limit=delta_char_limit,
                idle_flush_ms=idle_flush_ms,
                chunk_queue_maxsize=chunk_queue_maxsize,
                event_queue_maxsize=event_queue_maxsize,
                event_queue_warn_size=event_queue_warn_size,
            ):
                if not responses_emitted_user_visible and not _responses_event_is_user_visible(event):
                    responses_buffer.append(event)
                    continue
                if not responses_emitted_user_visible:
                    responses_emitted_user_visible = True
                    for pending in responses_buffer:
                        yield pending
                    responses_buffer.clear()
                yield event

        @timed
        async def _run_chat() -> AsyncGenerator[dict[str, Any], None]:
            async for event in self._pipe.send_openai_chat_completions_streaming_request(
                session,
                dict(responses_request_body or {}),
                api_key=api_key,
                base_url=base_url,
                valves=effective_valves,
                breaker_key=breaker_key,
            ):
                yield event

        if endpoint == "chat_completions":
            async for event in _run_chat():
                yield event
            return

        try:
            async for event in _run_responses():
                yield event
        except Exception as exc:
            if getattr(effective_valves, "AUTO_FALLBACK_CHAT_COMPLETIONS", True) and self._pipe._looks_like_responses_unsupported(exc):
                if responses_emitted_user_visible:
                    self.logger.info(
                        "Not falling back to /chat/completions for model=%s: /responses already emitted user-visible output before error: %s",
                        model_id,
                        exc,
                    )
                    raise
                if responses_buffer:
                    self.logger.debug(
                        "Discarding %d non-visible /responses events prior to fallback for model=%s",
                        len(responses_buffer),
                        model_id,
                    )
                    responses_buffer.clear()
                self.logger.info(
                    "Falling back to /chat/completions for model=%s after /responses error (status=%s openrouter_code=%s): %s",
                    model_id,
                    getattr(exc, "status", None),
                    getattr(exc, "openrouter_code", None),
                    exc,
                )
                async for event in _run_chat():
                    yield event
                return
            raise


    # ======================================================================
    # Tool Context Shutdown
    # ======================================================================

    @staticmethod
    @timed
    def _chat_usage_to_responses_usage(raw_usage: Any) -> dict[str, Any]:
        """Normalise Chat Completions usage counters into the Responses-style keys used by this pipe."""
        if not isinstance(raw_usage, dict):
            return {}
        usage: dict[str, Any] = {}

        prompt_tokens = raw_usage.get("prompt_tokens")
        completion_tokens = raw_usage.get("completion_tokens")
        total_tokens = raw_usage.get("total_tokens")
        if prompt_tokens is not None:
            usage["input_tokens"] = prompt_tokens
        if completion_tokens is not None:
            usage["output_tokens"] = completion_tokens
        if total_tokens is not None:
            usage["total_tokens"] = total_tokens

        for key in ("cost", "cache_discount", "cache_discount_pct"):
            if key in raw_usage:
                usage[key] = raw_usage[key]

        cost_details = raw_usage.get("cost_details")
        if isinstance(cost_details, dict) and cost_details:
            usage["cost_details"] = dict(cost_details)

        prompt_details = raw_usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict) and prompt_details:
            usage["input_tokens_details"] = dict(prompt_details)

        completion_details = raw_usage.get("completion_tokens_details")
        if isinstance(completion_details, dict) and completion_details:
            usage["output_tokens_details"] = dict(completion_details)

        return usage
