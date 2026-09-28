"""Chat Completions API adapter for OpenRouter.

This module handles Chat Completions API streaming and non-streaming requests.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal

import aiohttp
from tenacity import (
    stop_after_attempt,
)

from ...core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_TITLE,
    _apply_owui_forward_user_headers,
    _select_openrouter_http_referer,
)

# Imports from core.errors
from ...core.costs import chat_usage_to_responses_usage
from ...core.errors import (
    RequiredInternalFileError,
    _build_openrouter_api_error,
)
from ...core.timing_logger import timed, timing_mark
from ...core.utils import _apply_retry_after_metadata, http_timeout
from ...core.warn_latch import warn_level
from ...requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
    _debug_print_response,
)

# Imports from storage
from ...storage.owui_files import (
    is_internal_file_url,
)
from ...storage.persistence import generate_item_id
from ...streaming.nagle_coalescer import nagle_coalesce_stream

# Imports from api.transforms
from ..transforms import (
    _filter_openrouter_chat_request,
    _filter_openrouter_request,
    _parse_url_citation_annotations,
    _responses_payload_to_chat_completions_payload,
)
from .responses_adapter import (
    _body_not_an_object,
    _count_failed_call,
    _decode_json_body,
    _record_failed_call,
    _responses_event_is_user_visible,
    _retry_nonstreaming,
    _should_retry_stream,
    _split_sse_lines,
    _transient_retry_policy,
)

if TYPE_CHECKING:
    from ...pipe import Pipe

_CHAT_CHUNK_PARSE_WARN_COOLDOWN_S = 30.0
_CHAT_SSE_DONE_SENTINEL = b"[DONE]"
_warned_chat_chunk_parse: dict[str, float] = {}


def _refusal_split(answer: str, refusal: str) -> tuple[str | None, str]:
    refusal = refusal.strip()
    if not refusal:
        return None, answer
    if not answer:
        return refusal, refusal
    return f"\n\n{refusal}", f"{answer}\n\n{refusal}"


def _append_text_field(item: dict, key: str, value: str) -> None:
    text = item[key] if type(item.get(key)) is str else ""
    item[key] = ""
    text += value
    item[key] = text


def _build_output_items(
    *,
    assistant_text: str,
    annotations: list[dict[str, Any]] | None,
    reasoning_details: list[dict[str, Any]] | None,
    image_output_item: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if assistant_text or annotations or reasoning_details:
        message_item: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": assistant_text}],
        }
        if annotations:
            message_item["annotations"] = annotations
        if reasoning_details:
            message_item["reasoning_details"] = reasoning_details
        output.append(message_item)
    if image_output_item is not None:
        output.append(image_output_item)
    output.extend(tool_calls)
    return output


class ChatCompletionsAdapter:
    """Adapter for OpenRouter /chat/completions API endpoint."""

    @timed
    def __init__(self, pipe: Pipe, logger: logging.Logger):
        """Initialize ChatCompletionsAdapter.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging
        """
        self._pipe = pipe
        self.logger = logger

    def _timeout(self, effective_valves: Any) -> aiohttp.ClientTimeout:
        return http_timeout(effective_valves)

    @timed
    async def _inline_internal_chat_files(self, chat_payload: dict, effective_valves: Any, *, user: Any = None) -> None:
        """Inline OWUI internal file URLs in chat messages before sending to OpenRouter."""
        messages = chat_payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return
        chunk_size = effective_valves.IMAGE_UPLOAD_CHUNK_BYTES
        max_bytes = effective_valves.BASE64_MAX_SIZE_MB * 1024 * 1024
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
                if not is_internal_file_url(file_value):
                    continue
                try:
                    inlined = await self._pipe._file_gateway.inline_internal_file_url(
                        file_value, chunk_size=chunk_size, max_bytes=max_bytes, user=user,
                    )
                except RequiredInternalFileError:
                    raise
                except Exception as exc:
                    raise RequiredInternalFileError(
                        "A referenced file could not be prepared for the provider.", kind="file",
                    ) from exc
                if not inlined:
                    raise RequiredInternalFileError(
                        "A referenced file could not be prepared for the provider.", kind="file",
                    )
                file_obj["file_data"] = inlined.data_url
                file_obj.pop("file_url", None)
                if inlined.filename and "filename" not in file_obj:
                    file_obj["filename"] = inlined.filename

    @timed
    async def send_openai_chat_completions_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        breaker_key: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Send /chat/completions and adapt streaming output into Responses-style events."""
        effective_valves = valves or self._pipe.valves
        responses_payload = dict(responses_request_body or {})
        await self._pipe._file_gateway.inline_internal_responses_input_files_inplace(
            responses_payload,
            chunk_size=effective_valves.IMAGE_UPLOAD_CHUNK_BYTES,
            max_bytes=effective_valves.BASE64_MAX_SIZE_MB * 1024 * 1024,
            user=user,
        )
        chat_payload = _responses_payload_to_chat_completions_payload(
            responses_payload,
        )
        chat_payload = _filter_openrouter_chat_request(chat_payload)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            "HTTP-Referer": _select_openrouter_http_referer(effective_valves),
        }
        self._pipe._maybe_apply_anthropic_beta_headers(
            headers,
            chat_payload.get("model"),
            valves=effective_valves,
        )
        headers = _apply_owui_forward_user_headers(headers, user, owui_chat_id)
        _debug_print_request(headers, chat_payload, logger=self.logger)
        url = base_url.rstrip("/") + "/chat/completions"

        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        tool_call_added: set[int] = set()
        tool_calls_completed = False
        cut_off = False
        truncating_reason: str | None = None
        assistant_text_parts: list[str] = []
        latest_usage: dict[str, Any] = {}
        seen_citation_urls: set[str] = set()
        latest_message_annotations: list[dict[str, Any]] = []
        reasoning_item_id: str | None = None
        reasoning_text_parts: list[str] = []
        reasoning_text_seen = False
        reasoning_summary_text: str | None = None
        reasoning_details_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        reasoning_details_order: list[tuple[str, str]] = []
        image_item_id: str | None = None
        image_output_item: dict[str, Any] | None = None
        images_emitted = False
        provider_refusal_parts: list[str] = []
        refusal_text_seen = False

        @timed
        def _ensure_tool_call_id(index: int, current: dict[str, Any]) -> str:
            tid = current.get("id")
            if isinstance(tid, str) and tid.strip():
                return tid.strip()
            generated = ChatCompletionsAdapter._made_up_call_id(index)
            current["id"] = generated
            return generated

        def _arguments_parse(arguments: Any) -> bool:
            if not isinstance(arguments, str) or not arguments:
                return False
            try:
                json.loads(arguments)
            except (RecursionError, UnicodeDecodeError, ValueError):
                return False
            return True

        def _match_open_tool_call(
            slots: dict[int, dict[str, Any]], raw_call: dict[str, Any]
        ) -> int:
            raw_id = raw_call.get("id")
            has_raw_id = isinstance(raw_id, str) and bool(raw_id.strip())
            function_frame = raw_call.get("function")
            frame_name = function_frame.get("name") if isinstance(function_frame, dict) else None
            has_frame_name = isinstance(frame_name, str) and bool(frame_name)
            matched = None
            if slots and has_raw_id:
                for open_index, slot in slots.items():
                    if slot.get("id") == raw_id and not _arguments_parse(slot.get("arguments")):
                        matched = open_index
                        break
            if matched is None and slots and not has_raw_id and has_frame_name:
                for open_index, slot in slots.items():
                    if slot.get("name") == frame_name and not _arguments_parse(slot.get("arguments")):
                        matched = open_index
                        break
            if matched is None and slots and not has_raw_id and not has_frame_name and len(slots) == 1 and not _arguments_parse(slots[max(slots.keys())].get("arguments")):
                matched = max(slots.keys())
            if matched is None:
                matched = max(slots.keys(), default=-1) + 1
            return matched

        emitted_any = False
        received_any = False
        delivered_any = False
        saw_choice_chunk = False

        def _retry_streaming(retry_state) -> bool:
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            return _should_retry_stream(delivered_any, exc)

        retryer = _transient_retry_policy(effective_valves, retry=_retry_streaming)

        @timed
        def _record_reasoning_detail(detail: dict[str, Any]) -> None:
            dtype = detail.get("type")
            if not isinstance(dtype, str) or not dtype:
                return
            idx = detail.get("index")
            did = detail.get("id")
            if isinstance(idx, int) and not isinstance(idx, bool):
                disc = f"idx:{idx}"
            elif isinstance(did, str) and did.strip():
                disc = did.strip()
            else:
                disc = f"pos:{len(reasoning_details_order)}"
            key = (dtype, disc)
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
                    _append_text_field(merged, "text", next_text)
                elif isinstance(next_text, str):
                    merged["text"] = next_text
                next_signature = detail.get("signature")
                if isinstance(next_signature, str) and next_signature:
                    merged["signature"] = next_signature
                next_format = detail.get("format")
                if isinstance(next_format, str) and next_format:
                    merged["format"] = next_format
            elif dtype == "reasoning.summary":
                next_summary = detail.get("summary")
                if isinstance(next_summary, str) and next_summary.strip():
                    merged["summary"] = next_summary
            elif dtype == "reasoning.encrypted":
                prev_data = merged.get("data")
                next_data = detail.get("data")
                if isinstance(prev_data, str) and isinstance(next_data, str):
                    _append_text_field(merged, "data", next_data)
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
                    clean = dict(detail)
                    if not clean.get("signature"):
                        clean.pop("signature", None)
                    if not clean.get("format"):
                        clean.pop("format", None)
                    out.append(clean)
            return out

        def _chat_chunk_is_user_visible(chunk_obj: Any) -> bool:
            if not isinstance(chunk_obj, dict):
                return False
            choices = chunk_obj.get("choices")
            if not isinstance(choices, list) or not choices:
                return False
            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice0.get("delta") if isinstance(choice0, dict) else None
            if not isinstance(delta, dict):
                return False
            for key in ("content", "refusal", "reasoning", "reasoning_content"):
                value = delta.get(key)
                if isinstance(value, str) and value.strip():
                    return True
            for key in ("tool_calls", "reasoning_details"):
                value = delta.get(key)
                if isinstance(value, list) and value:
                    return True
            return False

        def _consume_blob(data_blob: bytes):
            nonlocal emitted_any, received_any, latest_usage, reasoning_item_id, reasoning_text_seen, \
                reasoning_summary_text, latest_message_annotations, image_item_id, \
                image_output_item, images_emitted, refusal_text_seen, tool_calls_completed, \
                truncating_reason, delivered_any, saw_choice_chunk
            try:
                chunk_obj = json.loads(data_blob.decode("utf-8"))
            except (RecursionError, UnicodeDecodeError, ValueError) as exc:
                self.logger.log(
                    warn_level(
                        _warned_chat_chunk_parse,
                        "chunk_parse",
                        cooldown_s=_CHAT_CHUNK_PARSE_WARN_COOLDOWN_S,
                    ),
                    "Chunk parse failed; the affected event is "
                    "discarded: %s",
                    exc,
                    exc_info=True,
                )
                return
            reported_error = self._pipe._ensure_error_formatter()._extract_streaming_error_event(
                chunk_obj, chat_payload.get("model")
            )
            if reported_error is not None:
                raise reported_error
            received_any = True
            emitted_any = emitted_any or _chat_chunk_is_user_visible(chunk_obj)

            if isinstance(chunk_obj, dict) and isinstance(chunk_obj.get("usage"), dict):
                latest_usage = dict(chunk_obj["usage"])

            choices = chunk_obj.get("choices") if isinstance(chunk_obj, dict) else None
            if not isinstance(choices, list) or not choices:
                return
            saw_choice_chunk = True
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
                        delivered_any = True
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
                            reasoning_text_seen = True
                            delivered_any = True
                            yield {
                                "type": "response.reasoning_text.delta",
                                "item_id": reasoning_item_id,
                                "delta": text,
                            }
                    elif rtype == "reasoning.summary":
                        summary = entry.get("summary")
                        if isinstance(summary, str) and summary.strip():
                            reasoning_summary_text = summary.strip()
                            delivered_any = True
                            yield {
                                "type": "response.reasoning_summary_text.done",
                                "item_id": reasoning_item_id,
                                "text": reasoning_summary_text,
                            }

            delta_reasoning_text = None
            for key in ("reasoning", "reasoning_content"):
                candidate = delta_obj.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    delta_reasoning_text = candidate
                    break
            if delta_reasoning_text:
                if reasoning_item_id is None:
                    reasoning_item_id = f"reasoning-{generate_item_id()}"
                    delivered_any = True
                    yield {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_item_id,
                            "status": "in_progress",
                        },
                    }
                reasoning_text_parts.append(delta_reasoning_text)
                reasoning_text_seen = True
                delivered_any = True
                yield {
                    "type": "response.reasoning_text.delta",
                    "item_id": reasoning_item_id,
                    "delta": delta_reasoning_text,
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
                if not reasoning_text_seen:
                    message_reasoning_text = None
                    for key in ("reasoning", "reasoning_content"):
                        candidate = message_obj.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            message_reasoning_text = candidate
                            break
                    if message_reasoning_text:
                        if reasoning_item_id is None:
                            reasoning_item_id = f"reasoning-{generate_item_id()}"
                            delivered_any = True
                            yield {
                                "type": "response.output_item.added",
                                "item": {
                                    "type": "reasoning",
                                    "id": reasoning_item_id,
                                    "status": "in_progress",
                                },
                            }
                        reasoning_text_parts.append(message_reasoning_text)
                        reasoning_text_seen = True
                        delivered_any = True
                        yield {
                            "type": "response.reasoning_text.delta",
                            "item_id": reasoning_item_id,
                            "delta": message_reasoning_text,
                        }
                if not refusal_text_seen:
                    message_refusal = message_obj.get("refusal")
                    if isinstance(message_refusal, str) and message_refusal.strip():
                        delivered_any = True
                        provider_refusal_parts.append(message_refusal)
                        refusal_text_seen = True
                message_images = message_obj.get("images")
                if (
                    not images_emitted
                    and isinstance(message_images, list)
                    and message_images
                ):
                    image_results: list[Any] = []
                    for entry in message_images:
                        if isinstance(entry, dict):
                            image_results.append(dict(entry))
                        elif isinstance(entry, str) and entry.strip():
                            image_results.append(entry.strip())
                    if image_results:
                        if image_item_id is None:
                            image_item_id = f"image-{generate_item_id()}"
                        image_output_item = {
                            "type": "image_generation_call",
                            "id": image_item_id,
                            "status": "completed",
                            "result": image_results,
                        }
                        delivered_any = True
                        yield {
                            "type": "response.output_item.added",
                            "item": dict(image_output_item, status="in_progress"),
                        }
                        delivered_any = True
                        yield {
                            "type": "response.output_item.done",
                            "item": image_output_item,
                        }
                        images_emitted = True

            if annotations:
                for url, title, content in _parse_url_citation_annotations(annotations):
                    if url in seen_citation_urls:
                        continue
                    seen_citation_urls.add(url)
                    delivered_any = True
                    yield {
                        "type": "response.output_text.annotation.added",
                        "annotation": {"type": "url_citation", "url": url, "title": title, "content": content},
                    }

            content_delta = delta_obj.get("content")
            if isinstance(content_delta, str) and content_delta:
                assistant_text_parts.append(content_delta)
                delivered_any = True
                yield {"type": "response.output_text.delta", "delta": content_delta}

            delta_refusal = delta_obj.get("refusal")
            if isinstance(delta_refusal, str) and delta_refusal.strip():
                delivered_any = True
                provider_refusal_parts.append(delta_refusal)
                refusal_text_seen = True

            tool_calls = delta_obj.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                for raw_call in tool_calls:
                    if not isinstance(raw_call, dict):
                        continue
                    index = raw_call.get("index")
                    if not isinstance(index, int):
                        index = _match_open_tool_call(tool_calls_by_index, raw_call)
                    current = tool_calls_by_index.setdefault(index, {})
                    delivered_any = True
                    raw_id = raw_call.get("id")
                    if isinstance(raw_id, str) and raw_id.strip():
                        current["id"] = raw_id
                    function = raw_call.get("function")
                    if isinstance(function, dict):
                        name = function.get("name")
                        if isinstance(name, str) and name:
                            current["name"] = name
                        args_delta = function.get("arguments")
                        if isinstance(args_delta, str) and args_delta:
                            _append_text_field(current, "arguments", args_delta)

                    if index not in tool_call_added:
                        tool_call_added.add(index)
                        call_id = _ensure_tool_call_id(index, current)
                        delivered_any = True
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
            if isinstance(finish_reason, str) and finish_reason in {
                "tool_calls",
                "stop",
                "length",
                "content_filter",
            }:
                tool_calls_completed = True
                if finish_reason == "length":
                    truncating_reason = "max_output_tokens"

        first_chunk_received = False
        async with _count_failed_call(self._pipe, breaker_key):
            async for attempt in retryer:
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        tool_calls_by_index.clear()
                        tool_call_added.clear()
                        assistant_text_parts.clear()
                        provider_refusal_parts.clear()
                        refusal_text_seen = False
                        reasoning_text_parts.clear()
                        reasoning_summary_text = None
                        reasoning_details_by_key.clear()
                        reasoning_details_order.clear()
                        seen_citation_urls.clear()
                        latest_message_annotations = []
                        images_emitted = False
                        emitted_any = False
                        received_any = False
                        saw_choice_chunk = False
                        cut_off = False
                        tool_calls_completed = False
                        truncating_reason = None
                        latest_usage = {}

                    await self._inline_internal_chat_files(chat_payload, effective_valves, user=user)

                    timing_mark("chat_http_request_start")
                    async with session.post(
                        url, json=chat_payload, headers=headers,
                        timeout=self._timeout(effective_valves),
                    ) as resp:
                        timing_mark("chat_http_headers_received")
                        if resp.status >= 400:
                            error_body = await _debug_print_error_response(resp, logger=self.logger)
                            extra_meta: dict[str, Any] = {}
                            _apply_retry_after_metadata(extra_meta, resp.headers)
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

                        buf = bytearray()
                        event_data_parts: list[bytes] = []
                        done = False

                        async def _chunks():
                            async for raw in resp.content.iter_any():
                                yield raw
                            if done:  # noqa: B023 - shares the loop's state by design
                                return
                            tail = bytes(buf)  # noqa: B023
                            del buf[:]  # noqa: B023
                            if not tail and not event_data_parts:  # noqa: B023
                                return
                            pending = list(event_data_parts)  # noqa: B023
                            event_data_parts.clear()  # noqa: B023
                            for part in pending:
                                yield b"data: " + part + b"\n\n"
                            if tail:
                                yield tail + b"\n\n"

                        async for chunk in _chunks():
                            if not chunk:
                                continue
                            if not first_chunk_received:
                                first_chunk_received = True
                                timing_mark("chat_first_chunk")
                            buf.extend(chunk)
                            for stripped in _split_sse_lines(buf):

                                if not stripped:
                                    if not event_data_parts:
                                        continue
                                    data_blob = b"\n".join(event_data_parts).strip()
                                    event_data_parts.clear()
                                    if not data_blob:
                                        continue
                                    if data_blob == _CHAT_SSE_DONE_SENTINEL:
                                        done = True
                                        timing_mark("chat_stream_done")
                                        break
                                    for ev in _consume_blob(data_blob):
                                        yield ev

                                if stripped.startswith(b":"):
                                    continue
                                if stripped.startswith(b"data:"):
                                    payload = bytes(stripped[5:].lstrip())
                                    if payload == _CHAT_SSE_DONE_SENTINEL:
                                        if event_data_parts:
                                            data_blob = b"\n".join(event_data_parts).strip()
                                            event_data_parts.clear()
                                            if data_blob and data_blob != _CHAT_SSE_DONE_SENTINEL:
                                                for ev in _consume_blob(data_blob):
                                                    yield ev
                                        done = True
                                        timing_mark("chat_stream_done")
                                        break
                                    event_data_parts.append(payload)
                                    continue

                            if done:
                                break
                        if event_data_parts and not done:
                            data_blob = b"\n".join(event_data_parts).strip()
                            event_data_parts.clear()
                            if data_blob and data_blob != _CHAT_SSE_DONE_SENTINEL:
                                for ev in _consume_blob(data_blob):
                                    yield ev
                        if not received_any:
                            raise aiohttp.ClientPayloadError("OpenRouter closed the stream before sending anything")
                        if not saw_choice_chunk:
                            raise aiohttp.ClientPayloadError("OpenRouter sent no choices on /chat/completions")
                        if not done and not tool_calls_completed:
                            _record_failed_call(self._pipe, breaker_key)
                            cut_off = True
                        break

        if cut_off:
            return

        if reasoning_item_id is not None:
            reasoning_text = "".join(reasoning_text_parts).strip()
            reasoning_item: dict[str, Any] = {
                "type": "reasoning",
                "id": reasoning_item_id,
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": reasoning_text}] if reasoning_text else [],
                "summary": [{"type": "summary_text", "text": reasoning_summary_text}] if reasoning_summary_text else [],
            }
            for detail in _final_reasoning_details():
                detail_type = detail.get("type")
                if detail_type == "reasoning.text":
                    for field in ("signature", "format"):
                        value = detail.get(field)
                        if isinstance(value, str) and value and field not in reasoning_item:
                            reasoning_item[field] = value
                elif detail_type == "reasoning.encrypted":
                    data = detail.get("data")
                    if isinstance(data, str) and data and "encrypted_content" not in reasoning_item:
                        reasoning_item["encrypted_content"] = data
            yield {
                "type": "response.output_item.done",
                "item": reasoning_item,
            }

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

        assistant_text = "".join(assistant_text_parts)
        refusal_delta, assistant_text = _refusal_split(
            assistant_text, "".join(provider_refusal_parts)
        )
        if refusal_delta is not None:
            yield {"type": "response.output_text.delta", "delta": refusal_delta}
        tool_call_items: list[dict[str, Any]] = []
        for index in sorted(tool_calls_by_index.keys()):
            current = tool_calls_by_index[index]
            name = current.get("name")
            if not isinstance(name, str) or not name:
                continue
            call_id = _ensure_tool_call_id(index, current)
            tool_call_items.append(
                {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": current.get("arguments") or "{}",
                }
            )
        output: list[dict[str, Any]] = _build_output_items(
            assistant_text=assistant_text,
            annotations=latest_message_annotations,
            reasoning_details=_final_reasoning_details(),
            image_output_item=image_output_item,
            tool_calls=tool_call_items,
        )

        if truncating_reason is not None:
            yield {
                "type": "response.completed",
                "response": {
                    "output": output,
                    "usage": ChatCompletionsAdapter._chat_usage_to_responses_usage(latest_usage),
                    "status": "incomplete",
                    "incomplete_details": {"reason": truncating_reason},
                },
            }
            return
        yield {
            "type": "response.completed",
            "response": {
                "output": output,
                "usage": ChatCompletionsAdapter._chat_usage_to_responses_usage(latest_usage),
            },
        }


    @timed
    async def send_openai_chat_completions_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        breaker_key: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
        transient_retry: bool = True,
    ) -> dict[str, Any]:
        """Send /chat/completions with stream=false and return the JSON payload."""
        effective_valves = valves or self._pipe.valves
        responses_payload = dict(responses_request_body or {})
        await self._pipe._file_gateway.inline_internal_responses_input_files_inplace(
            responses_payload,
            chunk_size=effective_valves.IMAGE_UPLOAD_CHUNK_BYTES,
            max_bytes=effective_valves.BASE64_MAX_SIZE_MB * 1024 * 1024,
            user=user,
        )
        chat_payload = _responses_payload_to_chat_completions_payload(
            responses_payload,
        )
        chat_payload = _filter_openrouter_chat_request(chat_payload)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            "HTTP-Referer": _select_openrouter_http_referer(effective_valves),
        }
        self._pipe._maybe_apply_anthropic_beta_headers(
            headers,
            chat_payload.get("model"),
            valves=effective_valves,
        )
        headers = _apply_owui_forward_user_headers(headers, user, owui_chat_id)
        _debug_print_request(headers, chat_payload, logger=self.logger)
        url = base_url.rstrip("/") + "/chat/completions"

        retryer = _transient_retry_policy(effective_valves, retry=_retry_nonstreaming)
        if not transient_retry:
            retryer.stop = stop_after_attempt(1)

        async with _count_failed_call(self._pipe, breaker_key):
            async for attempt in retryer:
                with attempt:
                    await self._inline_internal_chat_files(chat_payload, effective_valves, user=user)

                    timing_mark("chat_nonstream_http_request_start")
                    async with session.post(
                        url, json=chat_payload, headers=headers,
                        timeout=self._timeout(effective_valves),
                    ) as resp:
                        timing_mark("chat_nonstream_http_response")
                        if resp.status >= 400:
                            error_body = await _debug_print_error_response(resp, logger=self.logger)
                            extra_meta: dict[str, Any] = {}
                            _apply_retry_after_metadata(extra_meta, resp.headers)
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
                        data = await _decode_json_body(resp, self.logger, "/chat/completions")
                        if not isinstance(data, dict):
                            _debug_print_response(data, logger=self.logger)
                            raise _body_not_an_object("/chat/completions", data, resp)
                        _debug_print_response(data, logger=self.logger)
                        reported_error = self._pipe._ensure_error_formatter()._extract_streaming_error_event(
                            data, chat_payload.get("model")
                        )
                        if reported_error is not None:
                            raise reported_error
                        choices = data.get("choices")
                        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
                            raise aiohttp.ClientPayloadError(
                                "OpenRouter returned 200 with no choices on /chat/completions"
                            )
                        return data

        return {}


    @timed
    async def send_openrouter_streaming_request(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        workers: int = 4,
        breaker_key: str | None = None,
        delta_char_limit: int = 0,
        idle_flush_ms: int = 0,
        nagle_min_chars: int = 1,
        chunk_queue_maxsize: int = 100,
        chunk_queue_warn_size: int = 1000,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
        user: Any = None,
        owui_chat_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Unified streaming request entrypoint with endpoint routing + fallback."""
        effective_valves = valves or self._pipe.valves
        idle_flush_seconds = float(idle_flush_ms) / 1000 if idle_flush_ms > 0 else None
        passthrough_deltas = delta_char_limit <= 0 and idle_flush_ms <= 0
        model_id = (responses_request_body or {}).get("model") or ""
        endpoint = endpoint_override or self._pipe._streaming_handler._select_llm_endpoint(str(model_id), valves=effective_valves)
        forced_selected_endpoint, endpoint_forced = self._pipe._streaming_handler._select_llm_endpoint_with_forced(
            str(model_id), valves=effective_valves
        )

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
                nagle_min_chars=nagle_min_chars,
                chunk_queue_maxsize=chunk_queue_maxsize,
                chunk_queue_warn_size=chunk_queue_warn_size,
                event_queue_maxsize=event_queue_maxsize,
                event_queue_warn_size=event_queue_warn_size,
                user=user,
                owui_chat_id=owui_chat_id,
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
                user=user,
                owui_chat_id=owui_chat_id,
            ):
                yield event

        if endpoint == "chat_completions":
            async for event in nagle_coalesce_stream(
                _run_chat(),
                idle_flush_seconds=idle_flush_seconds,
                passthrough=passthrough_deltas,
                min_flush_chars=nagle_min_chars,
            ):
                yield event
            return

        try:
            async for event in _run_responses():
                yield event
        except Exception as exc:
            if (
                effective_valves.AUTO_FALLBACK_CHAT_COMPLETIONS
                and not (endpoint_forced and forced_selected_endpoint == "responses")
                and self._pipe._streaming_handler._looks_like_responses_unsupported(exc)
            ):
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
                yield {"type": "openrouter_pipe.chat_fallback"}
                async for event in nagle_coalesce_stream(
                    _run_chat(),
                    idle_flush_seconds=idle_flush_seconds,
                    passthrough=passthrough_deltas,
                    min_flush_chars=nagle_min_chars,
                ):
                    yield event
                return
            raise


    # Tool Context Shutdown

    @staticmethod
    def _chat_usage_to_responses_usage(raw_usage: Any) -> dict[str, Any]:
        return chat_usage_to_responses_usage(raw_usage)

    @staticmethod
    def _made_up_call_id(index: int) -> str:
        return f"toolcall-{index}-{uuid.uuid4().hex[:16]}"
