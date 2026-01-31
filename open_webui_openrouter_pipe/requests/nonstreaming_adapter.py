"""Non-streaming request adapter for OpenRouter API.

This module handles non-streaming requests to OpenRouter and converts them to
events compatible with the Responses API format.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Optional

import aiohttp

from ..api.transforms import _filter_openrouter_request
from ..storage.persistence import generate_item_id
from ..models.registry import normalize_model_id_dotted
from ..core.timing_logger import timed

if TYPE_CHECKING:
    from ..pipe import Pipe


class NonStreamingAdapter:
    """Adapter for non-streaming OpenRouter requests.

    Handles non-streaming requests and converts responses to event streams
    compatible with the Responses API format.
    """

    @timed
    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize NonStreamingAdapter.

        Args:
            pipe: Reference to Pipe instance for accessing helper methods
            logger: Logger instance
        """
        self._pipe = pipe
        self.logger = logger

    @timed
    async def send_openrouter_nonstreaming_request_as_events(
        self,
        session: aiohttp.ClientSession,
        responses_request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        breaker_key: Optional[str] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Send a non-streaming request and yield Responses-style events."""
        effective_valves = valves or self._pipe.valves
        model_id = (responses_request_body or {}).get("model") or ""
        endpoint = endpoint_override or self._pipe._select_llm_endpoint(str(model_id), valves=effective_valves)

        @timed
        def _extract_chat_message_text(message: Any) -> str:
            if not isinstance(message, dict):
                return ""
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                fragments: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_val = part.get("text")
                        if isinstance(text_val, str):
                            fragments.append(text_val)
                return "".join(fragments)
            return ""

        @timed
        async def _run_responses() -> AsyncGenerator[dict[str, Any], None]:
            request_payload = _filter_openrouter_request(dict(responses_request_body or {}))
            response = await self._pipe.send_openai_responses_nonstreaming_request(
                session,
                request_payload,
                api_key=api_key,
                base_url=base_url,
                valves=effective_valves,
            )
            output_items = response.get("output") if isinstance(response, dict) else None
            assistant_text_parts: list[str] = []
            if isinstance(output_items, list):
                for item in output_items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "message" and item.get("role") == "assistant":
                        content = item.get("content")
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "output_text":
                                    text_val = block.get("text")
                                    if isinstance(text_val, str) and text_val:
                                        assistant_text_parts.append(text_val)
                    if item.get("type") in {"reasoning", "web_search_call", "file_search_call", "image_generation_call", "local_shell_call"}:
                        yield {"type": "response.output_item.done", "item": item}
                    if item.get("type") == "function_call":
                        yield {"type": "response.output_item.added", "item": dict(item, status="in_progress")}
                        yield {"type": "response.output_item.done", "item": dict(item, status="completed")}
            assistant_text = "".join(assistant_text_parts)
            if assistant_text:
                yield {"type": "response.output_text.delta", "delta": assistant_text}
            yield {"type": "response.completed", "response": response if isinstance(response, dict) else {}}

        @timed
        async def _run_chat() -> AsyncGenerator[dict[str, Any], None]:
            chat_response = await self._pipe.send_openai_chat_completions_nonstreaming_request(
                session,
                dict(responses_request_body or {}),
                api_key=api_key,
                base_url=base_url,
                valves=effective_valves,
                breaker_key=breaker_key,
            )
            choices = chat_response.get("choices") if isinstance(chat_response, dict) else None
            message = None
            finish_reason = None
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                finish_reason = choices[0].get("finish_reason")
            message_obj = message if isinstance(message, dict) else {}

            usage = chat_response.get("usage") if isinstance(chat_response, dict) else None
            latest_usage = dict(usage) if isinstance(usage, dict) else {}

            reasoning_item_id: str | None = None
            reasoning_text_parts: list[str] = []
            reasoning_summary_text: str | None = None
            reasoning_details = message_obj.get("reasoning_details")
            if isinstance(reasoning_details, list) and reasoning_details:
                for entry in reasoning_details:
                    if not isinstance(entry, dict):
                        continue
                    rtype = entry.get("type")
                    if not isinstance(rtype, str) or not rtype:
                        continue
                    if reasoning_item_id is None:
                        rid = entry.get("id")
                        reasoning_item_id = rid.strip() if isinstance(rid, str) and rid.strip() else f"reasoning-{generate_item_id()}"
                        yield {
                            "type": "response.output_item.added",
                            "item": {"type": "reasoning", "id": reasoning_item_id, "status": "in_progress"},
                        }
                    if rtype == "reasoning.text":
                        text = entry.get("text")
                        if isinstance(text, str) and text:
                            reasoning_text_parts.append(text)
                            yield {"type": "response.reasoning_text.delta", "item_id": reasoning_item_id, "delta": text}
                    elif rtype == "reasoning.summary":
                        summary = entry.get("summary")
                        if isinstance(summary, str) and summary.strip():
                            reasoning_summary_text = summary.strip()
                            yield {"type": "response.reasoning_summary_text.done", "item_id": reasoning_item_id, "text": reasoning_summary_text}

            message_reasoning_text = None
            for key in ("reasoning", "reasoning_content"):
                candidate = message_obj.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    message_reasoning_text = candidate
                    break
            if message_reasoning_text and not reasoning_text_parts:
                if reasoning_item_id is None:
                    reasoning_item_id = f"reasoning-{generate_item_id()}"
                    yield {
                        "type": "response.output_item.added",
                        "item": {"type": "reasoning", "id": reasoning_item_id, "status": "in_progress"},
                    }
                reasoning_text_parts.append(message_reasoning_text)
                yield {"type": "response.reasoning_text.delta", "item_id": reasoning_item_id, "delta": message_reasoning_text}

            image_output_item: dict[str, Any] | None = None
            message_images = message_obj.get("images")
            if isinstance(message_images, list) and message_images:
                image_results: list[Any] = []
                for entry in message_images:
                    if isinstance(entry, dict):
                        image_results.append(dict(entry))
                    elif isinstance(entry, str) and entry.strip():
                        image_results.append(entry.strip())
                if image_results:
                    image_item_id = f"image-{generate_item_id()}"
                    image_output_item = {
                        "type": "image_generation_call",
                        "id": image_item_id,
                        "status": "completed",
                        "result": image_results,
                    }
                    yield {
                        "type": "response.output_item.added",
                        "item": dict(image_output_item, status="in_progress"),
                    }
                    yield {
                        "type": "response.output_item.done",
                        "item": image_output_item,
                    }

            annotations = message_obj.get("annotations")
            if isinstance(annotations, list) and annotations:
                seen_urls: set[str] = set()
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
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    if isinstance(title, str):
                        title = title.strip() or url
                    else:
                        title = url
                    yield {
                        "type": "response.output_text.annotation.added",
                        "annotation": {"type": "url_citation", "url": url, "title": title},
                    }

            tool_calls = message_obj.get("tool_calls")
            output_calls: list[dict[str, Any]] = []
            if isinstance(tool_calls, list) and tool_calls:
                for index, raw_call in enumerate(tool_calls):
                    if not isinstance(raw_call, dict):
                        continue
                    function = raw_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    args = function.get("arguments")
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
                    call_id = raw_call.get("id")
                    if not isinstance(call_id, str) or not call_id.strip():
                        model_val = (responses_request_body.get("model") or "model")
                        call_id = f"toolcall-{normalize_model_id_dotted(str(model_val))}-{index}"
                    call_id = call_id.strip()
                    item = {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "status": "completed",
                        "name": name,
                        "arguments": args,
                    }
                    yield {"type": "response.output_item.added", "item": dict(item, status="in_progress")}
                    yield {"type": "response.output_item.done", "item": item}
                    output_calls.append(
                        {
                            "type": "function_call",
                            "id": call_id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": args,
                        }
                    )

            if reasoning_item_id is not None:
                reasoning_text = "".join(reasoning_text_parts).strip()
                reasoning_item: dict[str, Any] = {
                    "type": "reasoning",
                    "id": reasoning_item_id,
                    "status": "completed",
                    "content": [{"type": "reasoning_text", "text": reasoning_text}] if reasoning_text else [],
                    "summary": [{"type": "summary_text", "text": reasoning_summary_text}] if reasoning_summary_text else [],
                }
                yield {"type": "response.output_item.done", "item": reasoning_item}

            assistant_text = _extract_chat_message_text(message_obj)
            if assistant_text:
                yield {"type": "response.output_text.delta", "delta": assistant_text}

            output_message: dict[str, Any] = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            }
            if isinstance(annotations, list) and annotations:
                output_message["annotations"] = [dict(a) for a in annotations if isinstance(a, dict)]
            if isinstance(reasoning_details, list) and reasoning_details:
                output_message["reasoning_details"] = [dict(a) for a in reasoning_details if isinstance(a, dict)]

            output: list[dict[str, Any]] = [output_message]
            if image_output_item is not None:
                output.append(image_output_item)
            output.extend(output_calls)

            yield {
                "type": "response.completed",
                "response": {
                    "output": output,
                    "usage": self._pipe._chat_usage_to_responses_usage(latest_usage),
                },
            }

        if endpoint == "chat_completions":
            async for event in _run_chat():
                yield event
            return

        try:
            async for event in _run_responses():
                yield event
        except Exception as exc:
            if getattr(effective_valves, "AUTO_FALLBACK_CHAT_COMPLETIONS", True) and self._pipe._looks_like_responses_unsupported(exc):
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
