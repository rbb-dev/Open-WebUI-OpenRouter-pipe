"""Responses API adapter for OpenRouter.

This module handles Responses API streaming and non-streaming requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional

import aiohttp
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from ...core.config import _OPENROUTER_TITLE, _select_openrouter_http_referer
from ...core.errors import OpenRouterAPIError, _build_openrouter_api_error
from ...core.timing_logger import timed, timing_scope, timing_mark
from ...requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
)
from ...core.logging_system import SessionLogger

if TYPE_CHECKING:
    from ...pipe import Pipe


class ResponsesAdapter:
    """Adapter for OpenRouter /responses API endpoint."""

    @timed
    def __init__(self, pipe: "Pipe", logger: logging.Logger):
        """Initialize ResponsesAdapter.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging
        """
        self._pipe = pipe
        self.logger = logger

    @timed
    async def send_openai_responses_streaming_request(
        self,
        session: aiohttp.ClientSession,
        request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: "Pipe.Valves | None" = None,
        workers: int = 4,
        breaker_key: Optional[str] = None,
        delta_char_limit: int = 0,
        idle_flush_ms: int = 0,
        chunk_queue_maxsize: int = 100,
        event_queue_maxsize: int = 100,
        event_queue_warn_size: int = 1000,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Producer/worker SSE pipeline with configurable delta batching."""

        effective_valves = valves or self._pipe.valves
        chunk_size = getattr(effective_valves, "IMAGE_UPLOAD_CHUNK_BYTES", self._pipe.valves.IMAGE_UPLOAD_CHUNK_BYTES)
        max_bytes = int(getattr(effective_valves, "BASE64_MAX_SIZE_MB", self._pipe.valves.BASE64_MAX_SIZE_MB)) * 1024 * 1024
        await self._pipe._inline_internal_responses_input_files_inplace(
            request_body,
            chunk_size=chunk_size,
            max_bytes=max_bytes,
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _select_openrouter_http_referer(effective_valves),
        }
        self._pipe._maybe_apply_anthropic_beta_headers(
            headers,
            request_body.get("model"),
            valves=effective_valves,
        )
        _debug_print_request(headers, request_body)
        url = base_url.rstrip("/") + "/responses"

        workers = max(1, min(int(workers or 1), 8))
        chunk_queue_size = max(0, int(chunk_queue_maxsize))
        event_queue_size = max(0, int(event_queue_maxsize))
        chunk_queue: asyncio.Queue[tuple[Optional[int], bytes]] = asyncio.Queue(maxsize=chunk_queue_size)
        event_queue: asyncio.Queue[tuple[Optional[int], Optional[dict[str, Any]]]] = asyncio.Queue(maxsize=event_queue_size)
        chunk_sentinel = (None, b"")
        delta_batch_threshold = max(1, int(delta_char_limit))
        idle_flush_seconds = float(idle_flush_ms) / 1000 if idle_flush_ms > 0 else None
        passthrough_deltas = delta_char_limit <= 0 and idle_flush_ms <= 0
        requested_model = request_body.get("model")

        @timed
        async def _producer() -> None:
            seq = 0
            first_chunk_received = False
            retryer = AsyncRetrying(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
                reraise=True,
            )
            try:
                async for attempt in retryer:
                    if breaker_key and not self._pipe._breaker_allows(breaker_key):
                        raise RuntimeError("Breaker open for user during stream")
                    with attempt:
                        buf = bytearray()
                        event_data_parts: list[bytes] = []
                        stream_complete = False
                        try:
                            timing_mark("responses_http_request_start")
                            async with session.post(url, json=request_body, headers=headers) as resp:
                                timing_mark("responses_http_headers_received")
                                if resp.status >= 400:
                                    error_body = await _debug_print_error_response(resp)
                                    if breaker_key:
                                        self._pipe._record_failure(breaker_key)
                                    special_statuses = {400, 401, 402, 403, 408, 429}
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
                                            requested_model=request_body.get("model"),
                                            extra_metadata=extra_meta or None,
                                        )
                                    if resp.status < 500:
                                        raise RuntimeError(
                                            f"OpenRouter request failed ({resp.status}): {resp.reason}"
                                        )
                                resp.raise_for_status()

                                chunk_count = 0
                                first_event_queued = False
                                async for chunk in resp.content.iter_chunked(4096):
                                    chunk_count += 1
                                    # Log chunk with content preview (first 40 chars, sanitized)
                                    preview = chunk[:40].decode("utf-8", errors="replace").replace("\n", "\\n").replace("\r", "\\r")
                                    timing_mark(f"chunk_{chunk_count}_len_{len(chunk)}_[{preview}]")
                                    if not first_chunk_received:
                                        first_chunk_received = True
                                        timing_mark("responses_first_chunk")
                                    view = memoryview(chunk)
                                    if breaker_key and not self._pipe._breaker_allows(breaker_key):
                                        raise RuntimeError("Breaker open during stream")
                                    buf.extend(view)
                                    start_idx = 0
                                    while True:
                                        newline_idx = buf.find(b"\n", start_idx)
                                        if newline_idx == -1:
                                            break
                                        line = buf[start_idx:newline_idx]
                                        start_idx = newline_idx + 1
                                        stripped = line.strip()
                                        if not stripped:
                                            if event_data_parts:
                                                data_blob = b"\n".join(event_data_parts).strip()
                                                event_data_parts.clear()
                                                if not data_blob:
                                                    continue
                                                if data_blob == b"[DONE]":
                                                    stream_complete = True
                                                    timing_mark("responses_stream_done")
                                                    break
                                                if not first_event_queued:
                                                    first_event_queued = True
                                                    timing_mark("producer_first_event_queued")
                                                await chunk_queue.put((seq, data_blob))
                                                seq += 1
                                            continue
                                        if stripped.startswith(b":"):
                                            continue
                                        if stripped.startswith(b"data:"):
                                            event_data_parts.append(bytes(stripped[5:].lstrip()))
                                            continue
                                    if start_idx > 0:
                                        del buf[:start_idx]
                                    if stream_complete:
                                        break

                                if event_data_parts and not stream_complete:
                                    data_blob = b"\n".join(event_data_parts).strip()
                                    event_data_parts.clear()
                                    if data_blob and data_blob != b"[DONE]":
                                        await chunk_queue.put((seq, data_blob))
                                        seq += 1
                        except Exception as producer_exc:
                            is_auth_failure = (
                                isinstance(producer_exc, OpenRouterAPIError)
                                and getattr(producer_exc, "status", None) in {401, 403}
                            )
                            if is_auth_failure:
                                self._pipe._note_auth_failure()
                                self.logger.warning(
                                    "Producer encountered auth error while streaming from OpenRouter: %s",
                                    producer_exc,
                                )
                            else:
                                self.logger.error(
                                    "Producer encountered error while streaming from OpenRouter: %s",
                                    producer_exc,
                                    exc_info=True,
                                )
                            if breaker_key:
                                self._pipe._record_failure(breaker_key)
                            raise
                        if stream_complete:
                            break
            finally:
                for _ in range(workers):
                    with contextlib.suppress(asyncio.CancelledError):
                        await chunk_queue.put(chunk_sentinel)

        worker_first_event_queued = False

        @timed
        async def _worker(worker_idx: int) -> None:
            nonlocal worker_first_event_queued
            first_chunk_got = False
            try:
                while True:
                    seq, data = await chunk_queue.get()
                    if not first_chunk_got:
                        first_chunk_got = True
                        timing_mark(f"worker_{worker_idx}_first_chunk_got")
                    try:
                        if seq is None:
                            break
                        if data == b"[DONE]":
                            continue
                        try:
                            event = json.loads(data.decode("utf-8"))
                        except json.JSONDecodeError as exc:
                            self.logger.warning("Chunk parse failed (seq=%s): %s", seq, exc)
                            continue
                        if not worker_first_event_queued:
                            worker_first_event_queued = True
                            timing_mark("worker_first_event_to_queue")
                        await event_queue.put((seq, event))
                    finally:
                        chunk_queue.task_done()
            finally:
                with contextlib.suppress(asyncio.CancelledError):
                    await event_queue.put((None, None))

        producer_task = asyncio.create_task(_producer(), name="openrouter-sse-producer")
        worker_tasks = [
            asyncio.create_task(_worker(idx), name=f"openrouter-sse-worker-{idx}")
            for idx in range(workers)
        ]

        pending_events: dict[int, dict[str, Any]] = {}
        next_seq = 0
        done_workers = 0
        delta_buffer: list[str] = []
        delta_template: Optional[dict[str, Any]] = None
        delta_length = 0
        event_queue_warn_last_ts: float = 0.0

        @timed
        def flush_delta(force: bool = False) -> Optional[dict[str, Any]]:
            if passthrough_deltas:
                return None
            nonlocal delta_buffer, delta_template, delta_length
            if delta_buffer and (force or delta_length >= delta_batch_threshold):
                combined = "".join(delta_buffer)
                base = dict(delta_template or {"type": "response.output_text.delta"})
                base["delta"] = combined
                delta_buffer = []
                delta_template = None
                delta_length = 0
                return base
            return None

        first_event_from_queue = False
        first_yield_done = False

        try:
            while True:
                timeout = idle_flush_seconds if (idle_flush_seconds and delta_buffer) else None
                timed_out = False
                seq: int | None = None
                event: dict[str, Any] | None = None
                if timeout is not None:
                    try:
                        seq, event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                    except asyncio.TimeoutError:
                        timed_out = True
                else:
                    seq, event = await event_queue.get()
                if not first_event_from_queue and seq is not None:
                    first_event_from_queue = True
                    timing_mark("consumer_first_event_from_queue")

                if timed_out:
                    batched = flush_delta(force=True)
                    if batched:
                        yield batched
                    continue

                event_queue.task_done()

                # Non-spammy queue monitoring
                now = time.perf_counter()
                if self._pipe._should_warn_event_queue_backlog(
                    event_queue.qsize(),
                    event_queue_warn_size,
                    now,
                    event_queue_warn_last_ts,
                ):
                    self.logger.warning(
                        "Event queue backlog high: %d items (session=%s)",
                        event_queue.qsize(),
                        SessionLogger.session_id.get() or "unknown",
                    )
                    event_queue_warn_last_ts = now

                if seq is None:
                    done_workers += 1
                    if done_workers >= workers and not pending_events:
                        break
                    continue
                if event is None:
                    self.logger.debug("Skipping empty SSE event (seq=%s)", seq)
                    continue
                pending_events[seq] = event
                while next_seq in pending_events:
                    current = pending_events.pop(next_seq)
                    next_seq += 1
                    streaming_error = self._pipe._extract_streaming_error_event(current, requested_model)
                    if streaming_error is not None:
                        raise streaming_error
                    etype = current.get("type")
                    if etype == "response.output_text.delta":
                        delta_chunk = current.get("delta", "")
                        if passthrough_deltas:
                            if delta_chunk:
                                if not first_yield_done:
                                    first_yield_done = True
                                    timing_mark("adapter_first_yield")
                                yield current
                            continue
                        if delta_chunk:
                            delta_buffer.append(delta_chunk)
                            delta_length += len(delta_chunk)
                            if delta_template is None:
                                delta_template = {k: v for k, v in current.items() if k != "delta"}
                        batched = flush_delta()
                        if batched:
                            yield batched
                        continue

                    batched = flush_delta(force=True)
                    if batched:
                        if not first_yield_done:
                            first_yield_done = True
                            timing_mark("adapter_first_yield")
                        yield batched
                    if not first_yield_done:
                        first_yield_done = True
                        timing_mark("adapter_first_yield")
                    yield current

            final_delta = flush_delta(force=True)
            if final_delta:
                yield final_delta

            await producer_task
        finally:
            if not producer_task.done():
                producer_task.cancel()
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    task_name = "producer" if idx == 0 else f"worker-{idx - 1}"
                    self.logger.error(
                        "SSE %s task failed during cleanup: %s",
                        task_name,
                        result,
                        exc_info=result,
                    )


    # ======================================================================
    # send_openai_chat_completions_streaming_request (437 lines)
    # ======================================================================

    @timed
    async def send_openai_responses_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        request_params: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        breaker_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a blocking request to the Responses API and return the JSON payload."""
        effective_valves = valves or self._pipe.valves
        chunk_size = getattr(effective_valves, "IMAGE_UPLOAD_CHUNK_BYTES", self._pipe.valves.IMAGE_UPLOAD_CHUNK_BYTES)
        max_bytes = int(getattr(effective_valves, "BASE64_MAX_SIZE_MB", self._pipe.valves.BASE64_MAX_SIZE_MB)) * 1024 * 1024
        await self._pipe._inline_internal_responses_input_files_inplace(
            request_params,
            chunk_size=chunk_size,
            max_bytes=max_bytes,
        )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": _OPENROUTER_TITLE,
            "HTTP-Referer": _select_openrouter_http_referer(effective_valves),
        }
        _debug_print_request(headers, request_params)
        url = base_url.rstrip("/") + "/responses"

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

                async with session.post(url, json=request_params, headers=headers) as resp:
                    if resp.status >= 400:
                        error_body = await _debug_print_error_response(resp)
                        if breaker_key:
                            self._pipe._record_failure(breaker_key)
                        if resp.status < 500:
                            special_statuses = {400, 401, 402, 403, 408, 429}
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
                                    requested_model=request_params.get("model"),
                                    extra_metadata=extra_meta or None,
                                )
                            raise RuntimeError(
                                f"OpenRouter request failed ({resp.status}): {resp.reason}"
                            )
                    resp.raise_for_status()
                    return await resp.json()
        self.logger.error("Responses API call completed without yielding a response body; returning empty payload.")
        return {}

    # ----------------------------------------------------------------------
    # _run_task_model_request (122 lines)
    # ----------------------------------------------------------------------
