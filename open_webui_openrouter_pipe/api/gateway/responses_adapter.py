"""Responses API adapter for OpenRouter.

This module handles Responses API streaming and non-streaming requests.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, TypeGuard

import aiohttp
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
)

from ...core.config import (
    _OPENROUTER_CATEGORIES,
    _OPENROUTER_TITLE,
    _apply_owui_forward_user_headers,
    _select_openrouter_http_referer,
)
from ...core.errors import (
    EmptyAnswerError,
    OpenRouterAPIError,
    UpstreamBodyUnreadable,
    _build_openrouter_api_error,
    _ChatRetryWait,
    _classify_retryable_openrouter_error,
    is_sign_in_failure,
)
from ...core.logging_system import SessionLogger
from ...core.timing_logger import timed, timing_mark
from ...core.utils import (
    _apply_retry_after_metadata,
    _data_url_log_subject,
    http_timeout,
)
from ...core.warn_latch import warn_level
from ...integrations.anthropic import _maybe_apply_responses_toplevel_cache_control
from ...requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
    _debug_print_response,
)
from ...storage.owui_files import loggable_session_id
from ...streaming.nagle_coalescer import (
    _MAX_DRAIN_PER_CYCLE,
    NagleCoalescer,
    idle_flush_timeout,
)

if TYPE_CHECKING:
    from ...pipe import Pipe

_RESPONSES_CHUNK_PARSE_WARN_COOLDOWN_S = 30.0
_BACKLOG_DEBUG_SAMPLE_INTERVAL_S = 1.0
_BODY_EXCERPT_CHARS = 200
_RESPONSES_SSE_DONE_SENTINEL = b"[DONE]"
_warned_responses_chunk_parse: dict[str, float] = {}
_warned_queue_backlog: dict[str, float] = {}
_warned_queue_backlog_sample: dict[str, float] = {}


def _backlog_cause(queue: str, request_id: str) -> str:
    return f"{queue}:{request_id}" if request_id else f"{queue}:unknown"


def _backlog_debug_due(queue: str, request_id: str) -> bool:
    key = _backlog_cause(queue, request_id)
    now = time.monotonic()
    previous = _warned_queue_backlog_sample.get(key)
    if previous is not None and now - previous < _BACKLOG_DEBUG_SAMPLE_INTERVAL_S:
        return False
    _warned_queue_backlog_sample[key] = now
    return True


def _drop_backlog_latch(request_id: str) -> None:
    if not request_id:
        return
    for queue_name in ("chunk_queue", "event_queue", "chat_pump_queue"):
        cause = _backlog_cause(queue_name, request_id)
        _warned_queue_backlog.pop(cause, None)
        _warned_queue_backlog_sample.pop(cause, None)


def _should_retry_stream(emitted_any: bool, exc: BaseException | None) -> bool:
    """Decide whether a streaming attempt may be retried.

    Retry is only safe BEFORE any event has reached the consumer. Once output
    has been emitted, a retry re-POSTs and re-streams the whole response with
    fresh (higher) sequence numbers, duplicating content the user already saw —
    so never retry after the first emitted event. Pre-output network failures
    (no event delivered yet) are still safely retried.
    """
    if emitted_any:
        return False
    if isinstance(exc, AcceptedResponseLostBody):
        return False
    return isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)) or _classify_retryable_openrouter_error(exc)[0]


_STREAM_END_EVENTS = frozenset({"response.completed", "response.done", "response.incomplete"})


def _terminal_frame_is_readable(event: dict[str, Any]) -> bool:
    if event.get("type") not in _STREAM_END_EVENTS:
        return False
    response = event.get("response")
    return isinstance(response, dict) and bool(response)


_RESPONSES_INVISIBLE_EVENTS = frozenset(
    {"response.created", "response.in_progress", "response.queued", "response.heartbeat"}
)


def _responses_event_is_user_visible(event: dict[str, Any]) -> bool:
    if not isinstance(event, dict):
        return False
    etype = event.get("type")
    if not isinstance(etype, str) or not etype:
        return True
    return etype not in _RESPONSES_INVISIBLE_EVENTS


_UNREADABLE = object()


def _decode_frame(data_blob: bytes, logger: logging.Logger | None = None) -> Any:
    try:
        return json.loads(data_blob.decode("utf-8"))
    except (RecursionError, UnicodeDecodeError, ValueError):
        if logger is not None:
            logger.log(
                warn_level(
                    _warned_responses_chunk_parse,
                    "producer_frame_decode",
                    cooldown_s=_RESPONSES_CHUNK_PARSE_WARN_COOLDOWN_S,
                ),
                "Producer frame parse failed; the affected event is discarded",
                exc_info=True,
            )
        return _UNREADABLE


def _is_ordered_object(event: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(event, dict)


def _task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and bool(task.cancelling())


class AcceptedResponseLostBody(aiohttp.ClientPayloadError, RuntimeError):
    pass


_LOST_BODY_READ_ERRORS = (
    aiohttp.ClientPayloadError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ServerTimeoutError,
    aiohttp.ClientOSError,
    asyncio.TimeoutError,
)


def _should_retry_nonstreaming(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, AcceptedResponseLostBody):
        return False
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    return _classify_retryable_openrouter_error(exc)[0]


def _retry_nonstreaming(retry_state) -> bool:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    return _should_retry_nonstreaming(exc)


def _transient_retry_stop(valves: Any) -> Any:
    extra = getattr(valves, "TRANSIENT_RETRY_MAX_ATTEMPTS", 2)
    try:
        extra = int(extra)
    except (TypeError, ValueError):
        extra = 2
    return stop_after_attempt(max(1, 1 + extra))


def _transient_retry_policy(valves: Any, *, retry: Any) -> AsyncRetrying:
    cap = getattr(valves, "TRANSIENT_RETRY_MAX_WAIT_SECONDS", 30)
    try:
        cap = float(cap)
    except (TypeError, ValueError):
        cap = 30.0
    return AsyncRetrying(
        stop=_transient_retry_stop(valves),
        wait=_ChatRetryWait(wait_exponential(multiplier=0.5, min=0.5, max=4), cap),
        retry=retry,
        reraise=True,
    )


async def _decode_json_body(resp: Any, logger: Any, endpoint: str) -> Any:
    try:
        return await resp.json()
    except _LOST_BODY_READ_ERRORS as exc:
        raise AcceptedResponseLostBody(str(exc)) from exc
    except Exception:
        logger.debug(
            "OpenRouter response was not decodable JSON; falling back to text",
            exc_info=True,
        )
        text = ""
        try:
            text = await resp.text()
            return json.loads(text)
        except _LOST_BODY_READ_ERRORS as exc:
            raise AcceptedResponseLostBody(str(exc)) from exc
        except Exception as exc:
            raise UpstreamBodyUnreadable(
                endpoint=endpoint,
                body_excerpt=str(text)[:_BODY_EXCERPT_CHARS],
                content_type=getattr(resp, "content_type", None),
            ) from exc


def _decoded_body_excerpt(payload: Any) -> str:
    return repr(payload)[:_BODY_EXCERPT_CHARS]


def _body_not_an_object(endpoint: str, payload: Any, resp: Any) -> UpstreamBodyUnreadable:
    return UpstreamBodyUnreadable(
        endpoint=endpoint,
        body_excerpt=_decoded_body_excerpt(payload),
        content_type=getattr(resp, "headers", {}).get("Content-Type"),
    )



class _FailureCharge:
    __slots__ = ("recorded_at",)

    def __init__(self) -> None:
        self.recorded_at: float | None = None


def _record_failed_call(
    pipe: Pipe, breaker_key: str | None, charge_holder: _FailureCharge | None = None
) -> None:
    if breaker_key:
        recorded_at = pipe._circuit_breaker.record_failure(breaker_key)
        if charge_holder is not None:
            charge_holder.recorded_at = recorded_at


def _withdraw_repaired_failure(
    pipe: Pipe, breaker_key: str | None, charge_holder: _FailureCharge | None
) -> None:
    if charge_holder is None:
        return
    pipe._circuit_breaker.retract_failure(breaker_key or "", charge_holder.recorded_at)
    charge_holder.recorded_at = None


def _once_only_charge(
    pipe: Pipe, breaker_key: str | None, charge_holder: _FailureCharge | None = None
) -> Callable[[], None]:
    charged = False

    def _charge() -> None:
        nonlocal charged
        if charged:
            return
        charged = True
        _record_failed_call(pipe, breaker_key, charge_holder)

    return _charge


def _split_sse_lines(buf: bytearray, scanned: int = 0) -> tuple[list[bytes], int]:
    lines: list[bytes] = []
    start_idx = 0
    search_idx = scanned if 0 <= scanned <= len(buf) else 0
    while True:
        newline_idx = buf.find(b"\n", search_idx)
        if newline_idx == -1:
            break
        lines.append(bytes(buf[start_idx:newline_idx]).strip())
        start_idx = search_idx = newline_idx + 1
    if start_idx > 0:
        del buf[:start_idx]
        scanned = 0
    else:
        scanned = len(buf)
    return lines, scanned


@contextlib.asynccontextmanager
async def _count_failed_call(
    pipe: Pipe,
    breaker_key: str | None,
    charge: Callable[[], None] | None = None,
    charge_holder: _FailureCharge | None = None,
) -> AsyncGenerator[None, None]:
    try:
        yield
    except (OpenRouterAPIError, aiohttp.ClientError, TimeoutError):
        if charge is not None:
            charge()
        else:
            _record_failed_call(pipe, breaker_key, charge_holder)
        raise


class ResponsesAdapter:
    """Adapter for OpenRouter /responses API endpoint."""

    @timed
    def __init__(self, pipe: Pipe, logger: logging.Logger):
        """Initialize ResponsesAdapter.

        Args:
            pipe: Parent Pipe instance for accessing configuration and methods
            logger: Logger instance for debugging
        """
        self._pipe = pipe
        self.logger = logger

    def _timeout(self, effective_valves: Any) -> aiohttp.ClientTimeout:
        return http_timeout(effective_valves)

    @timed
    async def send_openai_responses_streaming_request(
        self,
        session: aiohttp.ClientSession,
        request_body: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
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
        files_inlined: bool = False,
        charge_holder: _FailureCharge | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Producer/worker SSE pipeline with configurable delta batching."""

        _backlog_request_id = SessionLogger.request_id.get() or ""
        effective_valves = valves or self._pipe.valves
        if not files_inlined:
            request_body = await self._pipe._file_gateway.inline_internal_responses_input_files(
                request_body,
                chunk_size=effective_valves.IMAGE_UPLOAD_CHUNK_BYTES,
                max_bytes=effective_valves.BASE64_MAX_SIZE_MB * 1024 * 1024,
                user=user,
            )
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
            request_body.get("model"),
            valves=effective_valves,
        )
        headers = _apply_owui_forward_user_headers(headers, user, owui_chat_id)
        _maybe_apply_responses_toplevel_cache_control(request_body, valves=effective_valves)
        _debug_print_request(headers, request_body, logger=self.logger)
        url = base_url.rstrip("/") + "/responses"

        workers = max(1, min(int(workers or 1), 8))
        chunk_queue_size = max(0, int(chunk_queue_maxsize))
        event_queue_size = max(0, int(event_queue_maxsize))
        chunk_queue: asyncio.Queue[tuple[int | None, Any]] = asyncio.Queue(maxsize=chunk_queue_size)
        event_queue: asyncio.Queue[tuple[int | None, Any]] = asyncio.Queue(maxsize=event_queue_size)
        chunk_sentinel = (None, None)
        idle_flush_seconds = float(idle_flush_ms) / 1000 if idle_flush_ms > 0 else None
        passthrough_deltas = delta_char_limit <= 0 and idle_flush_ms <= 0
        requested_model = request_body.get("model")
        charge_failure = _once_only_charge(self._pipe, breaker_key, charge_holder)
        producer_reported = False

        def _raise_in_band_error(current: dict[str, Any] | None):
            if current is None:
                return
            return self._pipe._ensure_error_formatter()._extract_streaming_error_event(current, requested_model)

        async def _send_worker_sentinels() -> None:
            for _ in range(workers):
                await chunk_queue.put(chunk_sentinel)

        @timed
        async def _producer() -> None:
            nonlocal producer_reported
            seq = 0
            first_chunk_received = False

            async def _put_seq(event_obj: Any) -> None:
                nonlocal seq
                await chunk_queue.put((seq, event_obj))
                seq += 1

            def _retry_streaming(retry_state) -> bool:
                exc = retry_state.outcome.exception() if retry_state.outcome else None
                return _should_retry_stream(delivered_any, exc)

            def _probe_inband(parsed: Any) -> None:
                if not isinstance(parsed, dict):
                    return
                reported_error = _raise_in_band_error(parsed)
                if reported_error is not None:
                    raise reported_error

            retryer = _transient_retry_policy(effective_valves, retry=_retry_streaming)
            try:
                async with _count_failed_call(self._pipe, breaker_key, charge_failure):
                    async for attempt in retryer:
                        with attempt:
                            queued_any = False
                            saw_data_line = False
                            delivered_any = False
                            body_complete = False
                            accepted = False
                            buf = bytearray()
                            scanned = 0
                            excerpt = bytearray()
                            event_data_parts: list[bytes] = []
                            stream_complete = False
                            held: list[Any] = []
                            first_event_queued = False

                            async def _dispatch(data_blob: bytes, _held: list[Any] = held) -> bool:
                                nonlocal queued_any, delivered_any, first_event_queued
                                event_obj = _decode_frame(data_blob, self.logger)
                                if event_obj is _UNREADABLE:
                                    return False
                                queued_any = True
                                if not first_event_queued:
                                    first_event_queued = True
                                    timing_mark("producer_first_event_queued")
                                if not delivered_any:  # noqa: B023 - shares the attempt's state by design
                                    _probe_inband(event_obj)
                                if _responses_event_is_user_visible(event_obj):
                                    await _emit(event_obj, True, _held)
                                else:
                                    _held.append(event_obj)
                                return True

                            async def _emit(event_obj: Any, visible: bool, _held: list[Any] = held) -> None:
                                nonlocal delivered_any
                                for pending in _held:
                                    await _put_seq(pending)
                                _held.clear()
                                if visible:
                                    delivered_any = True
                                await _put_seq(event_obj)

                            try:
                                timing_mark("responses_http_request_start")
                                async with session.post(
                                    url, json=request_body, headers=headers,
                                    timeout=self._timeout(effective_valves),
                                ) as resp:
                                    timing_mark("responses_http_headers_received")
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
                                            requested_model=request_body.get("model"),
                                            extra_metadata=extra_meta or None,
                                        )

                                    accepted = True
                                    chunk_count = 0
                                    total_bytes = 0
                                    async for chunk in resp.content.iter_chunked(4096):
                                        chunk_count += 1
                                        total_bytes += len(chunk)
                                        if not first_chunk_received:
                                            first_chunk_received = True
                                            timing_mark("responses_first_chunk")
                                        view = memoryview(chunk)
                                        buf.extend(view)
                                        if len(excerpt) < _BODY_EXCERPT_CHARS:
                                            excerpt.extend(view[: _BODY_EXCERPT_CHARS - len(excerpt)])
                                        sse_lines, scanned = _split_sse_lines(buf, scanned)
                                        for stripped in sse_lines:
                                            if not stripped:
                                                if event_data_parts:
                                                    data_blob = b"\n".join(event_data_parts).strip()
                                                    event_data_parts.clear()
                                                    if not data_blob:
                                                        continue
                                                    if data_blob == _RESPONSES_SSE_DONE_SENTINEL:
                                                        stream_complete = True
                                                        timing_mark("responses_stream_done")
                                                        break
                                                    await _dispatch(data_blob)
                                                continue
                                            if stripped.startswith(b":"):
                                                continue
                                            if stripped.startswith(b"data:"):
                                                saw_data_line = True
                                                payload = bytes(stripped[5:].lstrip())
                                                if payload == _RESPONSES_SSE_DONE_SENTINEL:
                                                    if event_data_parts:
                                                        data_blob = b"\n".join(event_data_parts).strip()
                                                        event_data_parts.clear()
                                                        if data_blob and data_blob != _RESPONSES_SSE_DONE_SENTINEL:
                                                            await _dispatch(data_blob)
                                                    stream_complete = True
                                                    timing_mark("responses_stream_done")
                                                    break
                                                event_data_parts.append(payload)
                                                continue
                                        if stream_complete:
                                            break

                                    timing_mark(
                                        f"responses_body_read_chunks_{chunk_count}_bytes_{total_bytes}"
                                    )

                                body_complete = True

                                if not stream_complete and buf:
                                    tail_line = bytes(buf).strip()
                                    del buf[:]
                                    scanned = 0
                                    if tail_line.startswith(b"data:"):
                                        saw_data_line = True
                                        event_data_parts.append(bytes(tail_line[5:].lstrip()))

                                if event_data_parts and not stream_complete:
                                    data_blob = event_data_parts[0].strip()
                                    trailing = [p.strip() for p in event_data_parts[1:]]
                                    event_data_parts.clear()
                                    if data_blob == _RESPONSES_SSE_DONE_SENTINEL:
                                        stream_complete = True
                                        timing_mark("responses_stream_done")
                                    else:
                                        for blob in (data_blob, *trailing):
                                            if not blob or blob == _RESPONSES_SSE_DONE_SENTINEL:
                                                continue
                                            await _dispatch(blob)
                                if not queued_any:
                                    if saw_data_line or not chunk_count:
                                        raise aiohttp.ClientPayloadError("OpenRouter closed the stream before sending anything")
                                    raise UpstreamBodyUnreadable(
                                        endpoint="/responses",
                                        body_excerpt=excerpt.decode("utf-8", "replace"),
                                        content_type=resp.headers.get("Content-Type"),
                                    )
                                if not delivered_any:
                                    raise aiohttp.ClientPayloadError("OpenRouter sent no user-visible events on /responses")
                            except Exception as producer_exc:
                                is_auth_failure = isinstance(
                                    producer_exc, OpenRouterAPIError
                                ) and is_sign_in_failure(producer_exc)
                                if is_auth_failure:
                                    self._pipe._note_auth_failure()
                                    self.logger.warning(
                                        "Producer encountered auth error while streaming from OpenRouter: %s",
                                        _data_url_log_subject(str(producer_exc)),
                                    )
                                else:
                                    self.logger.exception(
                                        "Producer encountered error while streaming from OpenRouter"
                                    )
                                producer_reported = True
                                if accepted and not body_complete and not isinstance(
                                    producer_exc, OpenRouterAPIError
                                ):
                                    raise AcceptedResponseLostBody(str(producer_exc)) from producer_exc
                                raise
                            for pending in held:
                                await _put_seq(pending)
                            held.clear()
                            if stream_complete:
                                break
            finally:
                if _task_is_cancelling():
                    raise asyncio.CancelledError()
                await _send_worker_sentinels()

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
                    # Non-spammy chunk queue monitoring
                    if self._pipe._should_warn_event_queue_backlog(
                        chunk_queue.qsize(), chunk_queue_warn_size
                    ):
                        level = warn_level(
                            _warned_queue_backlog,
                            _backlog_cause("chunk_queue", _backlog_request_id),
                            cooldown_s=30.0,
                        )
                        if level != logging.DEBUG or _backlog_debug_due(
                            "chunk_queue", _backlog_request_id
                        ):
                            self.logger.log(
                                level,
                                "Chunk queue backlog high: %d items (session=%s)",
                                chunk_queue.qsize(),
                                loggable_session_id(SessionLogger.session_id.get()) or "unknown",
                            )
                    try:
                        if seq is None:
                            break
                        if not worker_first_event_queued:
                            worker_first_event_queued = True
                            timing_mark("worker_first_event_to_queue")
                        await event_queue.put((seq, data))
                    finally:
                        chunk_queue.task_done()
            finally:
                if _task_is_cancelling():
                    raise asyncio.CancelledError()
                await event_queue.put((None, None))

        producer_task = asyncio.create_task(_producer(), name="openrouter-sse-producer")
        worker_tasks = [
            asyncio.create_task(_worker(idx), name=f"openrouter-sse-worker-{idx}")
            for idx in range(workers)
        ]

        pending_events: dict[int, dict[str, Any] | None] = {}
        next_seq = 0
        done_workers = 0
        stream_ended = False
        coalescer = NagleCoalescer(min_flush_chars=nagle_min_chars)

        first_event_from_queue = False
        first_yield_done = False

        try:
            while True:
                timeout = idle_flush_timeout(coalescer, idle_flush_seconds)
                timed_out = False
                seq: int | None = None
                event: dict[str, Any] | None = None
                if timeout is not None:
                    try:
                        seq, event = await asyncio.wait_for(event_queue.get(), timeout=timeout)
                    except TimeoutError:
                        timed_out = True
                else:
                    seq, event = await event_queue.get()
                if not first_event_from_queue and seq is not None:
                    first_event_from_queue = True
                    timing_mark("consumer_first_event_from_queue")

                if timed_out:
                    idle_queue: list[dict[str, Any]] = []
                    coalescer.flush_all_to(idle_queue)
                    for item in idle_queue:
                        if not first_yield_done:
                            first_yield_done = True
                            timing_mark("adapter_first_yield")
                        yield item
                    continue

                event_queue.task_done()

                if self._pipe._should_warn_event_queue_backlog(
                    event_queue.qsize(), event_queue_warn_size
                ):
                    level = warn_level(
                        _warned_queue_backlog,
                        _backlog_cause("event_queue", _backlog_request_id),
                        cooldown_s=30.0,
                    )
                    if level != logging.DEBUG or _backlog_debug_due(
                        "event_queue", _backlog_request_id
                    ):
                        self.logger.log(
                            level,
                            "Event queue backlog high: %d items (session=%s)",
                            event_queue.qsize(),
                            loggable_session_id(SessionLogger.session_id.get()) or "unknown",
                        )

                if seq is None:
                    done_workers += 1
                    if done_workers >= workers and not pending_events:
                        break
                    continue

                pending_events[seq] = event

                yield_queue: list[dict[str, Any]] = []

                while next_seq in pending_events:
                    current = pending_events.pop(next_seq)
                    next_seq += 1
                    if not _is_ordered_object(current):
                        if current is not None:
                            self.logger.debug(
                                "Discarding a non-object SSE frame: %s", type(current).__name__
                            )
                        continue
                    streaming_error = _raise_in_band_error(current)
                    if streaming_error is not None:
                        charge_failure()
                        tail: list[dict[str, Any]] = []
                        coalescer.flush_all_to(tail)
                        for item in tail:
                            yield item
                        raise streaming_error
                    stream_ended = stream_ended or _terminal_frame_is_readable(current)
                    coalescer.process_event(current, yield_queue, passthrough=passthrough_deltas)

                drained = 0
                while not yield_queue and drained < _MAX_DRAIN_PER_CYCLE:
                    try:
                        extra_seq, extra_event = event_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    event_queue.task_done()
                    if extra_seq is None:
                        done_workers += 1
                        continue
                    drained += 1
                    pending_events[extra_seq] = extra_event
                    while next_seq in pending_events:
                        current = pending_events.pop(next_seq)
                        next_seq += 1
                        if not _is_ordered_object(current):
                            if current is not None:
                                self.logger.debug(
                                    "Discarding a non-object SSE frame: %s", type(current).__name__
                                )
                            continue
                        streaming_error = _raise_in_band_error(current)
                        if streaming_error is not None:
                            charge_failure()
                            tail: list[dict[str, Any]] = []
                            coalescer.flush_all_to(tail)
                            for item in tail:
                                yield item
                            raise streaming_error
                        stream_ended = stream_ended or _terminal_frame_is_readable(current)
                        coalescer.process_event(current, yield_queue, passthrough=passthrough_deltas)

                coalescer.flush_all_to(yield_queue, force=False)

                for item in yield_queue:
                    if not first_yield_done:
                        first_yield_done = True
                        timing_mark("adapter_first_yield")
                    yield item

                if done_workers >= workers and not pending_events:
                    break

            final_queue: list[dict[str, Any]] = []
            coalescer.flush_all_to(final_queue)
            for item in final_queue:
                yield item

            await producer_task
            if not stream_ended:
                charge_failure()
        finally:
            if not producer_task.done():
                producer_task.cancel()
            for task in worker_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(producer_task, *worker_tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if idx == 0 and producer_reported:
                    continue
                if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                    task_name = "producer" if idx == 0 else f"worker-{idx - 1}"
                    self.logger.error(
                        "SSE %s task failed during cleanup: %s",
                        task_name,
                        _data_url_log_subject(str(result)),
                        exc_info=result,
                    )


    @timed
    async def send_openai_responses_nonstreaming_request(
        self,
        session: aiohttp.ClientSession,
        request_params: dict[str, Any],
        api_key: str,
        base_url: str,
        *,
        valves: Pipe.Valves | None = None,
        breaker_key: str | None = None,
        user: Any = None,
        owui_chat_id: str | None = None,
        transient_retry: bool = True,
        files_inlined: bool = False,
        charge_holder: _FailureCharge | None = None,
    ) -> dict[str, Any]:
        """Send a blocking request to the Responses API and return the JSON payload."""
        effective_valves = valves or self._pipe.valves
        if not files_inlined:
            request_params = await self._pipe._file_gateway.inline_internal_responses_input_files(
                request_params,
                chunk_size=effective_valves.IMAGE_UPLOAD_CHUNK_BYTES,
                max_bytes=effective_valves.BASE64_MAX_SIZE_MB * 1024 * 1024,
                user=user,
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            "HTTP-Referer": _select_openrouter_http_referer(effective_valves),
        }
        self._pipe._maybe_apply_anthropic_beta_headers(
            headers,
            request_params.get("model"),
            valves=effective_valves,
        )
        headers = _apply_owui_forward_user_headers(headers, user, owui_chat_id)
        _maybe_apply_responses_toplevel_cache_control(request_params, valves=effective_valves)
        _debug_print_request(headers, request_params, logger=self.logger)
        url = base_url.rstrip("/") + "/responses"

        retryer = _transient_retry_policy(effective_valves, retry=_retry_nonstreaming)
        if not transient_retry:
            retryer.stop = stop_after_attempt(1)

        async with _count_failed_call(self._pipe, breaker_key, charge_holder=charge_holder):
            async for attempt in retryer:
                with attempt:
                    async with session.post(
                        url, json=request_params, headers=headers,
                        timeout=self._timeout(effective_valves),
                    ) as resp:
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
                                requested_model=request_params.get("model"),
                                extra_metadata=extra_meta or None,
                            )
                        payload = await _decode_json_body(resp, self.logger, "/responses")
                        if not isinstance(payload, dict):
                            raise _body_not_an_object("/responses", payload, resp)
                        _debug_print_response(payload, logger=self.logger)
                        reported_error = self._pipe._ensure_error_formatter()._extract_streaming_error_event(
                            payload, request_params.get("model")
                        )
                        if reported_error is not None:
                            raise reported_error
                        output = payload.get("output")
                        if "output" not in payload or not isinstance(output, (list, type(None))):
                            raise aiohttp.ClientPayloadError(
                                "OpenRouter returned 200 with no output on /responses"
                            )
                        if output is None or not output:
                            raise EmptyAnswerError(
                                "The model returned an empty answer (no output items) on /responses",
                                requested_model=request_params.get("model"),
                            )
                        return payload
        self.logger.error("Responses API call completed without yielding a response body; returning empty payload.")
        return {}

