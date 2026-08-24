# Streaming Pipeline & Emitters

This document describes how the pipe converts OpenRouter Responses **SSE** (Server-Sent Events) into the incremental UI updates Open WebUI expects: text deltas, status updates, citations, notifications, and a final completion frame.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Concurrency](concurrency_controls_and_resilience.md) · [Errors](error_handling_and_user_experience.md)

---

## 1. High-level architecture

The streaming path is implemented by `send_openai_responses_streaming_request()` and the streaming loop that consumes it.

Conceptually:

```text
OpenRouter SSE (aiohttp) → chunk queue → JSON parse workers → ordered event drain → emitters → Open WebUI client
```

Key design goals:
- Preserve event ordering, even with multiple parser workers.
- Avoid unbounded memory growth under load (tunable queue sizes).
- Degrade predictably under backpressure rather than silently dropping events.

---

## 2. SSE ingestion and ordering

### Producer (SSE reader)
The producer reads the OpenRouter `text/event-stream` response and:
- extracts `data:` lines into full SSE “data blobs”
- assigns an incrementing sequence number (`seq`)
- enqueues `(seq, data_blob)` into the **chunk queue**

### Workers (JSON parsers)
The pipe spawns `SSE_WORKERS_PER_REQUEST` worker tasks. Each worker:
- dequeues `(seq, data_blob)` from the chunk queue
- parses JSON
- enqueues `(seq, event_dict)` into the **event queue**

### Drain (ordered emission)
The drain loop:
- stores out-of-order events in a `pending_events` map
- emits events strictly in ascending `seq` order
- raises a structured error if it detects a streaming error event

---

## 3. Queue sizing, backpressure, and deadlocks

The streaming pipeline uses two queues with valve controls:

- `STREAMING_CHUNK_QUEUE_MAXSIZE`: raw SSE JSON blobs (pre-parse)
- `STREAMING_EVENT_QUEUE_MAXSIZE`: parsed JSON events (post-parse)

Defaults are `0` (unbounded) for both queues.

**Warning:** Very small bounded sizes can create deadlock-style stalls in tool-heavy or persistence-heavy workloads (slow drain → event queue fills → workers block → chunk queue fills → producer blocks). The valve descriptions explicitly warn that bounded values under a few hundred can hang under load.

Monitoring:
- `STREAMING_CHUNK_QUEUE_WARN_SIZE` emits a backend warning (rate-limited) when the raw-chunk queue backlog is high.
- `STREAMING_EVENT_QUEUE_WARN_SIZE` emits a backend warning (rate-limited) when the event queue backlog is high.

### Middleware streaming bridge queue (Open WebUI generator)

When `pipe(..., body={"stream": true})` is used, the pipe returns an async generator that yields Open WebUI-compatible chunks. Internally, request-scoped tasks enqueue items into a per-request queue that the generator drains.

This bridge is controlled by:
- `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` (default `0` = unbounded)
- `MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS` (only applies when maxsize > 0)

Rationale: a stalled or slow client should not allow unbounded memory growth, and teardown should not hang while attempting to enqueue the final sentinel.

---

## 4. What Open WebUI receives (emitters and event types)

The pipe emits Open WebUI-compatible events via the provided `event_emitter` callable.

Common event types:

| Event type | Purpose |
| --- | --- |
| `chat:message` | Sets the visible assistant message to `content`, replacing what is there (whole-text snapshots). |
| `chat:message:delta` | Appends `content` to the visible assistant message. |
| `chat:completion` | Final frame that ends the request; may include `usage` and must include `content` (even when empty). |
| `status` | Progress and warning messages displayed as status updates. |
| `source` | Normalized citation payloads (documents/metadata/source). |
| `notification` | Toast-style notifications (info/success/warning/error). |

Which of these survives a page reload depends entirely on whether the turn is streaming, and the two legs do not share a channel:

**Streaming on.** The pipe does not hand its events to Open WebUI as events at all. It returns an async generator, and its own translator converts each event into the OpenAI-style stream chunks Open WebUI's streaming handler accumulates. Both `chat:message` and `chat:message:delta` become answer text there — a delta contributes its `content` verbatim, and a `chat:message` contributes only the part of its snapshot that has not been sent yet, so alternating the two on one turn does not duplicate text. That subtraction is conditional: the translator forwards the remainder **only when the snapshot starts with everything already sent**, and forwards nothing at all when it does not. So a card — an error card, a notice, any block the pipe appends to an answer in progress — goes out as a snapshot of the **whole message**, the answer so far joined to the card, never as the card alone. A bare card is a non-prefix write the moment any text has streamed, and the translator drops it: the user never sees it. Where a card instead reaches Open WebUI as an event rather than through the translator, a bare one is worse than invisible — `chat:message` assigns there, per the table above, so it would overwrite the partial answer rather than follow it. `chat:completion` is **not** answer text: a frame carrying `content` is forwarded out of band, one carrying `error` or `usage` is forwarded as an error or usage record, and a frame carrying none of the three is dropped. So a whole answer sent only as `chat:completion` arrives as no text at all.

**Streaming off.** The translator does not exist; the events go straight to Open WebUI's socket emitter, which forwards them to the browser and writes only a subset to the database. `status`, `message`, `replace`, `embeds`, `files` and `source`/`citation` are written. `chat:message`, `chat:message:delta` and `chat:completion` are **not** — the browser shows them and the database never hears about them. What Open WebUI stores on this leg is the value the pipe RETURNS. An empty return also skips the outlet filters and the follow-up tasks that ride the same branch, so the turn reloads blank *and* the chat never gets a title.

The rule that covers both legs: **an answer must be both shown and returned.** Showing it satisfies the live view on either leg; returning it satisfies persistence with streaming off and costs nothing with streaming on, where the return value is discarded. A card that is only emitted is a card the user loses on reload; a card that is only returned is a card that appears late.

Notes:
- Answer text streams as `chat:message:delta` frames, and the turn ends with a `chat:completion` frame. The whole-message `chat:message` snapshot is the channel for cards, not for incremental text.
- An answer produced whole rather than streamed — a generated image, a finished video, a help panel, an error card — must go out as `chat:message` or `chat:message:delta`, never as `chat:completion` alone, and must also be the return value. Which of the two depends on what has already been sent: a `chat:message:delta` carries exactly the text appended to the running message and nothing that preceded it, while a `chat:message` carries the running message and the new block together. Neither may carry the new block by itself — a delta that repeats the answer duplicates it on screen, and a snapshot that omits the answer is the non-prefix write the translator drops.
- With streaming off, the wrapper the non-streaming path puts around the emitter suppresses `chat:message` and `chat:message:delta` outright, because the answer is travelling home as the return value instead. Anything raised inside the response loop that is only emitted as a card is therefore invisible on that leg; it has to come back as the return value to be seen at all.
- When `SHOW_FINAL_USAGE_STATUS=True`, the pipe formats a final status description using usage/cost/tokens when present.

---

## 5. Streaming errors and how they surface

OpenRouter can report failures mid-stream. The pipe detects streaming error payloads (for example `response.failed` or events carrying an `error` block) and converts them into an `OpenRouterAPIError`.

That error is then handled by the same OpenRouter template system described in:
- [Error Handling & User Experience](error_handling_and_user_experience.md)

This keeps user-facing failures consistent between streaming and non-streaming calls.

---

## 6. Nagle-inspired adaptive delta coalescing

### Problem

During heavy reasoning turns, OWUI's `serialize_output()` is called on every event emitted by the pipe. Each call rebuilds the entire HTML output — an O(n*m) cost. When the model streams hundreds of small reasoning deltas per second, OWUI's UI freezes because it cannot keep up with per-delta processing.

### Solution: RFC 896 Nagle algorithm adapted for async generators

The pipe uses a Nagle-inspired coalescing strategy (implemented in `streaming/nagle_coalescer.py`) that **self-regulates based on consumer backpressure** — no tuning parameters needed for the core mechanism.

The key insight from RFC 896: *”inhibit the sending of new TCP segments when previously transmitted data remains unacknowledged.”* In our async generator pipeline:

| TCP concept | Async generator equivalent |
|---|---|
| Send segment | `yield` event to consumer |
| ACK received | Consumer calls `__anext__()` — generator resumes |
| Data in flight | Generator suspended at `yield` |
| New data arrives | Events arriving in queue from OpenRouter |

**Behavior:** When OWUI is fast, events pass through with minimal latency (near 1:1). When OWUI is slow (processing a previous yield), events accumulate in the queue and are delivered in larger batches — automatically reducing the number of expensive `serialize_output()` calls.

### Multi-buffer architecture

Two independent buffers prevent reasoning floods from blocking text output:

```text
events  -->  [bounded micro-drain]  -->  process in order
                                       |
                    +------------------+------------------+
                    v                  v                  v
              text_buffer       reasoning_buffer    yield immediately
              (output_text)     (reasoning_text,    (non-batchable:
                                 reasoning)          tool calls,
                                                     structural, etc.)
```

Flush triggers:
- **Type boundary** — text delta while reasoning has content (or vice versa): force flush
- **Item_id boundary** — reasoning delta with different `item_id`: force flush
- **Non-batchable event** — flush both buffers, pass event through
- **Idle timeout** — `STREAMING_IDLE_FLUSH_MS` flushes buffers when producer pauses
- **End of micro-drain cycle** — flush if buffered chars >= `STREAMING_NAGLE_MIN_FLUSH_CHARS` (drain is bounded to 32 events and only runs while no output has been produced yet)
- **End of stream** — unconditional final flush

### Coverage: both streaming paths

- **Responses API path** (`responses_adapter.py`): Uses `NagleCoalescer` directly in the existing consumer loop (already has `event_queue` with workers).
- **Chat Completions API path** (`chat_completions_adapter.py`): Wrapped with `nagle_coalesce_stream()` which creates a lightweight pump task + queue + bounded micro-drain.

### Valves

| Valve | Default | Effect |
|---|---|---|
| `STREAMING_DELTA_CHAR_LIMIT` | 256 | **Toggle.** `> 0` enables Nagle coalescing. `0` (with `IDLE_FLUSH_MS=0`) = passthrough mode (1:1 emission). |
| `STREAMING_IDLE_FLUSH_MS` | 30 | Idle timeout (ms). Ensures buffered content is delivered even when the producer pauses. |
| `STREAMING_NAGLE_MIN_FLUSH_CHARS` | 3 | Minimum chars before end-of-cycle flush. `3` smooths single-char jitter. `1` = pure Nagle. `5-10` = aggressive event reduction. |

**Operator note:** The default `NAGLE_MIN_FLUSH_CHARS=3` eliminates single-character jitter in low-backpressure phases while keeping latency imperceptible (the 30ms idle timeout guarantees delivery). Set to `1` for pure Nagle, or `5-10` if OWUI still struggles during heavy reasoning.

---

## 7. Tuning checklist

- Increase throughput (at cost of per-request CPU): raise `SSE_WORKERS_PER_REQUEST` (up to the code-enforced cap).
- Avoid stalls under tool-heavy loads: keep `STREAMING_CHUNK_QUEUE_MAXSIZE=0` and `STREAMING_EVENT_QUEUE_MAXSIZE=0` unless you have a measured reason to bound them.
- Improve observability: set `STREAMING_EVENT_QUEUE_WARN_SIZE` low enough to signal stress early, but high enough to avoid constant warnings.
- Reduce UI freeze during heavy reasoning: increase `STREAMING_NAGLE_MIN_FLUSH_CHARS` (2-5 for moderate reduction, 5-10 for aggressive).
- Disable all coalescing (debugging): set `STREAMING_DELTA_CHAR_LIMIT=0` and `STREAMING_IDLE_FLUSH_MS=0` for passthrough mode.

All related defaults and ranges are listed in [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
