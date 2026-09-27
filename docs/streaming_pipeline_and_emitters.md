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

## 3. Queue sizing and backpressure

The streaming pipeline uses two queues with valve controls:

- `STREAMING_CHUNK_QUEUE_MAXSIZE`: raw SSE JSON blobs (pre-parse)
- `STREAMING_EVENT_QUEUE_MAXSIZE`: parsed JSON events (post-parse)

Defaults are `0` (unbounded) for both queues.

**Warning:** A bounded queue applies backpressure: when it fills, the pipe stops reading from OpenRouter until the backlog clears. The chain runs the other way from what it used to — a slow drain (tool-heavy or persistence-heavy workloads) → event queue fills → workers block → chunk queue fills → producer blocks on its next put → and the source stops being read. As long as the consumer keeps draining, the cost is added latency on a slow drain, not a stalled stream: the reply still ends on its own. A consumer that stops reading entirely, such as a closed browser tab, is held by that backpressure rather than ended.

Monitoring:
- `STREAMING_CHUNK_QUEUE_WARN_SIZE` emits a backend warning (rate-limited) when the raw-chunk queue backlog is high.
- `STREAMING_EVENT_QUEUE_WARN_SIZE` emits a backend warning (rate-limited) when the event queue backlog is high.

### Middleware streaming bridge queue (Open WebUI generator)

When `pipe(..., body={"stream": true})` is used, the pipe returns an async generator that yields Open WebUI-compatible chunks. Internally, request-scoped tasks enqueue items into a per-request queue that the generator drains.

This bridge is controlled by:
- `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` (default `0` = unbounded)
- `MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS` (only applies when maxsize > 0)

Rationale: a stalled or slow client should not allow unbounded memory growth, and teardown should not hang while attempting to enqueue the final sentinel.

Invariant: a job that reaches the worker's `try` puts the `None` terminator on its queue, so the generator always ends. This includes the emitter-construction path — a fault raised while building the middleware stream emitter is handled by the job's own guard, which puts the error chunk and then the terminator. Two exits sit above that `try` (`Semaphore unavailable`, and `CancelledError` before it) and put no terminator; the generator's own `future.done() and stream_queue.empty()` check at `pipe.py:1541` is what ends those, so such a job yields a silently empty stream rather than an error. A job that raised anywhere else before reaching that guard would leave the generator parked on `await queue.get()` forever, and the person's chat would hang with no reply and no error.

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

Each stretch of answer text streams into a message item that the pipe publishes first, under its own id: a `response.output_item.added` event carrying the empty message, then the text. Open WebUI puts a chunk's text into the reply's last item when that item is a message, so the text lands in the pipe's item, and Open WebUI names that item when it passes the text on to the browser. Every item the pipe publishes - a message, a tool card, a thinking box, a Fusion answer - also carries `output_index`, its place in the reply. The contract has a non-streamed half: with streaming off no deltas are published, so a stretch of answer text written before a later item is announced is published as its own message item first, at the index the record gave it, so the two legs fold the same list. A message segment is announced even though its deltas are not. Without `output_index`, the browser puts an item before the last one it holds, which showed a card above the text written before it. The pipe's closing record reuses the same ids. That matters because Open WebUI's browser merges the record by id into what it already shows: a record under fresh ids showed the answer twice until Open WebUI's final update.

**Streaming off.** The translator does not exist; the events go straight to Open WebUI's socket emitter, which forwards them to the browser and writes only a subset to the database. `status`, `message`, `replace`, `embeds`, `files` and `source`/`citation` are written. `chat:message`, `chat:message:delta` and `chat:completion` are **not** — the browser shows them and the database never hears about them. What Open WebUI stores on this leg is the value the pipe RETURNS. For a finished chat reply that is a completion carrying the reply's text and its structured record (`output`), which Open WebUI saves exactly as it saves a streamed reply's record, so tool cards and thinking survive a reload. That includes a reply that ended in an error: its record carries the error card with the rest. A reply that produced nothing but text comes back as that text, error card included. An empty return also skips the outlet filters and the follow-up tasks that ride the same branch, so the turn reloads blank *and* the chat never gets a title.

The rule that covers both legs: **an answer must be both shown and returned.** Showing it satisfies the live view on either leg; returning it satisfies persistence with streaming off and costs nothing with streaming on, where the return value is discarded. A card that is only emitted is a card the user loses on reload; a card that is only returned is a card that appears late.

Notes:
- Answer text streams as `chat:message:delta` frames, and the turn ends with a `chat:completion` frame. The whole-message `chat:message` snapshot is the channel for cards, not for incremental text.
- An answer produced whole rather than streamed — a generated image, a finished video, a help panel, an error card — must go out as `chat:message` or `chat:message:delta`, never as `chat:completion` alone, and must also be the return value. Which of the two depends on what has already been sent: a `chat:message:delta` carries exactly the text appended to the running message and nothing that preceded it, while a `chat:message` carries the running message and the new block together. Neither may carry the new block by itself — a delta that repeats the answer duplicates it on screen, and a snapshot that omits the answer is the non-prefix write the translator drops.
- With streaming off, the wrapper the non-streaming path puts around the emitter suppresses `chat:message` and `chat:message:delta` outright: the answer's text is travelling home in the record, which is what the turn returns whenever any output item was published. Its text is not published as deltas, though the message *item* that carries it still is, whenever a later item would otherwise be announced ahead of it. Anything raised inside the response loop that is only emitted as a card is therefore invisible on that leg; it has to come back as the return value to be seen at all.
- When `SHOW_FINAL_USAGE_STATUS` resolves True — the reader's own copy where they have set it, otherwise the site default an administrator chooses — the pipe formats a final status description using usage/cost/tokens when present, and writes the cost segment only for a charge above zero.

### 4.1 The stored output array, and who owns it

Every finished chat reply hands Open WebUI an `output` array, which it saves against the assistant message: a
streamed reply publishes it in its closing `response.completed` event, and a non-streamed one returns it with its
text. It holds the reasoning items, the tool cards that were shown, and the messages. Its messages carry the ids of
the items their text streamed into, so Open WebUI's browser replaces the items it already shows rather than
adding a second copy. Everything later turns replay comes from there, so who writes it matters.

Open WebUI's bookkeeping depends on what it is doing, and the pipe has to match it:

- **Continuing a message.** Open WebUI sets the message's existing stored output aside at stream start and puts
  it back in front afterwards. It already holds those items, so the pipe must publish only this generation's:
  republishing would store the earlier generation twice - and with it every hidden marker it carries, so the
  replayed reasoning would double as well. The pipe recognises this by the message id the frontend sends only
  when continuing.
- **Its own tool loop.** Each time Open WebUI runs a round and calls the pipe back, it sets the turn's output
  aside the same way, but sends no such id. The pipe therefore reads the stored output on that path - and finds
  nothing, because Open WebUI writes a message's output once, when the message finishes. Measured on a live
  server: across a two-round turn the row carried no output at all until the moment it was marked done.

So the read exists for one caller only: a client posting to the completions endpoint directly with the id of a
message that already holds output. Open WebUI's own chat never produces that state, and for the direct caller
republishing is the right answer, because nothing else will put those items back.

One consequence is worth stating plainly, because it is a deliberate choice rather than an oversight: a tool call
left unfinished by Stop, on a message that is then continued, stays unfinished in Open WebUI's saved copy instead of
being repaired, so Open WebUI does not replay it. The events that would repair it cannot reach storage -
`response.output_item.added` matches ids only within the new output, and `response.output_item.done` replaces by
position - so an attempt to heal it would duplicate or corrupt the saved copy instead. The model still learns the
round's calls before the first one still running: Open WebUI saves their cards as finished and hands them back, and
with cards off the pipe's own copy does (not in a temporary chat, where the pipe keeps no copy), since the pipe writes
a round's calls when the round starts and each result when its call returns, in call order (see
[History Reconstruction & Context Replay](history_reconstruction_and_context.md), section 5.4).

---

## 5. Streaming errors and how they surface

OpenRouter sends `200 OK` as soon as a provider accepts a request, so a failure after that point arrives inside the answer. On both `/responses` and `/chat/completions`, the pipe detects such error payloads (for example `response.failed`, or a chunk carrying an `error` block) and converts them into an `OpenRouterAPIError`. A non-streaming reply whose body carries an `error`, at the top level or in its first choice, is treated the same way.

That error is then handled by the same OpenRouter template system described in:
- [Error Handling & User Experience](error_handling_and_user_experience.md)

This keeps user-facing failures consistent between streaming and non-streaming calls. Each such error counts once toward the user's request breaker (see [Concurrency Controls & Resilience](concurrency_controls_and_resilience.md)); errors in housekeeping tasks such as title generation are not counted.

An attempt that has already published something Open WebUI keeps is not retried: the reader keeps the thinking box or the tool card they were watching and the error card is appended below it. Only an attempt that has published nothing is handed back for a retry.

A stream can also stop before its final event, without an error. If no event at all has arrived, the attempt counts as a failed call and is retried, up to three attempts in all; if every attempt comes back empty, the request ends with `CONNECTION_ERROR_TEMPLATE`, or with the `STREAM_INTERRUPTED_TEMPLATE` notice if answer text arrived earlier in the same reply. If anything has arrived, even just the event every stream opens with, whatever text has streamed (possibly none) is kept, the `STREAM_INTERRUPTED_TEMPLATE` notice is appended, and the call counts as a failed one.

A connection that drops or times out is handled identically on both transports, and each failed attempt counts once. If this happens before any answer text has arrived, a streaming attempt that has sent nothing is retried; if the request still fails, it ends with `NETWORK_TIMEOUT_TEMPLATE` for a timeout or `CONNECTION_ERROR_TEMPLATE` for a failed connection. If it happens after answer text has arrived, that text is kept and the `STREAM_INTERRUPTED_TEMPLATE` notice follows it. A stream that ends with `response.incomplete` is a finished answer, not an interruption: its usage is reported as usual, and a warning says the answer ended incomplete.
A retryable rejection hands the turn back to the orchestrator, which re-runs it on the same turn and the same emitter, and Open WebUI never discards what it has already accumulated. So a turn that has published an **accumulated** frame - `chat:message:delta` or `chat:tool_calls` - ends in a card rather than being retried: the reader never sees two attempts of one turn spliced together. The barrier stops the *second* publish, not the first, so the abandoned tool call stays in Open WebUI's list at `status: in_progress` with half its arguments and the turn ends on a card; that is no duplication, not no residue. `status` and `source` frames are additive rather than accumulated, and both are persisted, so a retried turn can carry two attempts' notices in its status history. For `chat:tool_calls`, the gate is the streaming leg, because there the pipe's own translator rewrites it into the OpenAI chunk Open WebUI folds; with streaming off the wrapper suppresses only `chat:message` and `chat:message:delta`, and `chat:tool_calls` leaves the pipe and is discarded by Open WebUI's own socket emitter (backend `socket/main.py:1178-1265`, frontend `Chat.svelte:1482`) - the string occurs nowhere in Open WebUI - so a hand-back on that leg costs nothing. `chat:message:delta` is **not** gated: a text delta ends the turn on either leg, including the non-streaming leg where the wrapper drops the frame, so a hand-back there costs a retry that would have succeeded. The rewrite itself is not covered by the test suite: were `chat:tool_calls` ever given the reshape-or-drop treatment `chat:message:delta` already has, the same splice could return with the tests still green. See [Plugin System](plugin_system.md) for the three rejections that are retried.

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
- Bound memory or latency: keep `STREAMING_CHUNK_QUEUE_MAXSIZE=0` and `STREAMING_EVENT_QUEUE_MAXSIZE=0` unless you have a measured reason to bound them. As long as the consumer keeps draining, a bound trades a larger buffer for a slower drain, not for a stream that stops early.
- Improve observability: set `STREAMING_EVENT_QUEUE_WARN_SIZE` low enough to signal stress early, but high enough to avoid constant warnings.
- Reduce UI freeze during heavy reasoning: increase `STREAMING_NAGLE_MIN_FLUSH_CHARS` (2-5 for moderate reduction, 5-10 for aggressive).
- Disable all coalescing (debugging): set `STREAMING_DELTA_CHAR_LIMIT=0` and `STREAMING_IDLE_FLUSH_MS=0` for passthrough mode.

All related defaults and ranges are listed in [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
