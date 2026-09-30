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
- recognises the `[DONE]` sentinel **per line**, not per blob: a `data:` line whose payload is exactly `[DONE]` ends the stream, and any event still pending behind it is flushed first. An empty line separates SSE events, so a provider whose final `data: [DONE]` is not followed by one would otherwise leave that line in the pending parts, where the next join fuses it into the preceding event's blob and the whole blob fails to parse. A blob is never rewritten to strip the sentinel: a legitimate event whose JSON *contains* the literal would be silently corrupted.
- on the chat leg, an ending recovered by that rule is **delivered, and not marked**. The per-line rule is what makes such an ending reachable at all: a provider whose final `data: [DONE]` never got its blank line still ends the stream, and the events queued behind it are flushed first, so the turn emits its `response.completed` record and the text that arrived is delivered. The cut-off guard is `if not done and not tool_calls_completed:` — `done` is set by every `[DONE]` arm, so a recovered ending never reaches it: the record carries no `status`, no `incomplete_details`, and the breaker stays quiet, whether or not the ending carried a `finish_reason` (`tests/test_the_chat_reader_keeps_every_frame.py::test_a_tool_call_with_no_finish_reason_still_reaches_the_completed_event` pins this for a `[DONE]` ending that carries a tool call and no `finish_reason`: `assert recorder.call_count == 0`). A body that **simply stops**, with no `[DONE]` at all, is the opposite case: the guard fires, `_record_failed_call` counts it toward the breaker, and `cut_off` returns no terminal event at all, so a truncated answer is never reported as a completed one. The one thing it does still return is a refusal that was buffered before the cut, published as an `output_text` delta on the same join the clean exit uses; a cut-off turn publishes that delta and no `response.completed` and no `response.incomplete`. Only a `finish_reason` the provider actually sent marks a record — `max_output_tokens`, from the provider's own `length` termination — and that one is delivered **and** marked `status: "incomplete"` with `incomplete_details.reason: "max_output_tokens"`.
- assigns an incrementing sequence number (`seq`)
- enqueues `(seq, data_blob)` into the **chunk queue**

### Tool-call slots (the `/chat/completions` leg)
A provider that streams parallel tool calls is supposed to label every fragment with the call's `index`. The `/chat/completions` reader matches a frame to the call it continues by three keys, in order: the frame's `id`, then its `function.name`, then neither. **A frame carrying no `index`, no `id` and no `function.name` continues the newest call whose accumulated arguments do not yet parse** — the call the provider opened last and is still writing — and opens a new call only when no call is open. A keyless frame used to answer `max(slots) + 1` whenever more than one call was open, which minted one call, and one announced card, per continuation frame. A call is **closed the moment its arguments parse**: a closed call is not extended by a later keyless fragment, and it is not decoded again while its argument string is unchanged, so the completeness test costs a bounded number of bytes per call rather than one whole-argument parse per delta. (An accumulator whose last non-space character is not one a JSON value can end on — `}]`, a quote, a digit, `e`, `E`, `l`, `L` — is not decoded at all; a slot the explicit-`index` path appends to re-opens, because its bytes no longer match what was declared whole. `json.loads` also accepts `NaN` and `Infinity`, which end in characters no such set contains; that is unreachable for an object-shaped tool call.)

### End of body
The read buffer is consumed line by line, so when the body ends the buffer still
holds the bytes after the last newline. A `data:` frame the body ended without a
terminator is dispatched from that residual rather than dropped, and a complete
frame already pending at end of body is dispatched first, on its own — the two are
never joined into one blob, because a worker parses a blob as a single JSON unit and
the pending frame would be lost with the residual. A body cut mid-frame produces a
residual that is not valid JSON; it takes the same per-chunk parse path as any other
malformed frame and is discarded, so a truncated tail is not reported as a transport
fault unless nothing in the body decoded. A `[DONE]`-terminated body is unaffected.

That is a malformed frame **inside** a stream, and it is discarded. A malformed **whole
body** behind an accepted status is a different failure: nothing was ever streamed, the
provider answered with something that is not an OpenRouter response, and the fault belongs
to whatever rewrote it — a proxy, a CDN or a WAF. It is reported under
`SERVICE_ERROR_TEMPLATE`, never retried and never charged to the breaker, and it is
reported on the streaming leg too, where it arrives as a `ClientPayloadError` and the
connection template would otherwise claim OpenRouter closed the stream. A body that was
accepted and then **lost on the way** is not this case: nothing was rewritten, the
connection simply died before the body finished arriving, so the call is charged once and
is still not re-POSTed. See
[Errors and how they surface](error_handling_and_user_experience.md).

A `data:` line is normally dispatched on the blank line that follows it, but a provider may end its stream without one. Both readers therefore flush whatever is left over when the response ends — bytes that never got their newline, and blobs already split out but not yet dispatched. Each complete leftover frame is dispatched as its own event rather than joined with its neighbours, so every complete frame the provider sent survives the missing blank line; a frame that is itself truncated mid-JSON is still rejected and not delivered. On `/chat/completions` a flushed end-of-stream marker also ends the turn, so the call is not counted as failed; see [Streaming errors and how they surface](#5-streaming-errors-and-how-they-surface) for how that differs from `/responses`.

### Workers (JSON parsers)
The pipe spawns `SSE_WORKERS_PER_REQUEST` worker tasks. Each worker:
- dequeues `(seq, data_blob)` from the chunk queue
- parses JSON
- enqueues `(seq, event_dict)` into the **event queue**

### Drain (ordered emission)
The drain loop:
- stores out-of-order events in a `pending_events` map
- emits events strictly in ascending `seq` order
- discards any event that is not a JSON object (an array, a string, a number, a
  boolean) and does not pass it on, so one such frame does not take the reply down

Non-visible events are held only until the first user-visible event **or the end of the stream**, whichever comes first: the `/responses` adapter buffers them so a fallback can drop them, and drains what it holds when the stream ends normally rather than carrying it past the end. The exception path is not that drain — an attempt that fails still hands over to `/chat/completions` with its own events discarded.

The in-band error guard is a single shared function the consumer calls on every event it takes off the queue, whichever loop fetched it, so an error is raised and counted identically in either path rather than only in the ordered one.

A turn that ends — by `break` on the terminal event, by exhaustion, by an error or by cancellation — closes its own event iterator as the first act of the streaming loop's `finally`, so the producer, its `SSE_WORKERS_PER_REQUEST` workers and the aiohttp response are released at the end of the turn rather than whenever the garbage collector gets to them. The close is delegated, three generators deep, to the adapter whose `finally` owns that teardown; the release is therefore bounded to one turn's worth rather than instantaneous, because that teardown awaits its workers. The tasks are gone by the time the turn returns, not before.

---

## 3. Queue sizing and backpressure

The streaming pipeline uses two queues with valve controls:

- `STREAMING_CHUNK_QUEUE_MAXSIZE`: raw SSE JSON blobs (pre-parse)
- `STREAMING_EVENT_QUEUE_MAXSIZE`: parsed JSON events (post-parse)

Defaults are `0` (unbounded) for both queues.

**Warning:** A bounded queue applies backpressure: when it fills, the pipe stops reading from OpenRouter until the backlog clears. The chain runs the other way from what it used to — a slow drain (tool-heavy or persistence-heavy workloads) → event queue fills → workers block → chunk queue fills → producer blocks on its next put → and the source stops being read. As long as the consumer keeps draining, the cost is added latency on a slow drain, not a stalled stream: the reply still ends on its own. A consumer that stops reading entirely, such as a closed browser tab, does not hang the request: cancelling it tears the pipeline down and returns, and no producer or worker task is left behind.

Monitoring:
- `STREAMING_CHUNK_QUEUE_WARN_SIZE` emits one backend `WARNING` per request per queue when the raw-chunk queue backlog crosses it, and repeats it at `DEBUG` at most once every second while the backlog stays high; the bound is on how often a line is emitted, not on its level.
- `STREAMING_EVENT_QUEUE_WARN_SIZE` emits one backend `WARNING` per request per queue when the event queue backlog crosses it, and repeats it at `DEBUG` at most once every second while the backlog stays high; the bound is on how often a line is emitted, not on its level.

**The pre-first-output buffer is a third buffer, and neither valve reaches it.** On `/responses` the producer parks every frame the reader cannot see — `response.created`, `response.in_progress`, `response.queued`, `response.heartbeat` — in a plain list until the first user-visible frame arrives, so an attempt that is handed back for a fallback can discard them. Those frames have not been enqueued yet, so bounding the chunk queue does not bound them: with `STREAMING_CHUNK_QUEUE_MAXSIZE=1` the queue still never holds more than one item while the producer holds every frame the provider sent before the first visible one. **No valve reaches this buffer and it is uncapped today**; the two named above, and their monitoring valves, cover the queues only. The frames cannot be dropped to save it — the Fusion clock reads the `response.created` and `response.in_progress` frames (`streaming/streaming_core.py`) — and OpenRouter documents exactly two of them as a stream's opening, which is what a provider is expected to send. It holds lifecycle frames, and it is released in order, ahead of the visible frame that released it, on the first visible frame or at the end of the stream.

### Middleware streaming bridge queue (Open WebUI generator)

When `pipe(..., body={"stream": true})` is used, the pipe returns an async generator that yields Open WebUI-compatible chunks. Internally, request-scoped tasks enqueue items into a per-request queue that the generator drains.

This bridge is controlled by:
- `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` (default `0` = unbounded)
- `MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS` (only applies when maxsize > 0)

Rationale: a stalled or slow client should not allow unbounded memory growth, and teardown should not hang while attempting to enqueue the final sentinel.

Invariant: the generator ends on whichever comes first — the `None` terminator a job puts on its queue, or its future reaching a terminal state. The check is the `future.done() and stream_queue.empty()` test in the `_stream` generator of `Pipe._pipe_impl`, re-evaluated on every `asyncio.wait` that generator makes against the future, so it holds whether the consumer is parked inside the get or sitting between items, and at every `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` including the shipped `0`. **One** exit settles a request's future without putting a terminator: the drain that abandons a queue replaced under a hot reload (`_abandon_request_queue`). The other three sit above the job's own teardown, where nothing would put one for them — the admission refusal the dispatch worker hands a job it will not run (`_admission_failed`), the worker's own `Semaphore unavailable` arm, and the same arm inside `_execute_pipe_job` — so each calls `_put_missing_stream_terminator` on the job itself and ends its stream on the `None` rather than on the reader. That put is a best-effort `put_nowait`, so a capped buffer whose slots are all taken drops it rather than waiting for one, and the generator's own check is what ends such a stream. A job resolved that way yields a silently empty stream rather than an error, and Open WebUI's consumer (`functions.py:334-336`) has no timeout to rescue it — it reads whatever arrives and then emits the finish chunk and `[DONE]`. A frame already on the queue when the resolve lands is still yielded first. A job that reaches the worker's `try` puts the terminator itself, and that includes the emitter-construction path — a fault raised while building the middleware stream emitter is handled by the job's own guard, which puts the error chunk and then the terminator. `future.set_result` is a terminator in its own right, in `Pipe._execute_pipe_job`, and so is the `None` sentinel: whichever fires first decides whether the queue is drained or abandoned, which is why a streaming job defers its `set_result` until the request's tool workers have stopped, and with it the invariant holds on both queue sizes. That holds on a capped buffer because the failure item and the terminator are not put through the drop valve: the valve applies to incremental output only, and the items that report or end a turn are put past it on a ceiling of their own, which `MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS = 0` does not remove. A reader still looping frees the slot those two waits need; a reader that has stopped entirely makes both give up, which costs the close and the failure notice but never the end of the stream, because the generator's own check is what ends a job whose queue nobody drains. Every arm of the middleware adapter that carries a frame returns that frame's delivery verdict, so a drop reads the same way whichever arm lost it: under a bound, `False` on `chat:tool_calls` and on `response.*` alike, and on a multi-frame `chat:completion` a drop of any part is a drop of the frame. An arm that carried no frame returns `None`, which the same rule ("what counts as published is what has been handed to the consumer") reads as published — it published nothing and lost nothing, and the empty-`chat:tool_calls` arm is read live by `_publish_turn_frame`, so reporting a drop there would close the retry barrier for a frame that was never on its way.

A Stop at the end of a turn still publishes the terminal status, `response.completed` and `chat:completion(done=True)`: nothing in the finalisation `finally` is skipped, and the shielded session-log write still completes behind the shield. The three awaits that guard those publications are guarded as Open WebUI guards its own two shields, with `except (asyncio.CancelledError, Exception)`, so a cancellation delivered at any of them is absorbed there and the `finally` runs to its end. This is scoped to a Stop that lands *inside* the finalisation window; a turn the body already recorded as cancelled is still published as incomplete, with no `response.completed` and no `chat:completion(done=True)`.

---

## 4. What Open WebUI receives (emitters and event types)

The pipe emits Open WebUI-compatible events via the provided `event_emitter` callable.

**An emitter being attached does not mean there is a chat.** `main.py:1277-1306` always populates `chat_id`, `message_id` and `session_id`, and `functions.py:239` tests key *presence*, so a caller with no chat is given an emitter too. The truthiness rule on `chat_id`/`message_id` — the same one `utils/middleware.py:3286` uses — is what actually distinguishes a chat from an API caller; `__event_emitter__` is not a discriminator.

Common event types:

| Event type | Purpose |
| --- | --- |
| `chat:message` | Sets the visible assistant message to `content`, replacing what is there (whole-text snapshots). |
| `chat:message:delta` | Appends `content` to the visible assistant message. |
| `chat:completion` | Final frame that ends the request; may include `usage` and normally includes `content` (even when empty). On a **continuing** turn it carries no truthy `content`, because the browser assigns that field absolutely (`if (content && !output) message.content = content`) and a replacement write would overwrite the prefix the turn is continuing. It is withheld on a second condition too: a turn that has **published an item of its own** — a thinking box, a tool card, a tool result, or a Fusion answer — carries no bare `content`, because the browser assigns `content` to `message.content` absolutely, and on a streamed leg the visible view is the `output` array Open WebUI has already folded from the pipe's `response.*` events, so a bare answer string is a second, competing copy of text the array already owns. `content` is withheld, not replaced: `output` stays Open WebUI's to fold, because its backend reducer replaces rather than folds (`middleware.py:912`) and its translator only forwards a `chat:completion` whose `content` is a `str`. The frame is still published: it is what carries `done` and `usage`, and with `content` absent the translator forwards it for its `usage` alone. Note what that means for the record Open WebUI writes from it — a `{'usage': ...}` entry carrying no content at all, which is not a content write and must not be read as one. |
| `status` | Progress and warning messages displayed as status updates. Opened and closed within each attempt, so no line is left open across a retry. |
| `chat:message:error` | Carries `error.content`; honoured by Open WebUI's **channel** emitter, which stores `Error: <content>` and closes the message. Ignored by the socket emitter. On a channel the card it carries is reduced first: the requester's `session_id`, `user_id` and the text they wrote — `detail`, `sanitized_detail`, `reason`, `openrouter_message`, `upstream_message`, `moderation_reasons`, `flagged_excerpt`, `raw_body`, `metadata_json`, `provider_raw_json` — render as though empty, because every member of the room reads what this frame writes. `error_id`, the model, the provider, `openrouter_code` and `status_code` still render. A saved chat's card is not reduced. |
| `source` | Normalized citation payloads (documents/metadata/source). The pipe's sources are Open WebUI's own five citing tools (`fetch_url`, `view_file`, `view_knowledge_file`, `query_knowledge_files`, `query_chat_files`), the provider's own `url_citation` annotations, and OpenRouter Fusion item sources. No other tool's result produces one, under any name, alias or origin, on any endpoint or mode: the result still reaches the model and its card, but a link in it is never published as a source — except when Open WebUI's extractor cannot be imported, where the harvester fallback publishes what it finds in the result text of those same five tools. The end-of-turn write of this field is a **union** with what the message already holds (see [4.1](#41-the-stored-output-array-and-who-owns-it)), so it cannot remove a chip Open WebUI stored. A source Open WebUI's extractor produced is kept on the message alongside the ones the pipe tracks, because the terminal `sources` write carries every citation published this turn: when Open WebUI's citation extractor imports, a builtin tool's citation is published as a `source` frame and appended to the write's own list, so it survives a turn with no emitter too. The union is what additionally preserves a source the pipe never accumulated at all — one Open WebUI stored on the message before the write landed. Those five tools' sources are withheld as well when the model's `Citations` capability is off, which is the same verdict Open WebUI reaches from the same model row; on that tool path the chip is not stored by the pipe but by Open WebUI's own socket emitter, which appends each `source` frame to the saved message, so withholding the frame is what leaves the chat row alone. The provider's own annotations and Fusion item sources are not gated by that box, because Open WebUI does not gate its own. A citation type the pipe cannot render (anything ending in `_citation` other than `url_citation`) produces no source, and produces a `notification` naming the type exactly once per turn: on `/responses`, and on `/chat/completions` and on a fallback hop too. |
| `notification` | Toast-style notifications (info/success/warning/error). |

Which of these survives a page reload depends entirely on whether the turn is streaming, and the two legs do not share a channel. A third distinction runs across both: a `channel:` chat id is emitted to a **different emitter** entirely, one with its own, much smaller set of types. Open WebUI's channel emitter honours exactly `chat:completion`, `response:completion`, `files`/`chat:message:files` and `chat:message:error` — it has **no `chat:message` branch at all**, and it stores `data['content'] or get_output_text(output)`. So a card on a channel travels as a `chat:message:error` frame, which Open WebUI stores prefixed with `Error: `, followed by a closing `chat:completion` whose `content` is the whole message, which overwrites that prefix with clean text. Because the channel leg *replaces* the message rather than appending, the closing frame's content must be the entire message: a `chat:completion` with empty content on a channel writes `''` over everything the channel already held. Because a channel card is reduced and a saved-chat card is not, the same template reads differently in the two places: on a channel a line whose placeholder came out empty is omitted, so a card that exists only to show the session id or the flagged excerpt is simply not there. Not every channel request reaches the stream translation at all — the pipe's five pre-job refusals (a tripped circuit breaker, warmup, a missing stream queue, a full queue's 503, and a pre-enqueue failure) return before a stream queue exists and emit straight to the channel emitter, so they take the replace-not-append leg whatever the streaming setting is.

**Streaming on.** The pipe does not hand its events to Open WebUI as events at all. It returns an async generator, and its own translator converts each event into the OpenAI-style stream chunks Open WebUI's streaming handler accumulates. Both `chat:message` and `chat:message:delta` become answer text there — a delta contributes its `content` verbatim, and a `chat:message` contributes only the part of its snapshot that has not been sent yet, so alternating the two on one turn does not duplicate text. The subtraction is against **what the buffer accepted**, so a frame the drop valve refused is never recorded as sent and the next snapshot re-derives it rather than assuming it arrived. That subtraction is conditional: the translator forwards the remainder **only when the snapshot starts with everything already sent**, and forwards nothing at all when it does not. So a card — an error card, a notice, any block the pipe appends to an answer in progress — goes out as a snapshot of the **whole message**, the answer so far joined to the card, never as the card alone. A bare card is a non-prefix write the moment any text has streamed, and it is now forwarded as a **full replacement** rather than dropped: the user sees it. Two things make that true, and are worth stating separately because the record and the queue disagree about it. First, the replacement is delivered *through an accumulator that appends* — Open WebUI calls `append_output_text` and there is no replace path — so a snapshot sharing no prefix with what was sent renders as `already_sent + content`, not as `content`. That is the same outcome `join_answer_and_card` produces for the common case and is strictly better than showing nothing. Second, the asymmetry after a `fusion_open` reset: the *record* is correct, because the accumulator it is built from was reset with `assistant_message`, while the *queue* is not, so a user comparing a reloaded turn to the live one can see different text. On the `/chat/completions` transport a **provider refusal** — the sentence the model wrote instead of, or alongside, an answer — is forwarded as answer text: joined to the answer with a blank line when the same message carried content, and delivered as the whole reply when it did not, on both the streamed and the non-streamed leg. The one exception is a housekeeping task request, where Open WebUI takes the return value as the title, the summary or the query rather than as an answer a person reads: there the non-streamed leg fails the attempt on a non-blank `refusal` instead of folding it in, so the refusal never becomes that value. It is forwarded on every exit the turn can take, including a stream cut off before its terminal frame, and the blank-line join is the same one on each. It travels on `delta.content` because that is the only channel Open WebUI reads. When one turn carries both streamed `delta.refusal` fragments and a closing `message.refusal`, the closing whole is what is shown and the fragments are dropped: the two are never concatenated, and the whole is never a truncation of them, whichever order the two channels arrive in. A closing chunk that carries answer text on `choices[0].message.content` rather than on `delta.content` is read when no `delta.content` has been seen for the turn, and is not read again once one has: both the `str` and the list-of-`text`-parts spellings are accepted, matching what the pipe's own non-streamed leg already reads off a message. The tie-break is the same in both directions and for both fields: whichever channel carries a field first is the one the reader gets, and the other is suppressed for the rest of the turn, so a `delta.content` or `delta.refusal` fragment arriving after a `message.content` or a `message.refusal` is not written on top of it, whichever of the two the provider sends first. Without it, a reply whose answer lived only on `message` produced no `output_text` delta at all. Where a card instead reaches Open WebUI as an event rather than through the translator, a bare one is worse than invisible — `chat:message` assigns there, per the table above, so it would overwrite the partial answer rather than follow it. `chat:completion` is **not** answer text: a frame carrying `content` is forwarded out of band — the browser assigns that field, so it replaces the whole live message rather than appending to it, which is why a refused terminal frame costs the reader a real write and why the next snapshot has to carry the tail it was meant to restore — one carrying `error` or `usage` is forwarded as an error or usage record, and a frame carrying none of the three is dropped. The drop valve applies to incremental output only; the items that report or end a turn — this frame's `error` and `usage` records alongside the failure item and the terminator — are put past it. So a whole answer sent only as `chat:completion` arrives as no text at all.

Each stretch of answer text streams into a message item that the pipe publishes first, under its own id: a `response.output_item.added` event carrying the empty message, then the text. Open WebUI puts a chunk's text into the reply's last item when that item is a message, so the text lands in the pipe's item, and Open WebUI names that item when it passes the text on to the browser. Every item the pipe publishes - a message, a tool card, a thinking box, a Fusion answer - also carries `output_index`, its place in the reply, and on a hand-back turn that numbering counts a call the pipe does not itself publish: Open WebUI creates the `function_call` item on the chunk, so the pipe reserves the slot Open WebUI will fill rather than writing a competing item into a list the host owns. A thinking box is published in the position the provider gave it, so it precedes a tool call the model emitted after the thought and follows one it emitted before, and the closing record is in the provider's own order. A key that arrives again in a *later* round is that round's own thinking and is published as a second box, not merged into the first: every tool round re-sends the previous round's reasoning item, `id` included, because the pipe normalises only `function_call` and `function_call_output` and mints an id for nothing else, so the reuse is the pipe's own doing and the ordinary shape of a tool turn with a reasoning model. What distinguishes a new block from a continuation is the round, not the key: two fragments under one key in one round are one block, and a key that recurs in a later round is published again with a **fresh** id, because every reducer the pipe feeds — Open WebUI's backend, the browser, and the pipe's own recorded output — replaces by id, so a second box published under the reused id would overwrite the first in the record rather than stand beside it. The *first* publication keeps the provider's id, because that is the id Open WebUI merges the stored record by. The re-publication carries only the text the pipe has not already published: a snapshot that arrived whole leaves the buffer holding both rounds' text, and publishing that whole buffer would show round one's thinking twice. A round that re-sends exactly what it sent before adds no box, and a late fragment inside one round continues its own block rather than splitting it — it **replaces** that block in place, as a `response.output_item.done` carrying the same id, the same `output_index` the `added` carried, and the **whole** buffer rather than the tail, and the closing record carries the whole buffer too. The round boundary also closes each key's timing window, so a re-published block is measured from its own round instead of from wherever the first one opened. The contract has a non-streamed half: with streaming off no deltas are published, so a stretch of answer text written before a later item is announced is published as its own message item first, at the index the record gave it, so the two legs fold the same list. A message segment is announced even though its deltas are not. A reasoning item published from a round that carried several summary blocks carries all of them, in the order they arrived, one `summary_text` entry per block; a block whose identity the provider repeats updates its own entry rather than adding a second one. Without `output_index`, the browser puts an item before the last one it holds, which showed a card above the text written before it. The pipe's closing record reuses the same ids. That matters because Open WebUI's browser merges the record by id into what it already shows: a record under fresh ids showed the answer twice until Open WebUI's final update.

**Streaming off.** The translator does not exist; the events go straight to Open WebUI's emitter, which forwards them to the browser and writes only a subset to the database. For a saved chat that emitter is the socket emitter: `status`, `message`, `replace`, `embeds`, `files` and `source`/`citation` are written, and `chat:message`, `chat:message:delta` and `chat:completion` are **not** — the browser shows them and the database never hears about them. (For a `channel:` chat id it is the channel emitter instead, which writes only the four types listed above and replaces the message on every write, so the streaming setting does not describe what a channel does.) What Open WebUI stores on this leg is the value the pipe RETURNS. For a finished chat reply that is a completion carrying the reply's text and its structured record (`output`), which Open WebUI saves exactly as it saves a streamed reply's record, so tool cards and thinking survive a reload. That includes a reply that ended in an error: its record carries the error card with the rest. A reply that produced nothing but text comes back as that text, error card included. An empty return also fails the outer guard in Open WebUI's `non_streaming_chat_response_handler`, `not continuing and choices and (content or response_output)` at `middleware.py:4271-4273`, and the `chat:completion` frame, the message save, the outlet filters and the title, tags and follow-up tasks all sit behind that one guard, so none of them runs: the turn reloads blank *and* no title is generated, because no assistant message was ever written. A non-streamed 200 from `/responses` that carries no `output` key, or an `output` that is `[]` or `null`, is refused before it can reach that guard: it costs the whole `TRANSIENT_RETRY_MAX_ATTEMPTS` budget and is then reported, so the person reads a card and can send the message again rather than an empty bubble. The connection-failure card covers the missing-key case; the empty and null ones say the model returned an empty answer, because the connection worked.

**An API caller's rejection is not stored at all.** A caller with no truthy `chat_id` **and** `message_id` has no chat to write a card into, so the rejection leaves the pipe as an HTTP error (`400`, upstream status in the body's `code`, and `502` when the pipe could not read an upstream status because the whole body was mangled) instead of as a return value — see [API callers with no chat](error_handling_and_user_experience.md#c-api-callers-with-no-chat-http-error-instead-of-a-card). A **mangled body** — a 200 whose body is not a decodable JSON object — takes the same escape, and it takes it on every leg, including the Fusion internal-divert leg, which runs before the streaming loop rather than inside it. On that path the `chat:message` and `chat:completion` frames are **not** emitted either, only `status`: the escape returns before the emitter that would have sent them runs, because there is nothing to show and nowhere to put it. The turn reaches plugins exactly as a chat's does, recording the same `["failed", "ok"]`: the escape dispatches `dispatch_on_generation_complete(…, "failed")` itself before it returns, and the backstop in `pipe.py` still closes the turn with `"ok"` because it derives its status from the job future, which resolved cleanly.

The rule that covers both legs: **an answer must be both shown and returned.** Showing it satisfies the live view on either leg; returning it satisfies persistence with streaming off and costs nothing with streaming on, where the return value is discarded. A card that is only emitted is a card the user loses on reload; a card that is only returned is a card that appears late.

Notes:
- Answer text streams as `chat:message:delta` frames, and the turn ends with a `chat:completion` frame. The whole-message `chat:message` snapshot is the channel for cards, not for incremental text. On a **streamed** continuing turn the closing frame carries no truthy `content` and no `output`: the browser assigns `content` absolutely, so a frame carrying this generation's text alone would shorten the message below the prefix the turn is continuing. The same withholding applies to the per-round frame that fires after every round that reports usage — a tool round or not — and the two frames now share the same three terms: a turn that has published an item of its own — a thinking box, a tool card, a tool result, or a Fusion answer — carries no bare `content`, because the browser assigns it to `message.content` absolutely, and the visible view on a streamed leg is the `output` array Open WebUI has already folded from the pipe's `response.*` events, which already owns that text. The withholding is position-relative rather than turn-relative: the frame that fires before the turn's first item still carries its text, so a turn that publishes no item of its own keeps the answer text it would have published, and the pipe's own open message item does not count as one. `done` still rides those frames, and the live merged array the person sees on a continuing turn is Open WebUI's own (`middleware.py:5002`/`:5053`), not the pipe's. With streaming off there is no such assignment to fear and no `chat:message` to carry the text, so the closing frame does carry the answer: that frame is the only settle a non-streamed turn gets. A `chat:completion` omits `usage` entirely when nothing was reported, so a reader never has to tell an absent cost from `{}`.
- An answer produced whole rather than streamed — a generated image, a finished video, a help panel, an error card — must go out as `chat:message` or `chat:message:delta`, never as `chat:completion` alone, and must also be the return value. Which of the two depends on what has already been sent: a `chat:message:delta` carries exactly the text appended to the running message and nothing that preceded it, while a `chat:message` carries the running message and the new block together. Neither may carry the new block by itself — a delta that repeats the answer duplicates it on screen, and a snapshot that omits the answer is a non-prefix write, which the translator forwards as a full replacement rather than dropping.
- With streaming off, the wrapper the non-streaming path puts around the emitter suppresses `chat:message` and `chat:message:delta` outright: the answer's text is travelling home in the record, which is what the turn returns whenever any output item was published. Its text is not published as deltas, though the message *item* that carries it still is, whenever a later item would otherwise be announced ahead of it. Anything raised inside the response loop that is only emitted as a card is therefore invisible on that leg; it has to come back as the return value to be seen at all.
- When `SHOW_FINAL_USAGE_STATUS` resolves True — the reader's own copy where they have set it, otherwise the site default an administrator chooses — the pipe formats a final status description using usage/cost/tokens when present, and writes the cost segment only for a charge above zero.

### 4.1 The stored output array, and who owns it

Every finished chat reply hands Open WebUI an `output` array, which it saves against the assistant message: a
streamed reply publishes it in its closing `response.completed` event, and a non-streamed one returns it with its
text. It holds the reasoning items, the tool cards that were shown, and the messages.

A chat turn that is nothing but tool calls stores `[function_call]` and no message item at all, on both the
streaming and the non-streaming leg. A message item carrying no text, no annotations and no reasoning details
is kept in exactly one case: the blank **round divider** that separates one tool round from the next, and it is
a deliberate marker rather than a phantom the model wrote. Nothing displays it - Open WebUI's renderer ignores
blank messages when it draws (`structuredOutput.ts:411` guards on `text.trim()`) - and nothing is sent to the
provider for it either; what it does is hold the round boundary in the stored turn. Open WebUI's own
`convert_output_to_messages` (`backend/open_webui/utils/misc.py:325`) is what reads the rounds back out of this
array, and it reads them from there: with the divider stored, a two-round turn converts to
`assistant(tool_calls=[a]) | tool | assistant(tool_calls=[b]) | tool`; without it, both calls batch into a
single assistant message and the turn comes back as `assistant(tool_calls=[a, b]) | tool | tool`, which is
not the turn that ran. The mechanism is `misc.py:473-476`, which flushes pending tool outputs on any item that
is not itself a call or a call output, and `misc.py:483`'s `if text:`, which then drops the divider's blank
text. So the item is a boundary marker Open WebUI drops from the text it sends and depends on for the shape of
the text it does send, and a turn with only one tool round has no boundary to record, so it stores none. A
message item that *does* carry something is kept even when its text is empty, because that is where the turn's
reasoning and its citations live: a turn whose only content is OpenRouter's structured `reasoning_details`
keeps its message item for exactly that reason. A turn that carries only plain `reasoning` or
`reasoning_content` - the shape DeepSeek-, Qwen- and Kimi-style OpenRouter models emit rather than the
structured form - keeps no message item, and that reasoning is still published and stored, as its own reasoning
item rather than as message text. The terminal `Chats` write reads `annotations` and `reasoning_details` off
every round of the turn, in round order, not off the round the loop ended on.

Its messages carry the ids of
the items their text streamed into, so Open WebUI's browser replaces the items it already shows rather than
adding a second copy. Everything later turns replay comes from there, so who writes it matters.

The three message fields beside it — `sources`, `annotations`, `reasoning_details` — are written separately, at the
end of the turn, in **one** call. Open WebUI's upsert is a whole-document rewrite: it takes the chat row
(`SELECT ... FOR UPDATE` on Postgres, `chats.py:1152-1179`), mutates `chat['history']` and commits, so three calls
to write one message meant three row locks and three rewrites. Open WebUI's own middleware batches its equivalent
save into a single call (`middleware.py:6555-6563` streaming, `:4366-4375` non-streaming), and so does this.

The batch is strictly cheaper on a successful turn and strictly worse on a failed one: one refused write now loses
all three fields where it previously lost one. That is the trade. What it buys back is the read, which the union
needs — a read-modify-write across three calls is three reads. The read is the chat blob rather than the
per-message fast path, because `chat_message` has no column for `annotations` or `reasoning_details`
(`chat_messages.py:127-175`, `:380-393`) and a read through it would merge the citations and silently drop the other
two. A turn with nothing to write does neither: the read sits behind the empty-payload check. The payload carries
the non-empty fields and nothing else — no `content`, no `output`, because `Chat.svelte:3216-3218` assigns
`message.content` absolutely from any frame carrying text and no output, so naming either would overwrite the
answer being read. There is no `touch=`: Open WebUI's socket handler passes it because it fires per citation, and
this is the one write that should refresh `updated_at`.

Open WebUI's bookkeeping depends on what it is doing, and the pipe has to match it:

- **Continuing a message.** Open WebUI sets the message's existing stored output aside at stream start and puts
  it back in front afterwards. It already holds those items, so the pipe must publish only this generation's:
  republishing would store the earlier generation twice - and with it every hidden marker it carries, so the
  replayed reasoning would double as well. The pipe recognises this by the message id the frontend sends only
  when continuing. The same turn also withholds the bare `content` from its `chat:completion` frames, for the
  separate reason above: the stored array is Open WebUI's to fold, and the live message is the one the absolute
  assignment would shorten. The merged array a continuing person watches being built is Open WebUI's own
  (`middleware.py:5002`, `:5053`), not the pipe's. The three message *fields* the pipe writes at the end of a turn
  — `sources`, `annotations`, `reasoning_details` — behave the other way round: the pipe does write them on a
  Continue, and additively, so the message ends the second generation holding both generations' entries in order.
  See [History Reconstruction & Context Replay](history_reconstruction_and_context.md), section 6.2.
- **Its own tool loop.** Each time Open WebUI runs a round and calls the pipe back, it sets the turn's output
  aside the same way, but sends no such id. The pipe therefore reads the stored output on that path - and finds
  nothing, because Open WebUI writes a message's output once, when the message finishes. Measured on a live
  server: across a two-round turn the row carried no output at all until the moment it was marked done.

So the read exists for one caller only: a client posting to the completions endpoint directly with the id of a
message that already holds output. Open WebUI's own chat never produces that state, and for the direct caller
republishing is the right answer, because nothing else will put those items back.

One consequence is worth stating plainly, because it is a deliberate choice rather than an oversight: a tool call
left genuinely unfinished — one whose function never returned, whether the turn ended on an error or a dropped
connection — on a message that is then continued, stays unfinished in Open WebUI's saved copy instead of
being repaired, so Open WebUI does not replay it. A tool call left running only by Stop is not in that class: a Stop
now runs the finalisation to its end, so the turn still publishes its terminal record. The events that would repair
an unfinished call cannot reach storage -
`response.output_item.added` matches ids only within the new output, and `response.output_item.done` replaces by
position - so an attempt to heal it would duplicate or corrupt the saved copy instead. The model still learns the
round's calls before the first one still running, in the order the model emitted them: Open WebUI saves their cards as finished and hands them back, and
with cards off the pipe's own copy does (not in a temporary chat, where the pipe keeps no copy), since the pipe writes
a round's calls when the round starts and each result when its call returns, that is as the calls are answered, so a
refused call's answer is written ahead of the results that were still running (see
[History Reconstruction & Context Replay](history_reconstruction_and_context.md), section 5.4).

---

## 5. Streaming errors and how they surface

OpenRouter sends `200 OK` as soon as a provider accepts a request, so a failure after that point arrives inside the answer. On both `/responses` and `/chat/completions`, the pipe detects such error payloads (for example `response.failed`, or a chunk carrying an `error` block) and converts them into an `OpenRouterAPIError`. The guard runs for every event it takes off the queue, whichever of the two consumer loops fetched it, so a failure is surfaced and counted the same way in both. A non-streaming reply whose body carries an `error`, at the top level or in its first choice, is treated the same way.

That error is then handled by the same OpenRouter template system described in:
- [Error Handling & User Experience](error_handling_and_user_experience.md)

This keeps user-facing failures consistent between streaming and non-streaming calls. A retryable in-band failure — one carrying a `429` or a `5xx` — is retried before anything has streamed, on the same terms as a `429` or `5xx` status at the HTTP boundary, honouring `Retry-After` up to the `TRANSIENT_RETRY_MAX_WAIT_SECONDS` cap; once content has been shown it is rendered immediately, as below. Each failed request counts once toward the user's request breaker (see [Concurrency Controls & Resilience](concurrency_controls_and_resilience.md)), however many attempts it took; errors in housekeeping tasks such as title generation are not counted.

An attempt that has already published something Open WebUI keeps is not retried: the reader keeps the thinking box or the tool card they were watching and the error card is appended below it. Only an attempt that has published nothing is handed back for a retry. What counts as published is what has been handed to the consumer, not what merely arrived: the `delta.role` frame every `/chat/completions` stream opens with, and the `response.created` event every `/responses` stream opens with, carry nothing the reader can see, so they leave the attempt still retryable. The third is a `delta.tool_calls` entry that carries neither a function name nor any argument text: it has opened a call nobody can see yet, so it leaves the attempt retryable too, while an entry that names a tool or writes any of its arguments closes the window on its own. This is why the in-band check on `/responses` lives in the **producer**: the consumer that would have to know whether anything was published runs outside the retryer, so moving the check back to the consumer silently disables the retry for every in-band failure.

A stream can also stop before its final event, without an error. If nothing the reader can see has arrived, the attempt counts as a failed call and is retried, up to `TRANSIENT_RETRY_MAX_ATTEMPTS` extra tries; if every attempt comes back empty, the request ends with `CONNECTION_ERROR_TEMPLATE`, or with the `STREAM_INTERRUPTED_TEMPLATE` notice if answer text arrived earlier in the same reply or the model had already named the tool it was calling. If anything the reader can see has arrived, whatever text has streamed (possibly none) is kept, the `STREAM_INTERRUPTED_TEMPLATE` notice is appended, and the call counts as a failed one. A frame that arrived but delivered nothing does not close the retry window, so a drop in the first second of a reply is retried rather than costing the whole reply. The two readers do not draw the completion line in the same place: on `/chat/completions` a flushed end-of-stream marker is enough to end the turn, while on `/responses` a terminal event is required, so a `/responses` stream that ends with the marker but no `response.completed` still counts as a failed call. A terminal event that arrives carrying no `response` object -- the key absent, an empty object, or something that is not an object -- counts the same way, because there is nothing in it to read as the run's record.

A connection that drops or times out is handled identically on both transports, and the request counts once against the breaker, however many attempts it took. If this happens before any answer text has arrived, a streaming attempt that has sent nothing the pipe could read is retried; if the request still fails, it ends with `NETWORK_TIMEOUT_TEMPLATE` for a timeout or `CONNECTION_ERROR_TEMPLATE` for a failed connection. If it happens after answer text has arrived, that text is kept and the `STREAM_INTERRUPTED_TEMPLATE` notice follows it. A stream that ends with `response.incomplete` is a finished answer, not an interruption: its usage is reported as usual, and a warning says the answer ended incomplete.
A retryable rejection hands the turn back to the orchestrator, which re-runs it on the same turn and the same emitter, and Open WebUI never discards what it has already accumulated. So a turn that has published an **accumulated** frame - `chat:message:delta` or `chat:tool_calls` - ends in a card rather than being retried: the reader never sees two attempts of one turn spliced together. The barrier stops the *second* publish, not the first, so the abandoned tool call stays in Open WebUI's list at `status: in_progress` with half its arguments and the turn ends on a card; that is no duplication, not no residue. `status` and `source` frames are additive rather than accumulated, and both are persisted, so a retried turn can carry two attempts' notices in its status history. That is also why each attempt opens and closes its own status: Open WebUI appends to the history and never marks a prior entry done, so an attempt that hands the turn back without closing the line it opened would leave a shimmering notice with no successor to close it. For `chat:tool_calls`, the gate is the streaming leg, because there the pipe's own translator rewrites it into the OpenAI chunk Open WebUI folds, and on that leg Open WebUI creates the `function_call` item itself, which is why the pipe reserves the slot rather than publishing it; with streaming off the wrapper suppresses only `chat:message` and `chat:message:delta`, and `chat:tool_calls` leaves the pipe and is discarded by Open WebUI's own socket emitter (backend `socket/main.py:1178-1265`, frontend `Chat.svelte:1482`) - the string occurs nowhere in Open WebUI - so a hand-back on that leg costs nothing. `chat:message:delta` is gated the same way: with streaming off the wrapper drops it before it leaves the pipe, so it ends the turn on the streaming leg only, and a hand-back on the non-streaming leg costs a retry that would have succeeded. The rewrite itself is not covered by the test suite: were `chat:tool_calls` ever given the reshape-or-drop treatment `chat:message:delta` already has, the same splice could return with the tests still green. See [Plugin System](plugin_system.md) for the three rejections that are retried.

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
- **Routing boundary** — a text delta whose `item_id`, `output_index` or `content_index` differs from the buffer's: force flush, and the batch carries the keys of the delta that opened it. The reasoning arm guards on `item_id` alone, because a reasoning block legitimately moves between content parts and summary parts inside one item, and a wider key set there would fragment every multi-part reasoning block. `summary_index` is not a text key: it belongs to `reasoning_summary_text.delta`, which is not batched.
- **Non-batchable event** — flush both buffers, pass event through
- **Idle timeout** — `STREAMING_IDLE_FLUSH_MS` flushes buffers when producer pauses
- **End of micro-drain cycle** — flush if buffered chars >= `STREAMING_NAGLE_MIN_FLUSH_CHARS` (drain is bounded to 32 events and only runs while no output has been produced yet)
- **End of stream** — unconditional final flush
- **Error mid-stream** — the buffered tail is force-flushed before the streaming error is re-raised

### Coverage: both streaming paths

- **Responses API path** (`responses_adapter.py`): Uses `NagleCoalescer` directly in the existing consumer loop (already has `event_queue` with workers).
- **Chat Completions API path** (`chat_completions_adapter.py`): Wrapped with `nagle_coalesce_stream()` which creates a lightweight pump task + queue + bounded micro-drain.

On both paths the tail the coalescer is holding is delivered to the consumer when a
streaming error arrives, on the Responses path by the consumer loop's own force flush
before the re-raise and on the Chat Completions path by the final flush in
`nagle_coalesce_stream()`.

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
- Bound memory or latency: keep `STREAMING_CHUNK_QUEUE_MAXSIZE=0` and `STREAMING_EVENT_QUEUE_MAXSIZE=0` unless you have a measured reason to bound them. As long as the consumer keeps draining, a bound trades a larger buffer for a slower drain, not for a stream that stops early. This is about the two queues and nothing else: the producer's pre-first-output buffer is a third buffer that these two settings do not reach, and it is uncapped (§3).
- Improve observability: set `STREAMING_EVENT_QUEUE_WARN_SIZE` low enough to signal stress early, but high enough to avoid constant warnings.
- Reduce UI freeze during heavy reasoning: increase `STREAMING_NAGLE_MIN_FLUSH_CHARS` (2-5 for moderate reduction, 5-10 for aggressive).
- Disable all coalescing (debugging): set `STREAMING_DELTA_CHAR_LIMIT=0` and `STREAMING_IDLE_FLUSH_MS=0` for passthrough mode.

All related defaults and ranges are listed in [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
