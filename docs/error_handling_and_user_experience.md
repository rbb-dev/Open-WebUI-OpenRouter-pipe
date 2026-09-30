# Error Handling & User Experience

This document describes how the pipe renders user-facing error messages, how operators correlate errors with backend logs, and how to customize error templates safely via valves.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [OpenRouter integration](openrouter_integrations_and_telemetry.md) · [Security](security_and_encryption.md)

---

## Overview

The pipe aims to avoid raw exception traces surfacing in the Open WebUI UI. Instead, it:

- Logs technical details server-side with a unique `error_id` for correlation.
- Emits user-facing Markdown messages (templates) with actionable remediation guidance.
- Includes contextual identifiers (session/user) when available.

There are two main error rendering paths, and a third that does not render at all:

1. **OpenRouter “request rejected” templates** (OpenRouter HTTP status handling and parsed provider errors).
2. **Generic templated errors** (timeouts/connectivity/internal failures handled by `_emit_templated_error`).
3. **API callers with no chat**, where there is nowhere to write a card: a provider rejection is returned as an HTTP error instead. The same leg carries the five pre-job refusals, which are further conditions on this path: a tripped per-user circuit breaker, warmup failure, a missing request queue, a full queue and a pre-enqueue setup failure. Each is raised locally and has no upstream status behind it, so it leaves as its own status rather than a 400 — **429** for the breaker, which is a per-entity rate limit, and **503** for the other four, which are conditions of this process rather than of one caller. Gated on a truthy `chat_id` **and** `message_id` — Open WebUI's own idiom for "is there a chat to write this into" (`main.py:1703`, `utils/middleware.py:3286`) — with `stream: false`, and never on an Anthropic Messages path (`main.py:2054-2063` re-wraps the response for the Anthropic converter, which has no error branch). The gate is a negative test on the request path, not a check for `/api/chat/completions`, so it also fires on Open WebUI's task routes. See [API callers with no chat](#c-api-callers-with-no-chat-http-error-instead-of-a-card).

### What is not an error, but is still said out loud

**A provider refusal is not on this list, and is not an error either.** A model that declines — a `refusal` sibling of `content` on `/chat/completions`, a `{"type": "refusal", "refusal": <str>}` part in a `/responses` output message, or a `response.refusal.delta` / `.done` pair on a streamed `/responses` call — wrote that sentence as its answer, so the pipe forwards it as answer text on every endpoint and every leg, joined to any answer with a blank line and delivered as the whole reply when there is none. No card is rendered and no fault is raised, because a card for the sentence the model actually wrote would be false to the reader. This is a different thing from the provider *moderation* refusal in section A, which is a rejection of the request and does render `OPENROUTER_ERROR_TEMPLATE`: there the provider refused to serve the request at all, and here it served it and declined the task. The one consumer that treats a refusal as a failure is a housekeeping `task_request`, where Open WebUI takes the return value as a persisted title or a stored summary rather than as an answer a person reads; there it fails the attempt with `task_model_refusal` on either endpoint and the refusal text is never carried in the fault.

A request can succeed and still be degraded: the replay budget replaces a tool result with a model-visible stub, and
the model then answers without ever having seen that result. Nothing raised, nothing logged as an error, and nothing
rendered as a card — but the answer the person reads was built without input they expected it to have, so the pipe
emits a warning notification naming the tools whose results were withheld, once per call id, on both the streaming and
the non-streaming path, and on the orchestrator leg and the tool-round leg alike. Two neighbouring conditions are kept
apart: a budget that could not trim far enough (`futile`) has its own notice and its own wording, because it is the
opposite outcome — trimming that succeeded versus trimming that failed — and neither text is folded into the other. The
task-model adapter path stays silent by design, because the classifier it runs never sees the tool round. See
[Tooling & Integrations](tooling_and_integrations.md).

The same shape has a second source, one layer further down. When the artifact store fails to load an earlier tool round,
the request also goes out without it: the store swallows its own **database** error, records the failure against the DB
breaker and returns whatever its cache already held, so the turn succeeds while degraded. A failure to refill that cache
afterwards is not a database failure and is not counted against the breaker: the read has already been answered, nothing
is lost, and the refill's own failure is logged instead. It emits a warning notification
of its own, through the same seam the breaker's notice uses, saying that earlier tool results could not be loaded and
that the model did not receive them. Only a failure of the *read* does that: a fault in the replay-cache refill that
follows a successful read returns the rows anyway, clears the breaker window, and is logged as a cache fault instead —
nothing was lost, so there is nothing to announce. It names no cause — the store cannot tell a database blip from a decryption failure
— and it promises no retry. It is kept apart from the two budget notices on purpose: a budget that trimmed successfully
and a read that failed are different conditions, and one turn can hit both, so folding the wordings together would have
each of them announce the other's condition. A marker that simply resolves to nothing stays log-only, because a
legitimately consumed row is indistinguishable from a lost one at that layer; see
[History Reconstruction & Context](history_reconstruction_and_context.md).

A read that fails outright is not the only short read. A row the current `ARTIFACT_ENCRYPTION_KEY` cannot
open is skipped while the rest of the batch is returned, so the turn replays a hole in the stored round and
the model is handed a call with no result. That gets its own notice, worded for what the store actually
knows: how many rows were unreadable and which artifact kinds they were, with no id and no content, because
an id is a handle on a turn. It is measured from the set of rows the cipher rejected rather than from a
`requested − returned` diff — a row deleted outright is missing for a reason that is not a fault, and
counting it would report every retention sweep as a loss. The round is not repaired: the transformer drops
a half-round rather than handing the model a fabricated "could not be read" result.

The write side of the same table has its own notice, in its own words. When
`ARTIFACT_ENCRYPTION_KEY` cannot be read under the current `WEBUI_SECRET_KEY` the store
refuses the write rather than store the item in the clear, and the turn that offered the
row carries a warning naming the key and the secret, saying the items will be missing from
later turns and that re-entering the key resumes storing them. It is the one write notice
that names a cause, and it has to: the store knows exactly why, and the repair is a valve
the person can edit. It is kept apart from both the database-failure notice and the read
notice above, because all three cost the same thing — a round the model does not get — and
only one of them is fixed by re-entering a key. It fires once per blocked episode rather
than once per turn, and re-arms when the key becomes readable again.

### Where a card lands: saved chat or channel

A card is delivered differently depending on whether the chat id is a `channel:` id. The
chat id is carried to the error arms on a context variable the pipe sets in two places
(before the job is enqueued, and inside the job itself), so the choice of path is a
property of the conversation, not of the frame:

- **Saved chat.** The card travels as a `chat:message` snapshot and the turn's closing
  `chat:completion` carries no content. The stored message is the answer plus the card, in
  that order. On a **continuing** turn it is the other way round: the card travels as
  `chat:message:error`, the closing frame carries no content on any leg, and the bubble
  keeps the prefix the turn was continuing. `chat:message` assigns `message.content`
  absolutely on the browser, so publishing the snapshot there would replace the whole
  stored reply with the card; the error frame goes to the error area under the message and
  leaves the content alone. The stored record is unaffected either way — it is the answer
  plus the card, in that order.
- **Channel.** The channel's emitter is a different function from the socket emitter, and it
  honours a different set of types. It has no `chat:message` branch at all, so on a channel
  the card is written by a `chat:message:error` frame — which Open WebUI prefixes with
  `Error: ` — and the closing `chat:completion` carries the same text as its `content`, so
  the prefixed write is replaced by the clean text and the stored message reads as the
  answer followed by the card. The channel leg **replaces** the message rather than
  appending to it, so the closing frame's content must be the whole message or the card is
  lost.

The five pre-job refusals (a tripped circuit breaker, warmup, a missing stream queue, a
full queue, and a pre-enqueue failure) never build a stream queue and so never
install the middleware emitter: they emit straight to the channel emitter and take the
same channel path as any other card. A caller with no chat to write that card into gets
the refusal as a status instead, carrying the same sentence as the error message: 429 for
the breaker, 503 for the other four.

**A channel card is reduced, a saved-chat card is not.** Every member of a channel can
read what the pipe writes there, so a card that reaches a channel is rendered as though
twelve of its placeholders were empty: `session_id`, `user_id`, `detail`,
`sanitized_detail`, `reason`, `openrouter_message`, `upstream_message`,
`moderation_reasons`, `flagged_excerpt`, `raw_body`, `metadata_json` and
`provider_raw_json`. Those are the requester's identifiers, the text they wrote and the
provider's own prose about it — and a provider's rejection body routinely quotes the
prompt back, so the prose *can* be their words. The reduction happens at the render, on
every path that can reach a channel, and it is keyed on the surface rather than on the
template: an admin's custom template gets the same answer, and there is no valve to put
the ids back. `error_id`, the model, the provider, `openrouter_code`, `status_code`, the
limits and the rate or balance fields still render, so the card still says what failed.

A template author therefore does have to write one thing differently for a channel: a
line whose placeholder came out empty is omitted, so a line that exists only to show the
session id or the flagged excerpt simply is not there in a room. A card that ends early,
or that relies on the `Error: ` prefix being visible, also reads differently in a channel
than in a saved chat. Nothing else about the routing changes.

---

## Error IDs and enriched context

For templated errors, the pipe generates:
- `error_id`: 16 hex characters (`secrets.token_hex(8)`)
- `timestamp`: ISO 8601 UTC timestamp
- `session_id` and `user_id` (when available) — withheld from a card that reaches a channel, because every
  member of the room would otherwise read them
- `support_email` and `support_url` (from valves `SUPPORT_EMAIL` and `SUPPORT_URL`)

Operator logs include the `error_id`, and templates can include it in user-facing text for support correlation. On a
channel it is also the only correlation handle the reader has, which is why it is never withheld.

---

## Exception/status mapping (which template is used)

### A) OpenRouter “request rejected” errors (status-aware templates)

The pipe selects the template for the status the failure resolves to, which is the typed kind OpenRouter named, then the body's own `error.code`, and the HTTP line only where the body yielded neither:

| Status | Template valve |
| --- | --- |
| `401` | `AUTHENTICATION_ERROR_TEMPLATE` |
| `402` | `INSUFFICIENT_CREDITS_TEMPLATE` |
| `408` | `SERVER_TIMEOUT_TEMPLATE` |
| `413` | `PAYLOAD_TOO_LARGE_TEMPLATE` |
| `429` | `RATE_LIMIT_TEMPLATE` |
| `>= 500` | `SERVICE_ERROR_TEMPLATE` |
| other / default | `OPENROUTER_ERROR_TEMPLATE` |

“Other / default” is every remaining case, including a missing status: `403`, `404`, `422`, a `400` for a malformed reference URL, and a provider moderation refusal all render `OPENROUTER_ERROR_TEMPLATE`. Anything written into that template is therefore read by failures that have nothing in common beyond not having a template of their own, so advice specific to one cause belongs in a `{{#if}}` block rather than in the template's unconditional text.

**A `429` or `5xx` is retried before its card renders, and so is a `408` that names the provider-timeout kind.** Such a failure is one of the temporary class, so the pipe re-sends the request up to `TRANSIENT_RETRY_MAX_ATTEMPTS` extra times before showing anything; a `Retry-After` the server sends sets the wait, truncated to `TRANSIENT_RETRY_MAX_WAIT_SECONDS` rather than replaced by it. The card therefore means the retries are spent, and wording in it can say what an operator would expect of a final failure. A `401` — and every other status in the table apart from that `408` — is never retried, and nothing at all is retried once text has been shown to the reader. A body carrying a content decision is the one exception to that set: it is not retried, whatever status it arrived on, because re-sending a prompt a guardrail declined cannot unblock it.

**A non-streamed 200 on `/responses` whose `output` is neither a list nor null is treated like a dropped answer.** The request is re-sent up to `TRANSIENT_RETRY_MAX_ATTEMPTS` extra times, then reported and counted once, because the reader downstream can consume nothing else: an object, a string or a number in that key is a finished turn with no item in it, which reads as a silent success and clears the reader's breaker record. An `output` that is an empty list or null is a different case and is described below: the connection worked and the model returned nothing.

**A `403` is not one thing, and a content decision does not pause the user's background work.** OpenRouter uses the status for at least four different answers: a key it rejected, a key that is valid but lacks permission for the model, a provider content policy, and a guardrail that declined the prompt before it reached any provider. The first is the person's to fix by signing in again; the other three are not, so only the first arms the sixty-second credential pause that otherwise leaves the next chats titled "Chat" with no error shown. A guardrail block is the case that carries no type at all — its only marker is the `patterns` the guardrail matched — and it is the case a status-only reading blames on the key. Which field named the block does not change the answer: the carve-out is position-independent, reading the typed kind or the native code, whichever field carried it, and both names are derived from the status tables rather than written out beside them, so a new `403` row added to either table is recognised as a decline without anyone editing a second list — and a property test over both tables is what keeps that derivation honest. Rendering follows the status, so the four answers above do not all render the same card: a body carrying a decline marker is read as a content decision whatever status it arrived on — it keeps that status, and on the `403` a guardrail normally arrives on it renders `OPENROUTER_ERROR_TEMPLATE`, while a `403` naming no marker and a kind documented at another status — a `rate_limit_exceeded`, say — resolves to that status and renders its card. Either way OpenRouter's own reason is shown.

These templates are used for the `OpenRouterAPIError` path (and for certain HTTP status errors that are converted into an OpenRouter error object by reading the response body best-effort).

**A rejected request and a failure reported inside a reply that has already started are read the same way.** OpenRouter commits `200 OK` as soon as a provider accepts a request, so a rate limit, an overloaded or unreachable provider, a provider timeout, an exhausted balance or a sign-in failure occurring after that point arrives inside the reply instead of as a status. The pipe reads the kind of failure OpenRouter names there - `error.metadata.error_type` on Chat Completions, the top-level `error_type` on Responses, `error.error_type` on the Anthropic Messages envelope a gateway in front of the pipe puts on the wire, the same fields the model-limits section below uses - and renders the template for the status that kind is documented with, so the person reads the message written for that failure rather than the rejected-request one. A kind the pipe does not recognise falls back to the status reported alongside it, and then to `OPENROUTER_ERROR_TEMPLATE`. On a rejection the pipe reads the same kind from the same three fields, in the same order, and renders the template for the status that kind is documented with, so a `5xx` carrying `authentication` on the Responses skin is read as the rejected credential it is rather than as a server-side error, and the same body classifies identically whichever of the two paths delivered it. The HTTP status is the floor in both directions: on a rejection, where the wire names no kind the pipe knows, the line stands whatever numeric `error.code` the body carries beside it — a `429` or `5xx` stays retriable, and a `401`, `403` or `400` is not re-read as a rate limit or a server fault — because the native code on that skin is collapsed to a `5xx`, the code is documented as best-effort, and a line the server chose must not be moved by a number in the body. Inside a reply there is no wire to floor, so the code reported there still decides. A temporary one — `429` or `5xx` — reported before any content is streamed is retried first, and renders exactly the same card once the retries are spent; the same kind reported after content is shown is rendered immediately, as described above. One consequence matters when editing `SERVER_TIMEOUT_TEMPLATE`: a provider timeout reported this way is documented as `504`, so it renders `SERVICE_ERROR_TEMPLATE`. The timeout template is reached by a `408`, whether that is the reply's own status or the code reported inside the reply.

**Every caller uses the table.** Status selection is the only thing that chooses one of these templates; no call site can pass a template that overrides it. The chat orchestrator, the outer request handler, image generation, video generation, and a streaming failure arriving after output has begun all render the same template for the same status.

**One exception, and it matters for the wording of `AUTHENTICATION_ERROR_TEMPLATE`.** When the pipe cannot read its own OpenRouter API key it renders that template directly. Nothing was sent, so there is no status and no rejection by OpenRouter — the cause is local: the key setting is blank, or the value stored in it was encrypted under a `WEBUI_SECRET_KEY` that has since changed and can no longer be decrypted. The `401` shown on the card in that case is a display value the pipe supplies, not a status any server returned. Wording written for that box therefore has to fit a local configuration fault as well as a key OpenRouter rejected, which is why the built-in text says the request was not authorised "or the pipe could not read one" rather than asserting that OpenRouter refused the credentials. It is the only template in the table above reached this way; every other template is chosen by status alone, from the table in section B below.

**OpenRouter's own request reference travels with the card.** When a rejection carries one, every template in the table above renders it on its own row, separate from the `error_id` the pipe generates: the pipe's id correlates the pipe's logs, and OpenRouter's is what its support can look up. A rejection that carries none renders no such row. The reference is unavailable on the paths where no request reached OpenRouter — a key the pipe could not read, and a `5xx` raised by the connection itself — and on a failure event that carries no id (a Responses `response.error` or plain `error` event); on Chat Completions OpenRouter's generation id is the reference, also available as `error_chunk_id`. A template edited to show the reference should therefore keep the row inside a conditional, as the built-in text does.

**Clearing a template box and saving restores its built-in text.** Every error template valve behaves this way: leave the box empty (or containing only spaces or newlines), save, and the pipe writes that valve's factory default back, so reopening the Config tab shows the original wording ready to edit again. Only the valve that was cleared is affected. This is how an operator recovers from an edit that went wrong, since the built-in text is not otherwise visible in the interface. The same restore applies when valves are edited through Open WebUI's own Functions valve panel. That rule is scoped to the error templates: clearing a *nullable numeric* setting's box does not restore text, it returns the setting to unset, so a saved row carries no name for it at all.

**A stored value the current release no longer accepts degrades one setting at a time, and says so.** This is the other way a configuration is not what the operator thinks it is. The trigger is a release, not an edit: a bound tightened, a `Literal` member removed or a type narrowed, against a row an installation already wrote. It used to be a total outage instead — the pipe refused to load at all, and Open WebUI re-raised that on every chat, so every user on every worker got a failure with no message naming a setting. It now leaves that one setting at its default, writes a warning naming the field, the stored value and the default it read as, and leaves every other saved value untouched. The warning is the whole signal on an installation with no Config tab, so it is WARNING on first sighting and after a five-minute cooldown, and DEBUG in between rather than a flood on a path that runs per request. The other half of that signal is the setting this release no longer publishes at all — a name a release renamed or removed, which the stored row still carries and which is read by nobody. That name is reported in the same warning, and its stored value is never quoted with it: the name is all the operator needs to delete it from the row, and a name this release does not publish has no annotation left to decide that its value was a secret. Secret valves are the exception in both directions: they are never replaced by their default, and neither their stored value nor their default is ever written to the log. The row repairs itself on the next save from Open WebUI's own valves panel; the Open WebUI *sync* route is the one door that does not repair it, so an operator who repairs a stale row must do it from that panel.

**A filter write the database refused is surfaced as a database fault, not a setting.** Open WebUI's `Functions.update_function_by_id` returns `None` — it never raises — both when the commit failed and when the row is gone, so a write that did not land is invisible to any caller that only wraps the call in `try`/`except`. Every switch-off, switch-on and retirement write the pipe makes goes through one helper that reads that return value, and when the answer is `None` it records the row, names the operation it was performing, and says the write will be retried on the next pass. The message is WARNING the first time an hour and DEBUG after that, per row, so a standing fault is one line rather than one per pass. **What to look for:** a `Open WebUI refused the write to <row id> while <operation>` line. `<operation>` is the only clue about *which* write was refused — the refusal set is matched by row-id prefix, so a retirement and an install of the same family read alike at that line, and the operation is what tells them apart. **What it means:** a database fault, not a valve to change; no setting reaches it. **What to do:** check the Open WebUI database's health and disk space, then let the next pass retry, or restart the worker to force one sooner.

### B) Generic templated errors (network/5xx/internal)

When a chat reply's call to OpenRouter fails without being rejected, or anything else goes wrong in the chat reply loop, the pipe picks one of these templates:

| Condition | Template valve |
| --- | --- |
| A timeout before any answer text has arrived | `NETWORK_TIMEOUT_TEMPLATE` |
| A connection that cannot be opened or drops, a stream that sent nothing the pipe could read on every attempt, or one whose readable frames published nothing a reader could use, an accepted status whose body was lost before it finished arriving, or a non-streamed 200 carrying no `output` key on `/responses`, or an `output` that is neither a list nor null, and no `choices` on `/chat/completions`, before any answer text has arrived and before the model has named the tool it is calling | `CONNECTION_ERROR_TEMPLATE` |
| An accepted status whose body is not a JSON object (a proxy, WAF or gateway rewrote the reply) | `SERVICE_ERROR_TEMPLATE` |
| A timeout, a failed or dropped connection, or a stream that published nothing a reader could use, after answer text arrived earlier in the reply or after the model named the tool it is calling | `STREAM_INTERRUPTED_TEMPLATE`, appended after the kept text |
| Any other exception | `INTERNAL_ERROR_TEMPLATE` |

The timeout template's `timeout_seconds` is the limit that ran out: `HTTP_CONNECT_TIMEOUT_SECONDS` while connecting, `HTTP_SOCK_READ_SECONDS` while waiting for data, or `HTTP_TOTAL_TIMEOUT_SECONDS` for the whole request. The connection template's `error_type` names the failure (for example `ClientConnectorError`). A connection that closes before the first byte is retried, up to `TRANSIENT_RETRY_MAX_ATTEMPTS` extra tries (three attempts in all by default), and the request is counted once against the breaker however many attempts it took; a retry that arrives after answer text has been shown is not made, and the table above's `STREAM_INTERRUPTED_TEMPLATE` row then applies. A stream that closes early without an error, once any of its events has arrived, also gets `STREAM_INTERRUPTED_TEMPLATE`; see [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md). Picture-only image models, video models and the panel, judge and final-answer calls inside internal Fusion report failures in their own way — a Fusion panel member that never got out is reported with a fixed, pipe-authored reason rather than a copy of its response, because a member's response can carry a storage URL the user was never shown, and a final-answer call that dies mid-stream keeps the text it had streamed and appends a short degrade marker naming that failure, which is stored with the reply rather than shown as a card — and a required internal file that cannot be read shows its own message. Their breaker accounting, unlike their reporting, does follow the text legs' arrival rule: a picture-only image or video generation whose response arrived whole and was not an OpenRouter document — a body a proxy, CDN or WAF rewrote, or one that is not a JSON object at all — never counts toward the breaker's limit, because the network path produced that body rather than OpenRouter; every other post-submit failure on those legs is charged exactly as it is here.

**A non-streamed 200 that holds no output items is a failure the person can retry, not an empty turn, and it is not a connection fault.** With streaming off, OpenRouter can answer a `/responses` request with `200` and an `output` that is `[]` or `null`: the call connected, the model ran, and it produced nothing. Both shapes take the same path as a body with no `output` key at all — the full `TRANSIENT_RETRY_MAX_ATTEMPTS` budget, one strike on the breaker — and the retry can still bring a real answer. The card is then `OPENROUTER_ERROR_TEMPLATE`, because the connection-failure card would tell the person to check their firewall and their DNS about a reply that arrived; its reason reads that the model returned an empty answer (no output items) on `/responses`, and the person can simply send the message again. A body carrying at least one output item is unchanged, and a body with no `output` key and a body with no `choices` key are still the connection card above.

**A mangled whole body is not retried and does not count against the breaker.** "Whole" is what this says: the body arrived, and what came back is not an OpenRouter document — the `UpstreamBodyUnreadable` case, which the provider produced and a proxy, CDN or WAF may have rewritten. The provider answered, with a reply code the pipe accepted, so a re-POST asks it the same question and a rewritten reply is rewritten again; and the fault is in the network path rather than in OpenRouter, so counting it would punish a user for somebody else's firewall. The reason a caller is handed names the endpoint that answered and carries the upstream `Content-Type`. Those two are the load-bearing half of the diagnosis, and the chat card additionally quotes the first 200 characters of what the provider sent, which is where the proxy's own error page becomes visible. The API error envelope quotes nothing: a chatless caller has no chat to write a diagnosis into, so its JSON body is composed by the pipe out of the two values the pipe itself derived, rather than transported from a body that never parsed into an `error` object at all (see [Security & Encryption](security_and_encryption.md#log-safety)). One sentence is therefore rendered on three deliveries — the chat card, a task route's card, and the Anthropic Messages card, each as its `reason` — while a chatless, non-streamed caller's envelope carries the same sentence without the excerpt, as its `message`.

**A body lost before it finished arriving is a different class, and it does count.** Here the connection was accepted and the body died on the way — a dropped connection behind a `200`, which reaches the pipe as a payload fault rather than as a document. OpenRouter has already run the call, so the request is **not** re-POSTed; but nothing was delivered, so it is one failed call and is counted once against the breaker, and it renders `CONNECTION_ERROR_TEMPLATE` like any other pre-answer connection fault. The two halves are deliberate: a body the provider produced and something else rewrote is nobody's bill but the reader's outage, while a body the transport lost is a real failure of a real call.

### What retrying does not change

On the Agent tool's own API — `/api/v1/messages` — a card that survives its retries still reaches the agent as a `text_delta` followed by `stop_reason: "end_turn"`, because Open WebUI's converter for that endpoint has no error branch to map a card onto. The retry reduces how often such a card is the outcome; it does not change the shape of the one that is left. A caller relying on `stop_reason` to tell success from failure has to read the card's own text on that endpoint.

### C) API callers with no chat: HTTP error instead of a card

A card is written into a chat, for a person to read. A caller with no chat to write it into has nothing to show it in, and no way to tell a failure from a success — so on that leg the rejection leaves the pipe as an HTTP error rather than as Markdown.

| Condition | Result |
| --- | --- |
| No truthy `chat_id` **or** no truthy `message_id`, `stream: false`, on any path except the two Anthropic Messages paths (`/api/v1/messages`, `/api/message`) | `StreamingResponse`, `status_code: 400`, `Content-Type: application/json` |
| No truthy `chat_id` **or** no truthy `message_id`, `stream: false`, a 200 whose body is **not a decodable JSON object** (a proxy's HTML error page, a truncated stream, `b"[1,2,3]"`), on any path except the two Anthropic Messages paths — the body is not a provider error, so it escapes the provider-error escape entirely and the orchestrator builds the envelope for it | `StreamingResponse`, `status_code: 400`, `Content-Type: application/json`, `code: 502`; `message` carries the endpoint that answered and the upstream `Content-Type`, and quotes none of the body — it is composed by the pipe, not transported from it |
| No truthy `chat_id` **or** no truthy `message_id`, `stream: false`, no task, on any path except the two Anthropic Messages paths, and the refusal is one of the five **local admissions** — a tripped circuit breaker, warmup failure, a missing request queue, a full queue (`Server busy (503)`, including the same refusal reaching a request already waiting for a permit when the pipe is superseded) or a pre-enqueue setup failure | `StreamingResponse`, `status_code: 429` for the breaker and `503` for the other four, `Content-Type: application/json` |
| Any truthy `chat_id` **and** `message_id` | the card, unchanged |
| `stream: true` | the card, as a terminal error chunk with `done: true` and then the end of the stream — the escape's return value is discarded: on a streamed turn `pipe.py:1554` hands back `_stream()`, which never reads the job future, so the `StreamingResponse` the escape built is thrown away. A streamed turn is never left with an empty assistant message: an admission refusal that reaches a request already dequeued ends the stream with the card, not with nothing. The card is the whole response body there, and it keeps the provider's verbatim text by design; see [Security & Encryption](security_and_encryption.md#log-safety) |

The body is the error envelope, not the upstream payload:

```json
{"error": {"message": "<OpenRouter's own message>", "code": 503}}
```

For a mangled body `<the upstream message>` is the reason described above — the endpoint and the `Content-Type`, composed rather than transported — and not an upstream-authored sentence, because there is no upstream message to quote.

`code` is the **upstream status when the provider sent one**, and **`502` when the pipe could not read one** — a mangled whole body is the case that produces a `502`, because the status the pipe accepted (a `200`) is not a status the caller can act on, and the provider never said anything about the exchange. The message names the endpoint that answered, not the pipe, and on a mangled body it carries the same `Content-Type` the chat card shows — but not the excerpt, because that leg's body is composed rather than transported. One sentence, rendered once, delivered on the three card legs.

**This row covers one failure class, and the row above it does not widen it.** The envelope is built for `UpstreamBodyUnreadable` and for provider errors; the other failures the pipe absorbs into a card — connection, timeout, and its own internal errors — still reach a chatless caller as their card text inside a `200`, because the streaming loop catches them first and renders a card. A caller branching on `status_code` sees a `400` for a mangled body and for a provider rejection, and a `200` for a timeout; that asymmetry is the loop's, and closing it would mean re-raising from the loop's own `except Exception`, which the task routes and `CancelledError` make unsafe.

**The status is normalised to `400` for an upstream rejection, and the upstream status travels in the body's `code`.** A locally-raised admission status is the exception and is returned as itself: the normalisation exists to mirror what Open WebUI does with a provider's status, and a refusal the pipe raised itself has no upstream status to normalise. There are five such refusals and two statuses between them, and the split is the cause's — 429 for the per-user breaker, 503 for the four conditions of the process. That is what Open WebUI does for its own models: `routers/openai.py:1736` returns `JSONResponse(status_code=r.status)`, and a task route only converts a *raised* exception into a 400 (`routers/tasks.py:205-211`) — a returned response passes through. Normalising to 400 matches that shape while keeping the caller's one branch (`code >= 400`) working, and OpenRouter's own message survives verbatim into `message`, so nothing of *OpenRouter's* account of the failure is lost.

The body is the uniform envelope for every case, including a 5xx that carries a provider's own `metadata.raw`. That object has no `code` field, so emitting it verbatim would drop the one signal the normalisation exists to preserve.

**`message` is composed from the top level of OpenRouter's own `error` object only.** A provider's own text, which OpenRouter nests under `error.metadata.raw` (its reference calls that field "the provider's verbatim payload"), is not promoted into it at any nesting depth; it reaches the chat card and the operator's WARNING instead, which is where the diagnosis lives. An integration that branched on the provider's specific wording in `error.message` now sees OpenRouter's, and when `metadata.raw` merely mirrors `error.message` — the shape OpenRouter sends when it has no provider text of its own — nothing changes at all. `error.code` and the `Retry-After` header are untouched.

**The gate is truthiness, on two keys.** `main.py:1243` builds `chat_id = form_data.pop('chat_id', None) or ''` for a caller with no chat, so the keys are *present and empty*; `utils/middleware.py:3286` tests truthiness for the same reason. A third key would be stricter than the idiom it copies, and a chat turn that races the socket (`session_id` is `undefined` until the socket connects) would lose its card. `__event_emitter__` is attached for both kinds of caller (`main.py:1277-1306` sets all three keys; `functions.py:239` tests key presence), so it is not a discriminator.

**Open WebUI's own task routes reach this gate**, and that is the point: they send a `chat_id` and no `message_id`, and Open WebUI's own models surface a provider failure on those routes as a non-2xx — `routers/openai.py:1736` returns `JSONResponse(status_code=r.status)`, and a `StreamingResponse` a task route returns passes through its own handler untouched. Making the pipe agree with its own models is what the truthiness gate buys. Of the eight task routes, seven are consumed by the pipe's task-model adapter, which answers 200 with a body that depends on the task kind — a contextual card for a kind Open WebUI persists, `""` for every other; **Mixture of Agents is the one that reaches the gate** (`task_model_adapter.py:90` excludes exactly one task name, `moa_response_generation`), and it gets the same non-2xx its host's own models get — a failure there reaches the browser console, not a chat. **Not for `context_compaction`, `context_summary` or `memory_review`**, though those three meet the gate too: the task adapter catches the fault itself, retries, and returns `""` before the request loop is ever reached, so no envelope is built and nothing reaches a console. Those three are visible only in the backend log, as `Task model attempt N/M failed` — see [Task models and housekeeping](task_models_and_housekeeping.md#how-the-pipe-detects-a-task-request).

**`/api/v1/messages` is excluded**, for every failure kind the escape covers. `main.py:2068-2077` re-wraps *any* `StreamingResponse` through `openai_stream_to_anthropic_stream`, which skips every line that is not SSE `data:` and every payload without `choices` — it has no error branch at all. An HTTP error would arrive as an empty `end_turn` with the error text gone, which is strictly worse than the card. A mangled body on that endpoint keeps its card for the same reason a provider rejection does.

---

## Template rendering rules

Templates are Markdown strings with:
- Placeholder variables like `{error_id}` and `{timestamp}`
- Optional conditional blocks:
  - `{{#if some_variable}} ... {{/if}}`

Rendering rules implemented by the pipe:
- A line is dropped when a placeholder it uses was supplied but came out empty; a name the message never supplies is left in the text verbatim, so wrap such a line in `{{#if name}}` and it is left out instead.
- A `{name}` token belongs to the template, never to a value: a substituted value is shown verbatim and is never re-read as a placeholder.
- Provider-controlled values arrive already shaped for the card: inside a backtick span, on a heading or on a bare card line they are collapsed to one line and their backticks removed; inside a fence they arrive fenced, so a template must not fence such a value again.
- A boolean placeholder renders `True` when the value is true, and its line is dropped when the value is false; the `{{#if}}` form of the same value behaves the same way.
- A value made only of backticks renders as `_` rather than deleting its own line. Backticks are stripped after whitespace collapses, so a value of ` ``` ` would otherwise become empty and the drop rule above would remove the whole bullet — on the shipped card, one such word from a provider erased the entire `### Error:` section. A value that is legitimately empty (`""`, `"   "`, a blank line) still reads as empty, so its `{{#if}}` gate and the drop rule keep working.
- Conditional blocks render only when the referenced variable is “present”, by the same rule the line-drop rule above uses, so a value's guarded and unguarded spellings always agree. A number counts as present at any value, `0` included: a rate-limit card whose `Retry-After` has expired shows `**Retry after:** 0s`, which is the pipe's way of saying the delay is over and the request may be sent again now. Only a missing value, an empty string, an empty collection, `None` and a `False` boolean count as absent.

Minimal example:

```markdown
### Request failed
**Error ID:** `{error_id}`
{{#if support_email}}**Support:** {support_email}{{/if}}
```

---

## Common template variables

### Variables available to most templates
The pipe always provides these for `_emit_templated_error` templates:
- `error_id`, `timestamp`, `session_id`, `user_id`, `support_email`, `support_url`

It then merges in per-error variables (for example `status_code`, `reason`, `timeout_seconds`, `error_type`).

### Variables for OpenRouter “request rejected” templates
The OpenRouter error formatter supports a larger set of optional values, including:
- `heading`, `detail`, `sanitized_detail`
- `openrouter_code`, `openrouter_message`
- `upstream_type`, `upstream_message`
- `provider`, `requested_model`, `api_model_id`, `normalized_model_id`
- `retry_after_seconds`, `rate_limit_type`
- `include_model_limits`, `context_limit_tokens`, `max_output_tokens`
- `metadata_json`, `provider_raw_json`, `diagnostics` — every value derived from the provider's payload is cut at 16,384 characters, with a marker naming how many characters were removed: the two JSON values `metadata_json` and `provider_raw_json`, the five inline copies of the provider's message (`detail`, `sanitized_detail`, `reason`, `upstream_message`, `openrouter_message`) and the joined `moderation_reasons` list. A cut JSON value is no longer parseable JSON; a cut inline value is still one logical line, its marker space-joined onto the card's line. `raw_body` and `flagged_excerpt` are not cut and arrive whole
- `error_chunk_id`, `error_chunk_created`, `is_streaming_error`, `native_finish_reason`, `request_id_reference`
- `streaming_provider`, `streaming_model` — filled only for a failure reported inside a reply that has already started; empty on a rejected request however the provider is named, and empty when such a failure names no provider at all

Because OpenRouter/provider responses vary, treat these fields as optional and wrap them in `{{#if ...}}` blocks. Guarding is optional for `streaming_provider` and `streaming_model` — they are filled for a mid-reply failure that names a provider and empty for every other failure, so a line using them already disappears when it should.

`max_output_tokens` on an error card is the provider's **advertised** ceiling, read straight from the catalog entry (`core/errors.py`). It is deliberately not the value the pipe sends when `USE_MODEL_MAX_OUTPUT_TOKENS` is on: that is the smaller of the advertised ceiling and half the model's context window. A diagnostic about a failure should report the provider's own limit, so the two numbers are meant to differ.

The pipe does the span and fence work on these values itself: a value placed inside a backtick span, on a `### ` heading, or on a bare `**…**` / `- ` line arrives as one logical line with its backticks removed, and a value placed in a fenced block arrives inside a fence long enough to contain it, so a custom template does not have to. The pipe's own numbers and labels (`status_code`, `retry_after_seconds`, `context_limit_tokens`, `max_output_tokens`, `diagnostics`) are already single-line.

### When the model-limits block renders

`include_model_limits` guards the section that prints the model's context window and output cap. It is set when the rejection looks like a context overflow **and** the catalog knows at least one of those two numbers for the model.

A rejection counts as a context overflow when either holds:
- OpenRouter tags the error with the typed code `context_length_exceeded`. This is read from `error.metadata.error_type` on Chat Completions, from the top-level `error_type` on Responses, and from `error.error_type` on the Anthropic Messages envelope a gateway in front of the pipe puts on the wire — the three places it can arrive in, read in that order. On Responses the native error code is lossy — `context_length_exceeded` collapses into `invalid_prompt` — so the typed field is the only reliable signal there.
- The message text names the remedy, matched against both the current wording (“context compression”) and the wording OpenRouter used before the feature was renamed (`or use the "middle-out"`).

The provider's own error type (surfaced as `upstream_type`) is deliberately **not** consulted: it carries the upstream vendor's category, such as `invalid_request_error`, which covers far more than context overflows.

---

## Provider-specific recovery (reasoning/thinking mismatches)

Some providers reject “reasoning” requests when their own “thinking” mode is not enabled for that model/provider combination (for example error messages referencing `thinking_config.include_thoughts`).

When the pipe detects this condition, it can retry once with reasoning disabled by:
- clearing `reasoning`, or setting it to `{"effort": "none"}`
- removing the legacy `include_reasoning` flag when any id in `models` fails to list it — and, where the dropped value was `false` and the primary lists `reasoning`, carrying thinking off as `reasoning: {"effort": "none"}` instead; on a request with no `model_fallback` the flag is simply turned off

This retry is only available while the attempt has published nothing the reader keeps: once a thinking box or a tool card is on screen the attempt owns the turn, and the error is reported below what the reader already has instead.

This behavior is intended to convert certain provider-side “configuration mismatch” failures into a successful answer without requiring the user to change settings mid-conversation.

A row the catalogue marks `reasoning.mandatory` is the exception: the pipe does not retry it, because stripping reasoning from a request that must think is the very shape the row refuses. The first response is returned to the user with the provider's own diagnostic rather than a resend that would fail the same way and hide it.

### When the endpoint rejects the effort itself

A second, separate retry fires on a rejection of `reasoning.effort` as a value rather than of reasoning as such: the provider names `reasoning.effort` with code `unsupported_value` and lists the levels it accepts (`Supported values are: …`). The pipe then resends once at the level nearest the one that was refused, measured in the order `none`, `minimal`, `low`, `medium`, `high`, `xhigh`; where two supported levels are the same distance away the lower of the two is chosen. A request at or below the lowest level the endpoint accepts is retried at that lowest, and a request at or above the highest at that highest.

The chat sees a status line, `Adjusting reasoning effort from '<refused>' to '<retried>' (model doesn't support '<refused>')`, naming the level the resend actually carries — on a model whose reasoning is mandatory that is the level the resend goes out with after the rule below has been applied, not the one the endpoint's list offered.

The `reasoning.mandatory` rule described in [Model catalog and routing intelligence](model_catalog_and_routing_intelligence.md) is re-established on this arm as well as on the two build sites: after the retry writes `reasoning["effort"]`, a body carrying `none` is sent only to a model that is allowed to stop reasoning, and a model that must think receives the lowest level its catalogue row lists, or no `effort` key at all when the row lists none other than `none`. A model that can stop keeps the object the first attempt built, so the retry does not strip the `enabled` and `summary` the pipe's own settings had already put there.

---

## Valve configuration (where to customize)

### Support contact valves
- `SUPPORT_EMAIL`
- `SUPPORT_URL`

### Template override valves
- `OPENROUTER_ERROR_TEMPLATE`
- `AUTHENTICATION_ERROR_TEMPLATE`
- `INSUFFICIENT_CREDITS_TEMPLATE`
- `RATE_LIMIT_TEMPLATE`
- `SERVER_TIMEOUT_TEMPLATE`
- `PAYLOAD_TOO_LARGE_TEMPLATE`
- `NETWORK_TIMEOUT_TEMPLATE`
- `CONNECTION_ERROR_TEMPLATE`
- `SERVICE_ERROR_TEMPLATE`
- `INTERNAL_ERROR_TEMPLATE`
- `ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE`
- `DIRECT_UPLOAD_FAILURE_TEMPLATE`
- `MODEL_RESTRICTED_TEMPLATE`
- `STREAM_INTERRUPTED_TEMPLATE`

Every valve above restores itself: clear its box and save, and the built-in text is written back for that valve alone. A box holding only spaces or newlines counts as cleared.

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for defaults and descriptions.

---

## Customization workflow (recommended)

1. Start from the default templates (edit minimally).
2. Wrap optional fields in `{{#if ...}}` blocks so missing variables do not render blank or confusing lines.
3. Keep user-facing messages actionable (what happened, what to do next).
4. Validate changes by triggering known failure modes in a controlled environment (see “Testing”).
5. To start over on any one of these valves, clear its box and save. The built-in text for that valve is written back and appears in the box on the next load, ready to edit again; nothing else is changed.

---

## Customization examples

### Example: minimal corporate support footer

```markdown
### ⚠️ Request failed
**Error ID:** `{error_id}`
{{#if timestamp}}**Time:** {timestamp}{{/if}}

If this persists, contact support and include the Error ID.
{{#if support_url}}**Support:** {support_url}{{/if}}
```

### Example: operator-forward template (adds diagnostic JSON)

A value that arrives already fenced must not be fenced again by the template: `metadata_json` carries its
own fence, so a template that adds a ```json … ``` pair around it closes that fence early and the JSON
escapes the block.

You do not have to edit a stored template that already fences one of these values. The renderer tracks
the template's own fences, so a row written as ` ```json ` / `{metadata_json}` / ` ``` ` — the shape
shipped before 2.7.4, and the shape the pipe's own example used to show — renders as exactly one
code block holding the payload, with the label above it and the card text after it. A template that
does not fence the value is unaffected. The same holds for `{raw_body}`, `{flagged_excerpt}` and
`{provider_raw_json}`; a value is only unwrapped when the template supplies the fence. A value that arrives cut ends mid-payload and its `...(truncated: N characters omitted)...` marker sits on its own line inside the fence, so read a cut value as the start of the payload rather than the whole of it. An inline cut value is the other shape: its marker is space-joined onto the card's line rather than given a line of its own, because an inline value is always one logical line, so the same marker text appears mid-line there.

````markdown
### 🧾 Provider error
**Error ID:** `{error_id}`
{{#if provider}}**Provider:** {provider}{{/if}}
{{#if requested_model}}**Model:** {requested_model}{{/if}}

{{#if metadata_json}}
**Metadata:**
{metadata_json}
{{/if}}
````

---

## Troubleshooting (template system)

- Template changes do not apply:
  - Ensure you edited the correct function’s valves in Open WebUI (and saved).
  - Ensure you edited the correct template valve for the error type you’re testing.
- Conditionals do not render:
  - The variable may be empty for that error path; wrap the whole section in `{{#if var}}`.
- Templates render blank:
  - A placeholder on a line can cause the entire line to be dropped if the variable is empty; prefer multi-line blocks with conditionals for optional sections.
  - A line whose **own** placeholder resolves to a missing or empty value is omitted; a value that itself contains `{name}` is shown verbatim, never re-read as a placeholder.
  - If every line drops, the card is not empty: the pipe falls back to its built-in provider-error card, which names the model and an error id to quote. The same applies on a channel chat, where a card written entirely out of withheld values renders empty before the scrub — the fallback is rendered through the same scrub, so it names the model and the error id and nothing withheld.

---

## Operator runbook (correlating user reports with logs)

1. Ask the user for the `error_id` displayed in the UI.
2. Search backend logs for `[{error_id}]`.
3. Use `session_id` and `user_id` (when available) to correlate with other telemetry (Redis cost snapshots, session log archives, etc.). On a channel those two are withheld from the card the reader sees, so the `error_id` is the handle to start from there; the ids are still in the operator's log line, which the card's audience never sees.

See also: [Session Log Storage](session_log_storage.md) and [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md).

---

## Testing

This repository includes tests for template behavior and error rendering. Prefer running the specific test modules first, then the full suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_error_handling.py tests/test_template_valve_restore.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```
