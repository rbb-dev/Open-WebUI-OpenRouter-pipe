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
3. **API callers with no chat**, where there is nowhere to write a card: a provider rejection is returned as an HTTP error instead, carrying the resolved status on the status line and the same number in the body's `code`. The same leg carries the five pre-job refusals, which are further conditions on this path: a tripped per-user circuit breaker, warmup failure, a missing request queue, a full queue and a pre-enqueue setup failure. Each is raised locally and has no upstream status behind it, so it leaves as its own status rather than a 400 — **429** for the breaker, which is a per-entity rate limit, and **503** for the other four, which are conditions of this process rather than of one caller. Gated on a truthy `chat_id` **and** `message_id` — Open WebUI's own idiom for "is there a chat to write this into" (`main.py:1703`, `utils/middleware.py:3286`) — streamed or not, since a streamed caller receives the same envelope as a terminal in-band frame where no HTTP status can travel, and never on an Anthropic Messages path (`main.py:2054-2063` re-wraps the response for the Anthropic converter, which has no error branch). The gate is a negative test on the request path, not a check for `/api/chat/completions`, so it also fires on Open WebUI's task routes. That escape is for *failures*: a turn that ran and answered, degraded but successful, stays a `200` and carries what was degraded in its body — the refused-attachment row of [API callers with no chat](#c-api-callers-with-no-chat-http-error-instead-of-a-card) is that shape. It reaches the chat, task, image and video legs alike: a video turn that OpenRouter rejects, or that fails on a fault the pipe owns, leaves as an HTTP error on that leg too, and an image turn that OpenRouter rejects, or whose body a proxy rewrote, does the same. See [API callers with no chat](#c-api-callers-with-no-chat-http-error-instead-of-a-card).

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
— and it promises no retry. A third arm says the same thing without the fault: a store that never came up has no
database to read from, so it announces the same missing round in the same words, logs a warning saying the store is not
configured, and does **not** charge the DB breaker — it is a state, not a failure, and charging it would open the
breaker and tell the next turn its round was skipped "due to repeated errors", which is a cause this deployment does
not have. All three read-side arms fire **once per degraded episode** rather than once per turn, and re-arm on the
first read that reaches the success arm: the loader runs once per artifact group per turn, and Open WebUI fires a toast
per notification frame without deduping or rate-limiting them, so an unlatched notice repeats the same banner on every
turn of a broken chat for as long as the fault lasts. The fault and breaker arms share one latch, because both say the
same thing — earlier results did not reach the model — and separate keys would let a flapping database alternate two
banners forever; the not-configured arm keeps its own, because it is a state that cannot clear while it can still fire.
A latch is set only when a live turn actually received the notice, so a read with no emitter to report on does not mute
the next one. The write side has the same split. Sealing a round for storage runs in its own guarded arm ahead of the database write, so a cipher that cannot be built or a payload no serialiser will take drops the round and logs a WARNING naming the seal and the row count, without charging the DB breaker and without a notice that names the database — nothing asked the database to do anything, and a fault the database did not have would send the reader looking in the wrong place. A failure of the write itself keeps every consequence it has today: the rows are dropped, the breaker is charged, the ERROR is logged and the person is told the items could not be written to the database. It is kept apart from the two budget notices on purpose: a budget that trimmed successfully
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
than once per turn, and re-arms when the key becomes readable again. The read-side notices above use the same rule and
re-arm on the first successful read.

### Where a card lands: saved chat or channel

A card is delivered differently depending on whether the chat id is a `channel:` id. The
chat id is carried to the error arms on a context variable the pipe sets in three
places (before the job is enqueued, inside the job itself, and at the head of the stream
generator the caller consumes that job's answer from — its body does not start until the
caller iterates it, which is after the other two have been reset), so the choice of path is
a property of the conversation, not of the frame:

- **Saved chat.** The card travels as a `chat:message` snapshot and the turn's closing
  `chat:completion` carries no content. The stored message is the answer plus the card, in
  that order. On a **continuing** turn it is the other way round: the card travels as
  `chat:message:error`, the closing frame carries no content and no `error` key on any
  leg, and the bubble keeps the prefix the turn was continuing. The closing frame omits
  `error` because that frame has already delivered the card: a `chat:completion` carrying
  `error` puts the browser's generic "Uh-oh! There was an issue with the response." over
  whatever arrived before it, which is what would lose the `error_id` to quote.
  `chat:message` assigns `message.content`
  absolutely on the browser, so publishing the snapshot there would replace the whole
  stored reply with the card; the error frame goes to the error area under the message and
  leaves the content alone. The stored record is unaffected either way — it is the answer
  plus the card, in that order.
- **A notice the turn is not over for.** A second saved-chat case travels on
  `chat:message:error` too, with **no `done` key**: a refusal the pipe issues before the model
  is called — an attachment it dropped — is not the end of the turn, and the model answers past
  it. The browser gates that branch's close on `data.done === true`, so the card is read in the
  error area while the reply stays open, the caret keeps pulsing and Stop keeps working, and
  the turn's own closing `chat:completion` is the one frame that finishes it. A
  `chat:completion` carrying `error` would set `message.error` **and** `message.done`
  regardless of its own `done`, because the browser's `handleOpenAIError` sets the close from
  the presence of `error` and never reads that field — so a notice sent there ends the reply
  early, badges it as a failed response, and leaves the model answering into a message the UI
  already considers finished. On a **channel** the same notice still closes the message: the
  channel emitter closes on any `chat:message:error` and ignores the frame's `done`, and no
  frame the pipe can send leaves a channel message open after one. That is the host's
  behaviour, accepted and recorded in a test rather than worked around.
- **Channel.** The channel's emitter is a different function from the socket emitter, and it
  honours a different set of types. It has no `chat:message` branch at all, so on a channel
  the card is written by a `chat:message:error` frame — which Open WebUI prefixes with
  `Error: ` — and the closing `chat:completion` carries the same text as its `content`, so
  the prefixed write is replaced by the clean text and the stored message reads as the
  answer followed by the card. The channel leg **replaces** the message rather than
  appending to it, so the closing frame's content must be the whole message or the card is
  lost.

The five pre-job refusals (a tripped circuit breaker, warmup, a missing request queue, a
full queue, and a pre-enqueue failure) never install the middleware emitter: four of
them return before a stream queue is built, and the queue-full arm builds one and
refuses before the dispatch worker dequeues it.
So all five emit straight to the channel emitter and take the same channel path as any
other card. A caller with no chat to write that card into gets
the refusal as a status instead, carrying the same sentence as the error message: 429 for
the breaker, 503 for the other four.

**A channel card is reduced, a saved-chat card is not.** Every member of a channel can
read what the pipe writes there, so a card that reaches a channel is rendered as though
nineteen of its placeholders were empty: `session_id`, `user_id`, `detail`,
`sanitized_detail`, `reason`, `openrouter_message`, `upstream_message`, `upstream_type`,
`moderation_reasons`, `flagged_excerpt`, `raw_body`, `metadata_json`,
`provider_raw_json`, `body_excerpt`, `required_cost`, `account_balance`,
`model_id_filter`, `free_model_filter` and `tool_calling_filter`. Those nineteen cover the
templated arms. The **plain** arm has no template and so no placeholder to blank, and it is
reduced twice over: the file-error messages the pipe builds never carry the reference at all
— the sentence says a referenced picture or file is no longer available in Open WebUI storage,
and no host, path, query or file id is interpolated into it, which is also what keeps that
reference out of the operator's own WARNING and out of a Fusion judge's prompt, both of which
read the same string — and then the emission chokepoint reduces whatever reaches it on a
channel through `_channel_safe_card_text()`, which replaces a reference-shaped span whole
(path, file id, filename and query string together) with *a file reference withheld on a
channel* and leaves the pipe's own sentence around it intact. The first fourteen are the
requester's identifiers, the text they wrote and the
provider's own prose about it — and a provider's rejection body routinely quotes the
prompt back, so the prose *can* be their words. `body_excerpt` is on the list for the
same reason `raw_body` is: it is the part of a reply the pipe could not parse, which is
the part an attacker chooses, and a WAF challenge page is not something to broadcast to a
room. The last five are not the requester's at all but the deployment's own: what one turn
would cost and what the account has left, and the three model-filter settings the
orchestrator copies straight out of the valves, `MODEL_ID` among them, whose ids an admin
may have deliberately kept off the picker. Both subjects are somebody else's data in a
room of other people, so the reduction covers both. The reduction happens at the render, on
every path that can reach a channel, and it is keyed on the surface rather than on the
template: an admin's custom template gets the same answer, and there is no valve to put
the ids back. `error_id`, the model, the provider, `openrouter_code`, `status_code`, the
limits and the rate-limit fields still render, so the card still says what failed. The
account's balance, what one request would have cost, and the three filter settings the
admin chose are withheld with the rest: they are the deployment's own figures, not
anything about the request that failed.

**A plain error card on a channel is reduced too, and by a different rule.** Not every
card is rendered from a template: the plain (non-templated) error arm is handed a finished
sentence by its caller and puts it on the wire as it stands. Two of those callers hand it
`e.user_message` from a file error, and the sentence those errors are built from — a request
whose file cannot be read is reported as *"A referenced file is no longer available in Open
WebUI storage."* — interpolates no reference into itself at all: no host, path, filename,
file id or presigned query string, so the arm is handed nothing a channel could learn. That
is the closure, and it is also what keeps the reference out of the operator's own WARNING
and out of a Fusion judge's prompt, both of which read the same string. The chokepoint is
the second, defensive layer under it: the plain arm withholds any reference-shaped span that
reaches it anyway on a channel chat, replacing the whole span (path, filename, id and query
string together) with *a file reference withheld on a channel*, and the pipe's own sentence
around it survives. The pipe's own refusals — *"Request failed. Please retry."*, *"Server
busy (503)"* — carry no such span and reach the room in their own words. A saved chat and an
API caller are not reduced by the chokepoint, so anything that does reach them with a
reference in it still carries it, and the operator's own log line always does; with the
construction no longer producing one, a file error now names the file only as *"A referenced
file"*, never as the thing itself.

**A temporary chat's card is reduced by one key, and for a different reason.** A
`temporary:` or `local:` chat's `session_id` is withheld on the same terms, and the rest
of the card is untouched: the reduction is by one key rather than by thirteen, and it is
not there because the audience is shared. Open WebUI mints a temporary chat's id by
prefixing the browser's own socket id, so the value is a live session handle rather than a
label — the `/api/tasks/chat/{chat_id}` routes resolve the chat id back to it and use the
user it resolves to as the ownership check — and a card that printed it would hand out a
capability rather than a correlation string. `user_id` is *not* withheld there: it is the
Open WebUI account GUID, present in every chat, and it is not a session handle. The
operator's own log line is withheld the same way, so on a temporary chat `error_id` is
the only handle either side has.

A template author therefore does have to write one thing differently for a channel: a
line whose placeholder came out empty is omitted, so a line that exists only to show the
session id or the flagged excerpt simply is not there in a room. The same is true of a
line that exists only to show the session id on a temporary chat. A card that ends early,
or that relies on the `Error: ` prefix being visible, also reads differently in a channel
than in a saved chat. Nothing else about the routing changes.

---

## Error IDs and enriched context

For templated errors, the pipe generates:
- `error_id`: 16 hex characters (`secrets.token_hex(8)`)
- `timestamp`: ISO 8601 UTC timestamp
- `session_id` and `user_id` (when available) — withheld from a card that reaches a channel, because every
  member of the room would otherwise read them. `session_id` alone is also withheld on a temporary chat, where
  it is the browser's own socket id and therefore a live session handle; `user_id` is not, and still renders
  there
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
| `402` | `INSUFFICIENT_CREDITS_TEMPLATE`, which adds a **Retry after** row carrying the delay from the rejection's own `Retry-After` header, and only when that rejection carried one |
| `408` | `SERVER_TIMEOUT_TEMPLATE` |
| `413` | `PAYLOAD_TOO_LARGE_TEMPLATE` |
| `429` | `RATE_LIMIT_TEMPLATE` |
| `>= 500` | `SERVICE_ERROR_TEMPLATE` |
| other / default | `OPENROUTER_ERROR_TEMPLATE` |

“Other / default” is every remaining case, including a missing status: `403`, `404`, `422`, a `400` for a malformed reference URL, and a provider moderation refusal all render `OPENROUTER_ERROR_TEMPLATE`. Anything written into that template is therefore read by failures that have nothing in common beyond not having a template of their own, so advice specific to one cause belongs in a `{{#if}}` block rather than in the template's unconditional text.

**One input other than the status bypasses the table, and it changes the card only.** A body carrying a content decision — `metadata.patterns`, `metadata.reasons` or `metadata.flagged_input` — renders `OPENROUTER_ERROR_TEMPLATE` whatever status it resolved to, on the rejection path and inside a started reply alike, because a decline is not the cause any of the other six cards describe. The status itself does not change: the wire line stays the floor for the retry decision, for the credential latch and for the breaker, and it stays on the card's `{status_code}` row, so the number the operator's logs and OpenRouter's support are keyed on is the number that was sent. A `401` is the one exclusion: `is_sign_in_failure` accepts every `401` before it looks at a marker, and the pipe has already noted the credential failure by the time the card is chosen, so a card saying the prompt was declined would sit on top of a pipe that is still blaming the key.

**A `429` or `5xx` is retried before its card renders, and so is a `408` that names the provider-timeout kind.** Such a failure is one of the temporary class, so the pipe re-sends the request up to `TRANSIENT_RETRY_MAX_ATTEMPTS` extra times before showing anything; a `Retry-After` the server sends sets the wait, truncated to `TRANSIENT_RETRY_MAX_WAIT_SECONDS` rather than replaced by it. The card therefore means the retries are spent, and wording in it can say what an operator would expect of a final failure. A `401` — and every other status in the table apart from that `408` — is never retried, and nothing at all is retried once text has been shown to the reader. A body carrying a content decision is the one exception to that set: it is not retried, whatever status it arrived on, because re-sending a prompt a guardrail declined cannot unblock it, and it is rendered with the same card — `OPENROUTER_ERROR_TEMPLATE` — on every status, so the two halves read as one decision rather than as a status-dependent retry and a status-independent message.

**A non-streamed 200 on `/responses` whose `output` is neither a list nor null is treated like a dropped answer.** The request is re-sent up to `TRANSIENT_RETRY_MAX_ATTEMPTS` extra times, then reported and counted once, because the reader downstream can consume nothing else: an object, a string or a number in that key is a finished turn with no item in it, which reads as a silent success and clears the reader's breaker record. An `output` that is an empty list or null is a different case and is described below: the connection worked and the model returned nothing.

**A `403` is not one thing, and a content decision does not pause the user's background work.** OpenRouter uses the status for at least four different answers: a key it rejected, a key that is valid but lacks permission for the model, a provider content policy, and a guardrail that declined the prompt before it reached any provider. The first is the person's to fix by signing in again; the other three are not, so only the first arms the sixty-second credential pause that otherwise leaves the next chats titled "Chat" with no error shown. A guardrail block is the case that carries no type at all — its only marker is the `patterns` the guardrail matched — and it is the case a status-only reading blames on the key. Which field named the block does not change the answer: the carve-out is position-independent, reading the typed kind or the native code, whichever field carried it, and both names are derived from the status tables rather than written out beside them, so a new `403` row added to either table is recognised as a decline without anyone editing a second list — and a property test over both tables is what keeps that derivation honest. Rendering follows the status, so the four answers above do not all render the same card: a body carrying a decline marker is read as a content decision whatever status it arrived on — on a rejection it keeps the status the wire delivered, and inside a started reply it keeps the status the body's own `error.code` names, as the next paragraph explains — and it renders `OPENROUTER_ERROR_TEMPLATE` on every one of them, the `403` a guardrail normally arrives on included, while a `403` naming no marker and a kind documented at another status — a `rate_limit_exceeded`, say — resolves to that status and renders its card. Either way OpenRouter's own reason is shown.

These templates are used for the `OpenRouterAPIError` path (and for certain HTTP status errors that are converted into an OpenRouter error object by reading the response body best-effort).

**A rejected request and a failure reported inside a reply that has already started are read the same way.** OpenRouter commits `200 OK` as soon as a provider accepts a request, so a rate limit, an overloaded or unreachable provider, a provider timeout, an exhausted balance or a sign-in failure occurring after that point arrives inside the reply instead of as a status. The pipe reads the kind of failure OpenRouter names there - `error.metadata.error_type` on Chat Completions, the top-level `error_type` on Responses, `error.error_type` on the Anthropic Messages envelope a gateway in front of the pipe puts on the wire, the same fields the model-limits section below uses - and renders the template for the status that kind is documented with, so the person reads the message written for that failure rather than the rejected-request one; a body that also carries a decline marker is the exception, and renders the decline card whatever the kind resolved to. A kind the pipe does not recognise falls back to the status reported alongside it, and then to `OPENROUTER_ERROR_TEMPLATE`. On a rejection the pipe reads the same kind from the same three fields, in the same order, and renders the template for the status that kind is documented with, so a `5xx` carrying `authentication` on the Responses skin is read as the rejected credential it is rather than as a server-side error, and the same body classifies identically whichever of the two paths delivered it. One case needs the leg it arrived on, and it is the `401`-naming kinds: inside a started reply there is no wire to floor, so a decline marker means the typed kind cannot resolve the status at all — a provider that collapses its guardrail block into `error_type: authentication` would otherwise read as a dead key, be carded as one, and arm the credential pause for a prompt the reader declined — and the body's own `error.code` is what carries it instead. A decline that names no numeric code takes the status that code's own table row gives, or the caller's placeholder when the body names none, and the only body that still arms the pause is one whose own `error.code` is a `401`. A body that names no kind is still read by a named `error.code`: `invalid_api_key` is read as a rejected key, `image_content_policy_violation` as a content block and `server_error` as a server fault, so the card and the credential pause follow the failure the body names rather than the line it arrived on. One bound holds there: a `401` or `403` line is never moved to a `5xx` by a code, so a `401` carrying `server_error` still reads as the rejected credential it is, still arms the pause and is still sent once. The HTTP status is the floor in both directions: on a rejection, where the wire names no kind the pipe knows and no code the pipe knows either, the line stands whatever numeric `error.code` the body carries beside it — a `429` or `5xx` stays retriable, and a `401`, `403` or `400` is not re-read as a rate limit or a server fault — because the native code on that skin is collapsed to a `5xx`, the code is documented as best-effort, and a line the server chose must not be moved by a number in the body. Inside a reply there is no wire to floor, so the code reported there still decides. A temporary one — `429` or `5xx` — reported before any content is streamed is retried first, and renders exactly the same card once the retries are spent; the same kind reported after content is shown is rendered immediately, as described above. One consequence matters when editing `SERVER_TIMEOUT_TEMPLATE`: a provider timeout reported this way is documented as `504`, so it renders `SERVICE_ERROR_TEMPLATE` — unless the same body carries a decline marker, which takes precedence over the kind and renders the decline card. The timeout template is reached by a `408`, whether that is the reply's own status or the code reported inside the reply.

**Every caller uses the table.** Status selection chooses one of these templates, and a body carrying a content decision overrides it onto the decline card; no call site can pass a template that overrides either. The chat orchestrator, the outer request handler, image generation, video generation, and a streaming failure arriving after output has begun all render the same template for the same status and the same marker.

**One exception, and it matters for the wording of `AUTHENTICATION_ERROR_TEMPLATE`.** When the pipe cannot read its own OpenRouter API key it renders that template directly. Nothing was sent, so there is no status and no rejection by OpenRouter — the cause is local: the key setting is blank, or the value stored in it was encrypted under a `WEBUI_SECRET_KEY` that has since changed and can no longer be decrypted. The `401` shown on the card in that case is a display value the pipe supplies, not a status any server returned. Wording written for that box therefore has to fit a local configuration fault as well as a key OpenRouter rejected, which is why the built-in text says the request was not authorised "or the pipe could not read one" rather than asserting that OpenRouter refused the credentials. It is the only template in the table above reached this way; every other template is chosen by status alone, from the table in section B below.

**OpenRouter's own request reference travels with the card.** When a rejection carries one, every template in the table above renders it on its own row, separate from the `error_id` the pipe generates: the pipe's id correlates the pipe's logs, and OpenRouter's is what its support can look up. A rejection that carries none renders no such row. The reference is unavailable on the paths where no request reached OpenRouter — a key the pipe could not read, and a `5xx` raised by the connection itself — and on a failure event that carries no id (a Responses `response.error` or plain `error` event); on Chat Completions OpenRouter's generation id is the reference, also available as `error_chunk_id`. A template edited to show the reference should therefore keep the row inside a conditional, as the built-in text does.

**A turn the pipe refused before sending is archived as a failed turn, and its reason names the control that refused it.** Seven arms in the request orchestrator refuse a turn locally — the two Zero Data Retention arms (routing on, model not on the roster; and routing on, roster unreadable), the two endpoint-override conflicts (a preset, or Direct Uploads, that needs an endpoint the valve denies), a Fusion model forced to `/chat/completions`, an attachment that would not load, and the operator's model restrictions. Each returns a card rather than raising, because a refusal is an answer the person reads and one the Fusion panel needs in full. Returning rather than raising means the job's future resolves cleanly, so the session-log archive would otherwise record the turn as `complete` with an empty `reason` — a clean turn whose chat holds nothing but a refusal card. It is archived as `error` instead, with the pipe's own sentence as the `reason`: the valve that refused it, through the operator's own labels, not the rendered card. The video adapter's own pre-send refusals are archived the same way: its per-user cap, on both the new-generation arm and the resume arm, and its empty-prompt arm each write the control that refused them, so a chat holding nothing but a video refusal card is never filed as a clean turn. An arm there that *answers* instead — the controls card, a clarification, a clip or failure block already stored on the message — writes whether that answer was a success, so the same archive and the same circuit-breaker reset read what the person was actually shown. The card text is deliberately *not* what lands in `meta.json` — it carries a generated error id and a timestamp that identify one rendered message and nothing else, while the sentence names the setting to change. See [Session Log Storage](session_log_storage.md) for what `status` and `reason` hold.

**Clearing a template box and saving restores its built-in text.** Every error template valve behaves this way: leave the box empty (or containing only spaces or newlines), save, and the pipe writes that valve's factory default back, so reopening the Config tab shows the original wording ready to edit again. Only the valve that was cleared is affected. This is how an operator recovers from an edit that went wrong, since the built-in text is not otherwise visible in the interface. The same restore applies when valves are edited through Open WebUI's own Functions valve panel. That rule is scoped to the error templates: clearing a *nullable numeric* setting's box does not restore text, it returns the setting to unset, so a saved row carries no name for it at all.

**A stored value the current release no longer accepts degrades one setting at a time, and says so.** This is the other way a configuration is not what the operator thinks it is. The trigger is a release, not an edit: a bound tightened, a `Literal` member removed or a type narrowed, against a row an installation already wrote. It used to be a total outage instead — the pipe refused to load at all, and Open WebUI re-raised that on every chat, so every user on every worker got a failure with no message naming a setting. It now leaves that one setting at its default, writes a warning naming the field, the stored value and the default it read as, and leaves every other saved value untouched. The warning is the whole signal on an installation with no Config tab, so it is WARNING on first sighting and after a five-minute cooldown, and DEBUG in between rather than a flood on a path that runs per request. The other half of that signal is the setting this release no longer publishes at all — a name a release renamed or removed, which the stored row still carries and which is read by nobody. That name is reported in the same warning, and its stored value is never quoted with it: the name is all the operator needs to delete it from the row, and a name this release does not publish has no annotation left to decide that its value was a secret. Secret valves are the exception in both directions: they are never replaced by their default, and neither their stored value nor their default is ever written to the log. The row repairs itself on the next save from Open WebUI's own valves panel; the Open WebUI *sync* route is the one door that does not repair it, so an operator who repairs a stale row must do it from that panel.

**A filter write the database refused is surfaced as a database fault, not a setting.** Open WebUI's `Functions.update_function_by_id` returns `None` — it never raises — both when the commit failed and when the row is gone, so a write that did not land is invisible to any caller that only wraps the call in `try`/`except`. The diagnostic that names the row as a database fault rather than a setting to change reads only the refusals **that build** met, so an overlapping build — a second model-list refresh, or a second installed copy in the same process — cannot turn a refused install into "enable the install valve". Every switch-off, switch-on and retirement write the pipe makes goes through one helper that reads that return value, and when the answer is `None` it records the row, names the operation it was performing, and says the write will be retried on the next pass. The message is WARNING the first time an hour and DEBUG after that, per row, so a standing fault is one line rather than one per pass. **What to look for:** a `Open WebUI refused the write to <row id> while <operation>` line. `<operation>` is the only clue about *which* write was refused — the refusal set is matched by row-id prefix, so a retirement and an install of the same family read alike at that line, and the operation is what tells them apart. **What it means:** a database fault, not a valve to change; no setting reaches it. **What to do:** check the Open WebUI database's health and disk space, then let the next pass retry, or restart the worker to force one sooner.

### B) Generic templated errors (network/5xx/internal)

When a chat reply's call to OpenRouter fails without being rejected, or anything else goes wrong in the chat reply loop, the pipe picks one of these templates:

| Condition | Template valve |
| --- | --- |
| A timeout before OpenRouter accepted the request, or on one it has accepted and before any answer text arrived | `NETWORK_TIMEOUT_TEMPLATE`. A timeout that lands while an accepted body is still arriving is not this row: the body is reported as lost, nothing is re-sent, and the connection card below is what a reader sees |
| A connection that cannot be opened or drops, a stream that sent nothing the pipe could read on every attempt, or one whose readable frames published nothing a reader could use, an accepted status whose body was lost before it finished arriving, on either endpoint, or a non-streamed 200 carrying no `output` key on `/responses`, or an `output` that is neither a list nor null, and no `choices` on `/chat/completions`, before any answer text has arrived and before the model has named the tool it is calling | `CONNECTION_ERROR_TEMPLATE` |
| An accepted status whose body is not a JSON object (a proxy, WAF or gateway rewrote the reply) | `SERVICE_ERROR_TEMPLATE`, with the quoted body in a separate `{body_excerpt}` block rather than inside `{reason}`. A picture-only image generation whose `/images` reply was rewritten reaches this row too, so it gets the same split — the two facts prose, the excerpt fenced — with the excerpt arriving already fenced rather than fenced by a placeholder in the template |
| A timeout, a failed or dropped connection, or a stream that published nothing a reader could use, after answer text arrived earlier in the reply or after the model named the tool it is calling | `STREAM_INTERRUPTED_TEMPLATE`, appended after the kept text |
| Any other exception | `INTERNAL_ERROR_TEMPLATE` |

The timeout template's `timeout_seconds` is the limit that ran out: `HTTP_CONNECT_TIMEOUT_SECONDS` while connecting, `HTTP_SOCK_READ_SECONDS` while waiting for data, or `HTTP_TOTAL_TIMEOUT_SECONDS` for the whole request. The connection template's `error_type` names the failure (for example `ClientConnectorError`). A connection that closes before the first byte is retried, up to `TRANSIENT_RETRY_MAX_ATTEMPTS` extra tries (three attempts in all by default), and the request is counted once against the breaker however many attempts it took; a retry that arrives after answer text has been shown is not made, and the table above's `STREAM_INTERRUPTED_TEMPLATE` row then applies. A stream that closes early without an error, once any of its events has arrived, also gets `STREAM_INTERRUPTED_TEMPLATE`; see [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md). Picture-only image models, video models and the panel, judge and final-answer calls inside internal Fusion report failures in their own way — a chat caller sees all of them as cards, and a chatless caller sees the failures of the image and video legs as HTTP errors instead — the video leg's both before the submission is answered and on the lifecycle that watches it afterwards, and on the **image** leg two post-submit failures join them there, a provider rejection and a body a proxy rewrote, which answer a caller with no `chat_id` **and** `message_id` with the HTTP error envelope of section C below rather than a card, while a pipe-authored `ImageGenerationError` keeps its card on every caller — and where one of those cards is composed rather than templated, a value a reader must not act on is carried fenced by the value itself rather than by a placeholder in an operator's template, so the fence is there on an installation that has never opened one. The picture-only image card's split of the two facts and the provider's excerpt works that way whichever template renders it, because the excerpt arrives already fenced: contained rather than scrubbed, quoted whole, never shortened, never rewritten. A Fusion panel member that never got out is reported with a fixed, pipe-authored reason rather than a copy of its response, because a member's response can carry a storage URL the user was never shown, and a final-answer call that dies mid-stream keeps the text it had streamed and appends a short degrade marker naming that failure, which is stored with the reply rather than shown as a card — and a required internal file that cannot be read shows its own message, which says the file is no longer available and, since B936 (H3595-2), names no reference on any surface — not the card, not the operator's WARNING, not the session archive. That fatal path is now narrower than it was: a *tool's* picture reference whose path yields no readable id never reaches a storage read at all, so it no longer ends the turn. It is dropped from its block, the body is still dispatched, and the person is told in one `Images: skipped N (a picture naming … cannot be read from Open WebUI storage).` status — a different vocabulary, saying what was tried rather than claiming the file is gone. A reference whose id *does* read, and finds no record behind it, is unchanged: the pipe read the record, so the fatal card still stands. Their breaker accounting, unlike their reporting, does follow the text legs' arrival rule: a picture-only image or video generation whose response arrived whole and was not an OpenRouter document — a body a proxy, CDN or WAF rewrote, or one that is not a JSON object at all — never counts toward the breaker's limit, because the network path produced that body rather than OpenRouter; every other post-submit failure on those legs is charged exactly as it is here. Both media legs now hold that rule in code, so this sentence is a statement about them rather than about the text legs. Their **cards** still differ, deliberately. A picture-only image generation renders its rewritten body through `SERVICE_ERROR_TEMPLATE`, exactly as a chat does, so it gets the fence and the channel withholding for free. The video leg builds its own card and does not: `_extract_video_job_marker` and `_looks_like_final_video_content` parse the `[openrouter:v1:videojob:…]` markers and the literal `### Video generation failed` heading back out of the persisted transcript when a turn continues, and the template has no marker placeholder, so routing the video card through one would produce a card no reader could resume a job from. The video card therefore borrows only the fence — and withholds the excerpt on a channel chat itself, because it bypasses `channel_safe_values` and would otherwise make the withheld-on-a-channel claim false on the one leg with a channel audience.

**A non-streamed 200 that holds no output items is a failure the person can retry, not an empty turn, and it is not a connection fault.** With streaming off, OpenRouter can answer a `/responses` request with `200` and an `output` that is `[]` or `null`: the call connected, the model ran, and it produced nothing. Both shapes take the same path as a body with no `output` key at all — the full `TRANSIENT_RETRY_MAX_ATTEMPTS` budget, one strike on the breaker — and the retry can still bring a real answer. The card is then `OPENROUTER_ERROR_TEMPLATE`, because the connection-failure card would tell the person to check their firewall and their DNS about a reply that arrived; its reason reads that the model returned an empty answer (no output items) on `/responses`, and the person can simply send the message again. A body carrying at least one output item is unchanged, and a body with no `output` key and a body with no `choices` key are still the connection card above.

**The same is true on `/chat/completions`, and the test for it is the first choice rather than the body.** A non-streamed 200 there can carry a `choices` array whose first entry is an object and still give the reader nothing: an empty object, a null or empty `message`, content that is null, an empty string or an empty list, a list whose every part is blank (`[""]`, `["  "]`, `[{"type": "text", "text": "  "}]`, or a part that is neither text nor a typed block), and a bare `finish_reason: "stop"`. Whether the provider spelled that blankness as a string or as a list of parts makes no difference — both are the same missing answer, and both are read by the same reader that publishes the text, so neither gets a completed turn. Those take the same path — the full `TRANSIENT_RETRY_MAX_ATTEMPTS` budget, one strike on the breaker, no `response.completed` — and the card is `OPENROUTER_ERROR_TEMPLATE` for the same reason as on `/responses`, with a reason that names the first choice on `/chat/completions` as carrying nothing a reader could use. A choice that carries something is never re-billed: a tool call, a refusal, images, annotations, reasoning, or any `finish_reason` other than `stop`, because the provider is then deciding about the turn and another POST would bill it again. `stop` is the one that says nothing about whether anything was produced, which is why a choice carrying only that is treated as an empty answer. A content list is never refused for being a list: one non-text block in it — an image, in either the `image_url` or the `input_image` spelling — is an answer the person can be shown, as is a text part carrying text or a bare string carrying text, so only a list with nothing usable in it is re-sent. The four outer shapes — no `choices` key, an empty array, a non-list, a first entry that is not an object — keep the message and the connection card they have always produced: they ask whether the body is a chat completion at all, which is a different question from whether it carries an answer.

**A mangled whole body is not retried and does not count against the breaker.** "Whole" is what this says: the body arrived, and what came back is not an OpenRouter document — the `UpstreamBodyUnreadable` case, which the provider produced and a proxy, CDN or WAF may have rewritten. The provider answered, with a reply code the pipe accepted, so a re-POST asks it the same question and a rewritten reply is rewritten again; and the fault is in the network path rather than in OpenRouter, so counting it would punish a user for somebody else's firewall. The reason a caller is handed names the endpoint that answered and carries the upstream `Content-Type`. Those two are the load-bearing half of the diagnosis, and the chat card additionally quotes the first 200 characters of what the provider sent, which is where the proxy's own error page becomes visible. **Those first 200 characters are the provider's own text, so how they are rendered is a property of the surface, not one sentence for all of them.** On the card the excerpt arrives separately in `{body_excerpt}`, already inside a code fence, while `{reason}` carries only the two facts the pipe states itself. That split is deliberate: a body a proxy rewrote is the part of the reply an attacker chose, and left inline on a card its `![beacon](…)`, `[click me](…)`, `<script>` or unclosed ``` would render — a live image request, a live link, a live tag, and a fence that swallowed the rest of the card. The fence is not a scrub: the excerpt reaches the card verbatim, in full, and an operator who needs the whole payload reads it on the error object; the session log's copy of the provider's body is cut at 16,384 characters like every other payload-derived record. The picture-only image leg records the rewrite at ERROR with the excerpt in the message, so on that card the excerpt is on the card and in the log rather than on the card alone; the picture-only video leg writes no such record, so on that card the excerpt is on the card and nowhere else. The API error envelope quotes nothing: a chatless caller has no chat to write a diagnosis into, so its JSON body is composed by the pipe out of the two values the pipe itself derived, rather than transported from a body that never parsed into an `error` object at all. That composed, excerpt-free envelope is built on the **image** leg as well as the chat leg, for a chatless caller on that leg; and the image leg's breaker exemption — a rewritten body never charges the call — holds on the escaped arm too, because the arm that decides the charge is settled before the escape is built. A picture-only image generation and a video generation that fail on a fault the pipe owns rather than one the provider reported are the two places a chatless caller is told only the fault class: each gets the same standard envelope at 500, saying `Image generation failed.` or `Video generation failed.` respectively, and the full message and traceback are at ERROR in the session log. The chat caller on that same fault keeps its card, because an exception's `str()` is not a leak a person in the chat cannot already see in the same UI, and several committed tests assert on that card's text (see [Security & Encryption](security_and_encryption.md#log-safety)). A third case is the pipe-authored `ImageGenerationError` — a blank prompt, a caption the filter refused — which is delivered as its own sentence to every caller rather than composed into an envelope, because that sentence was written for a person and is what the model refused with. One sentence is therefore rendered on four deliveries — the chat card, a task route's card, the Anthropic Messages card, and a picture-only image generation's card, each as its `reason` — while a chatless, non-streamed caller's envelope carries the same sentence without the excerpt, as its `message`. The video leg is a fifth delivery with a shape of its own: same reason, its own heading and markers, the excerpt fenced beside it and withheld on a channel; it has two arms, the submit arm and the lifecycle that watches an accepted job, and **both** of them send that sentence without the excerpt to a chatless caller, at the same `502`. This holds on the streaming leg as well as the non-streaming one: a 200 whose body yielded **no decodable frame** because it carried no `data:` line at all is not retried and does not charge the breaker, and the connection card is not what a person sees. A body that carried at least one `data:` line and decoded none of them is a different fault — the provider did answer in SSE and then sent nothing readable — and stays a retried, counted `ClientPayloadError`, as does a body with no bytes at all.

**A body lost before it finished arriving is a different class, and it does count.** Here the connection was accepted and the body died on the way — behind a `200`, as a payload that stopped arriving, a server that hung up, an idle-read timeout, a reset socket or an expired total timeout, each of which reaches the pipe as a transport fault rather than as a document, and the same classification is shared by both non-streaming legs, which read their answer through one helper. OpenRouter has already run the call, so the request is **not** re-POSTed; but nothing was delivered, so it is one failed call and is counted once against the breaker, and it renders `CONNECTION_ERROR_TEMPLATE` like any other pre-answer connection fault. The two halves are deliberate: a body the provider produced and something else rewrote is nobody's bill but the reader's outage, while a body the transport lost is a real failure of a real call. **This is one rule on both endpoints**, and it is about the body rather than about what reached the reader: a `/chat/completions` body accepted and then lost costs one POST exactly as a `/responses` one does, whatever frames — visible or not — it had already published, and the fault crosses as `AcceptedResponseLostBody`, which every handler above the transport reads as the connection-class fault it has always seen. The rule is deliberately independent of the exception class, because the adapter raises its own `ClientPayloadError` for its answer-less shapes *from the same read*; a body that finished arriving and delivered nothing a reader could use is still re-POSTed on the full budget and keeps its own class.

### What retrying does not change

On the Agent tool's own API — `/api/v1/messages` — a card that survives its retries still reaches the agent as a `text_delta` followed by `stop_reason: "end_turn"`, because Open WebUI's converter for that endpoint has no error branch to map a card onto. The retry reduces how often such a card is the outcome; it does not change the shape of the one that is left. A caller relying on `stop_reason` to tell success from failure has to read the card's own text on that endpoint. That advice is unchanged by the streamed frame, and it applies to the streamed leg too — which is precisely why the frame is deliberately not sent on this endpoint: the converter would drop it as silently as it drops the HTTP status, leaving the agent an empty `end_turn` with no diagnosis at all, where the card at least carries one.

### C) API callers with no chat: HTTP error instead of a card

A card is written into a chat, for a person to read. A caller with no chat to write it into has nothing to show it in, and no way to tell a failure from a success — so on that leg the rejection leaves the pipe as an HTTP error rather than as Markdown. **One sentence, rendered once**, on the status line and in the body: the two carry the same number, so a caller that can only read one of them is not reading less than a caller that can read both. The HTTP escape is not extended to a *soft* refusal: a turn that answered anyway is a success, and its one remaining channel is the body.

| Condition | Result |
| --- | --- |
| No truthy `chat_id` **or** no truthy `message_id`, streamed or not, on any path except the two Anthropic Messages paths (`/api/v1/messages`, `/api/message`), and not an internal Fusion member | `StreamingResponse`, `status_code`: the status the pipe resolved off the provider, clamped to 400-599, and the same number as the body's `code`; `Content-Type: application/json` — that is the shape when the caller asked for a whole body. A caller that asked for a stream gets the same `error` object as a terminal in-band frame `{"error": {...}}` instead, because no HTTP status can travel on that leg at all |
| No truthy `chat_id` **or** no truthy `message_id`, streamed or not, on the chat leg **or the image leg**, a 200 whose body is **not a decodable JSON object** (a proxy's HTML error page, a truncated stream, `b"[1,2,3]"`), on any path except the two Anthropic Messages paths — the body is not a provider error, so it escapes the provider-error escape entirely and the envelope is built for it | `StreamingResponse`, `status_code: 502`, `Content-Type: application/json`, `code: 502` — or the same envelope as a terminal in-band frame when the caller asked for a stream; either way `code: 502`; `message` carries the endpoint that answered and the upstream `Content-Type`, and quotes none of the body — it is composed by the pipe, not transported from it |
| No truthy `chat_id` **or** no truthy `message_id`, `stream: false`, no task, on any path except the two Anthropic Messages paths, and the refusal is one of the five **local admissions** — a tripped circuit breaker, warmup failure, a missing request queue, a full queue (`Server busy (503)`, including the same refusal reaching a request already waiting for a permit when the pipe is superseded) or a pre-enqueue setup failure | `StreamingResponse`, `status_code: 429` for the breaker and `503` for the other four, `Content-Type: application/json` |
| The same, on the **image** leg: a provider rejection (`OpenRouterAPIError`), `stream: false`, chatless. The image leg's card renders the operator's own error template, which is prose written for a chat, so it answers the envelope instead — and the upstream status is what a caller can actually act on here | `StreamingResponse`, `status_code` and `code`: the same number, the status the pipe resolved off the provider, clamped to 400-599; `Content-Type: application/json`; `Retry-After` when the rejection carries one |
| No truthy `chat_id` **or** no truthy `message_id`, `stream: false`, on a **video** model, and the turn failed **before** OpenRouter answered the submission: a provider rejection | `StreamingResponse`, the status the pipe resolved off the provider, clamped to 400-599, and the same number as the body's `code`; `Content-Type: application/json`, `message` OpenRouter's own — the chat leg's escape, built in the video adapter because the orchestrator sees one opaque `str` and cannot tell a failure from a clip |
| The same, and the 200 from `/v1/videos` was **not a decodable JSON object** (a proxy's error page) | `StreamingResponse`, `status_code: 502`, `code: 502`, `message` naming the endpoint and the `Content-Type` and quoting no excerpt |
| The same, and the failure is one the **pipe owns** rather than one the provider reported | `StreamingResponse`, `status_code: 500`, `message: "Video generation failed."`, no Python class name anywhere in the body; the full message and traceback are at ERROR in the session log |
| The same, and OpenRouter had already **accepted** the submission but the lifecycle's status poll (`GET /videos/<job_id>`) came back not a decodable JSON object — the post-submit half of the `UpstreamBodyUnreadable` case. That arm returns a `VideoLifecycleResult` rather than a body, so the envelope is built by the consumer *after* `_settle_request`, which is why the breaker ledger and `error_occurred` move exactly as they did on the card | `StreamingResponse`, `status_code: 502`, `code: 502`, `message` naming the endpoint that answered and the upstream `Content-Type` and quoting no excerpt — the same sentence the submit arm sends |
| Any truthy `chat_id` **and** `message_id` | the card, unchanged |
| A refused attachment, no truthy `chat_id` **or** no truthy `message_id`, `stream: false` | `200`, unchanged — the answer, plus the same refusal sentence the status would have carried, joined after it under `choices[0].message.content`. The aggregated `Files:` / `Images:` line is the one that is folded in: a refusal that already travels on its own error card (`chat:message:error`) is excluded from that line on purpose and is not folded in as well, so one fact is still reported once |
| `stream: true` | the in-band frame, as the stream's terminal record: `{"error": {"message": …, "code": …}}`, and nothing else. This leg cannot carry an HTTP status at all — Open WebUI hands a pipe's non-2xx back inside a `StreamingResponse` that carries no `status_code`, so a status returned here would arrive as a bodyless `200` — and the escape is never built on a streamed turn in the first place, so there is no response to throw away. Three kinds of leg keep the card instead, and on each of them the frame would be lost rather than read: a **chat**, where a person is watching and a socket receiving `{"error": …}` renders an error box; an **internal Fusion member**, whose panel card is assembled by the collector, which ignores such a frame and would leave the member's own wording unreadable; and the two **Anthropic Messages mounts**, whose converter has no error branch at all and would drop the frame silently, arriving as an empty `end_turn`. The image and video legs are a fourth, and on them this is not merely a discarded value: the video leg's pipe-owned-fault arm is guarded on `not stream` precisely so the card is emitted *instead of* the envelope, and a streamed chatless video failure yields the card in its stream chunks rather than nothing at all; and the image leg, whose blanket arm used to be the counterexample this sentence was written to exclude, now carries the same `not stream` term. That arm built its envelope with the one envelope builder that has no `stream` term, so a streamed chatless failure discarded it and delivered only the terminal status — one empty delta and `[DONE]`, indistinguishable from a model that answered nothing. It now falls through to the card instead, and an envelope is never built there, because a `StreamingResponse` on a streamed turn is a value `pipe.py` cannot deliver. The three provider-shaped image arms were never affected: they emit rather than build an envelope, which is why the reach of that defect was occasional. A streamed turn is never left with an empty assistant message: an admission refusal that reaches a request already dequeued still ends the stream with the card, and the five pre-job refusals still leave as their own status. See [Security & Encryption](security_and_encryption.md#log-safety) |

The body is the error envelope, not the upstream payload:

```json
{"error": {"message": "<OpenRouter's own message>", "code": 503}}
```

For a mangled body `<the upstream message>` is the reason described above — the endpoint and the `Content-Type`, composed rather than transported — and not an upstream-authored sentence, because there is no upstream message to quote.

The status line and `code` are **the same number** — the status the pipe resolved off the provider's error type, clamped to 400-599 — so a caller reading either surface reads the same decision, and the error card above was selected from that same number. `code` is the **upstream status when the provider sent one**, and **`502` when the pipe could not read one** — a mangled whole body is the case that produces a `502`, because the status the pipe accepted (a `200`) is not a status the caller can act on, and the provider never said anything about the exchange. The message names the endpoint that answered, not the pipe, and on a mangled body it carries the same `Content-Type` the chat card shows — but not the excerpt, because that leg's body is composed rather than transported. One sentence, rendered once, delivered on the four template-rendered card legs; the video card carries the same sentence beside its own heading and markers.

**This row covers one failure class, and the rows above it do not widen it.** The envelope is built for `UpstreamBodyUnreadable` and for provider errors; the other failures the pipe absorbs into a card — connection, timeout, and its own internal errors — still reach a chatless caller as their card text inside a `200`, because the streaming loop catches them first and renders a card. The two media legs' pipe-owned-fault rows are the exception the image leg's 500 already was and the video leg's now matches: a fault the pipe itself owns reaches a chatless caller as the same envelope at `500` with a fixed sentence, because there is no upstream status to normalise and an exception's `str()` is a Python repr rather than a diagnosis. A caller branching on `status_code` sees the resolved status — a `502` for a mangled body, the provider's own status for a rejection — and a `200` for a timeout; that asymmetry is the loop's, and closing it would mean re-raising from the loop's own `except Exception`, which the task routes and `CancelledError` make unsafe. The streamed leg does not change that asymmetry, and it is not evidence that it is fixed: a caller branching on `status_code` there has no status to branch on at all, whatever the fault. What the streamed leg now carries is the in-band `{"error": ...}` frame for the two faults named above, and a timeout still arrives as its card inside a `200`.

**The status line carries the resolved status, and it is the same number the body's `code` carries.** That number is what the pipe decided the failure was after reading OpenRouter's error type, so it is occasionally not what OpenRouter sent on the wire: a `500` OpenRouter labels an authentication problem resolves to `401`, a `400` it labels a provider overload resolves to `503`. It is nevertheless the number to act on, because it is the one already published in the body, the one the error card was selected from, and the one the pipe's own retry classification reads — so a status line carrying anything else would make the two surfaces disagree about what happened. It is clamped to 400-599, because `_resolved_error_status` returns an unrecognised kind's wire status verbatim and a `700` is constructible but is not a status line. **Exact Open WebUI parity is not available here**: Open WebUI's own routers return the upstream status verbatim (`routers/openai.py:1694`, `:1736`) and its token-count proxy does the same, but it has no resolution step, and it does not have a second number to keep in step with the first. The chat and task routes pass a `StreamingResponse` escape through untouched (`main.py:1678` tests `isinstance(response, JSONResponse)`; `utils/middleware.py:6720-6733`; `routers/tasks.py:202`), so the status on the line is the status the caller receives, and a 5xx on the wire stays a 5xx all the way out instead of becoming an Open WebUI-side exception. The five local admissions still answer with their own status, and there are two statuses between them: a refusal the pipe raised itself has no upstream status to resolve. The split is the cause's — 429 for the per-user breaker, 503 for the four conditions of the process. **One behaviour change a caller will see: a caller with a blanket retry-on-5xx policy now retries a provider 5xx it previously saw as a 400.** That is correct — the pipe already retried internally per `TRANSIENT_RETRY_MAX_ATTEMPTS` and already sends `Retry-After` — but it is a change. OpenRouter's own message survives verbatim into `message`, so nothing of *OpenRouter's* account of the failure is lost.

The body is the uniform envelope for every case, including a 5xx that carries a provider's own `metadata.raw`. That object has no `code` field, so emitting it verbatim would drop the one signal the normalisation exists to preserve.

**`message` is composed from the top level of OpenRouter's own `error` object only.** A provider's own text, which OpenRouter nests under `error.metadata.raw` (its reference calls that field "the provider's verbatim payload"), is not promoted into it at any nesting depth; it reaches the chat card and the operator's WARNING instead, which is where the diagnosis lives. An integration that branched on the provider's specific wording in `error.message` now sees OpenRouter's, and when `metadata.raw` merely mirrors `error.message` — the shape OpenRouter sends when it has no provider text of its own — nothing changes at all. `error.code` and the `Retry-After` header are untouched.

**The gate is truthiness, on two keys.** `main.py:1243` builds `chat_id = form_data.pop('chat_id', None) or ''` for a caller with no chat, so the keys are *present and empty*; `utils/middleware.py:3286` tests truthiness for the same reason. A third key would be stricter than the idiom it copies, and a chat turn that races the socket (`session_id` is `undefined` until the socket connects) would lose its card. `__event_emitter__` is attached for both kinds of caller (`main.py:1277-1306` sets all three keys; `functions.py:239` tests key presence), so it is not a discriminator.

**Open WebUI's own task routes reach this gate**, and that is the point: they send a `chat_id` and no `message_id`, and Open WebUI's own models surface a provider failure on those routes as a non-2xx — `routers/openai.py:1736` returns `JSONResponse(status_code=r.status)`, and a `StreamingResponse` a task route returns passes through its own handler untouched. Making the pipe agree with its own models is what the truthiness gate buys. Of the eight task routes, seven are consumed by the pipe's task-model adapter, which answers 200 with a body that depends on the task kind — a contextual card for a kind Open WebUI persists, `""` for every other; **Mixture of Agents is the one that reaches the gate** (which excludes exactly one task name, `moa_response_generation`), and it gets the same non-2xx its host's own models get — a failure there reaches the browser console, not a chat. **Not for `context_compaction`, `context_summary` or `memory_review`**, though those three meet the gate too: the task adapter catches the fault itself, retries, and returns `""` before the request loop is ever reached, so no envelope is built and nothing reaches a console. Those three are visible only in the backend log, as `Task model attempt N/M failed` — a bounded excerpt of the provider's text rather than all of it, cut at 8,192 characters with a marker naming how much was removed — see [Task models and housekeeping](task_models_and_housekeeping.md#how-the-pipe-detects-a-task-request).

**`/api/v1/messages` is excluded**, for every failure kind the escape covers, and on the streamed leg as well as the non-streamed one. `main.py:2068-2077` re-wraps *any* `StreamingResponse` through `openai_stream_to_anthropic_stream`, which skips every line that is not SSE `data:` and every payload without `choices` — it has no error branch at all, which is measurable rather than inferred: a payload of `{"error": {...}}` has no `choices`, so the converter skips it and the frame is dropped without a trace. An HTTP error would arrive as an empty `end_turn` with the error text gone, and so would the in-band frame, which is why the streamed row above excludes these mounts rather than sending the frame there for uniformity. A mangled body on that endpoint keeps its card for the same reason a provider rejection does.

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

`session_id` is always one of them, and on a temporary chat it is supplied as the empty
string, so a line guarded with `{{#if session_id}}` is left out there rather than
rendering an empty field.

It then merges in per-error variables (for example `status_code`, `reason`, `timeout_seconds`, `error_type`).

### Variables for OpenRouter “request rejected” templates
The OpenRouter error formatter supports a larger set of optional values, including:
- `heading`, `detail`, `sanitized_detail`
- `openrouter_code`, `openrouter_message`
- `openrouter_error_type` — the typed kind OpenRouter reported for this rejection (for
  example `invalid_request_error` or `content_policy_violation`). This is not the
  `error_type` named above: that one is the Python exception class the pipe caught, and
  the two are different vocabularies for different cards.
- `upstream_type`, `upstream_message` — `upstream_type` is the provider's own error type on **both** legs: the name inside `error.metadata.raw.error.type` or inside the failure's own `error.type` wins, so the same body names the same category however it was delivered. A failure reported inside a started reply whose body names none falls back to the numeric `code`, then to the stream marker. It is withheld on a channel chat, as `upstream_message` beside it is, because a provider is free to put whatever it likes in that name
- `provider`, `requested_model`, `api_model_id`, `normalized_model_id`
- `retry_after_seconds`, `rate_limit_type`
- `include_model_limits`, `context_limit_tokens`, `max_output_tokens`
- `metadata_json`, `provider_raw_json` — every value derived from the provider's payload is cut at 16,384 characters, with a marker naming how many characters were removed: the two JSON values `metadata_json` and `provider_raw_json`, the five inline copies of the provider's message (`detail`, `sanitized_detail`, `reason`, `upstream_message`, `openrouter_message`) and the joined `moderation_reasons` list. A cut JSON value is no longer parseable JSON; a cut inline value is still one logical line, its marker space-joined onto the card's line. `raw_body` and `flagged_excerpt` are not cut and arrive whole
- `body_excerpt` — the provider's own first 200 characters when an accepted response's body was not decodable at all, arriving inside a code fence. Filled on that fault only, never cut, and never scrubbed: a proxy's own words are the evidence. Its sources are the chat, `/responses`, task and Anthropic routes and a picture-only image generation's `/images` call; the video leg produces the same fenced excerpt but builds its card itself rather than through a template, and on the video lifecycle leg it reaches no chatless caller's body at all — that arm's envelope quotes none of it, on both the fresh submit and the join. Pair it with a `reason` that names the endpoint and the `Content-Type`, and do not put the excerpt in `reason` as well. It is withheld on a channel chat, so a room's card is the diagnosis that names no one
- `error_chunk_id`, `error_chunk_created`, `is_streaming_error`, `native_finish_reason`, `request_id_reference` — `error_chunk_created` is the provider's own chunk clock: a Unix-second epoch is rendered as a Z-suffixed UTC ISO-8601 instant, in the same form and zone as the `Time` row beside it, and any other value reaches the card as the provider's own text
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
- removing the legacy `include_reasoning` flag when any id in `models` fails to list it — and, where the dropped value was `false` and the primary lists `reasoning`, carrying thinking off by merging the row's own effort into the `reasoning` object the request already carries, so its `max_tokens` and `summary` survive instead; on a request with no `model_fallback` the flag is simply turned off

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

A value that arrives already fenced must not be fenced again by the template: `metadata_json` and
`body_excerpt` carry their
own fence, so a template that adds a ```json … ``` pair around it closes that fence early and the JSON
escapes the block.

You do not have to edit a stored template that already fences one of these values. The renderer tracks
the template's own fences, so a row written as ` ```json ` / `{metadata_json}` / ` ``` ` — the shape
shipped before 2.7.4, and the shape the pipe's own example used to show — renders as exactly one
code block holding the payload, with the label above it and the card text after it.
A fence written after a blockquote marker stays inside it, whichever way it is written — the
one-line spelling `> ```{raw_body}``` ` and the multi-line spelling with the value alone between
`> ``` ` and `> ``` ` render the same block. The renderer keeps the `>`
you wrote on every line it emits from that row, so the block is a block *inside* the quote, and a
`>` on that line with nothing after it is the marker rather than a label — which is why a quoted row
never renders as an empty quote above a code block that is not in it. A template that
does not fence the value is unaffected. The same holds for `{raw_body}`, `{flagged_excerpt}`,
`{provider_raw_json}` and `{body_excerpt}`; a value is only unwrapped when the template supplies the fence. A value that arrives cut ends mid-payload and its `...(truncated: N characters omitted)...` marker sits on its own line inside the fence, so read a cut value as the start of the payload rather than the whole of it. An inline cut value is the other shape: its marker is space-joined onto the card's line rather than given a line of its own, because an inline value is always one logical line, so the same marker text appears mid-line there.

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
3. Use `session_id` and `user_id` (when available) to correlate with other telemetry (Redis cost snapshots, session log archives, etc.). On a channel those two are withheld from the card the reader sees, so the `error_id` is the handle to start from there; the ids are still in the operator's log line, which the card's audience never sees. On a temporary chat `session_id` is withheld from the card **and** from the operator's log line, because it is the browser's own socket id and a live session handle, so there `error_id` is the only handle and there is nowhere else to look; `user_id` is still on the record.

### Reading a WARNING that repeats

Three lines warn once per episode and repeat at `DEBUG` on a 300-second cooldown, so an
outage does not fill the log with copies of itself: the per-user database breaker's own
refusal lines (`DB writes disabled`, `DB reads disabled`, `DB deletes disabled`, each
keyed on the verb and the user so one user's outage cannot silence another's first
report). Every occurrence of each is logged; the repeats are at `DEBUG` rather than at
`WARNING`, so raise the level to see them. The first of each is always at `WARNING` —
that line is the only evidence the condition is active — so **one such line does not
mean the condition has cleared**: check for a later `DEBUG` with the same text before
concluding the database came back. Not every repeating warning uses that 300-second
window: the session log assembler's per-turn faults, and now the four store reads on its
pass, warn once per cause and repeat at `DEBUG` on an hour's cooldown, so a line from
that family is worth an hour of silence before you conclude anything from its absence.
The first line of each still arrives, so a still-active fault is never invisible.

The filter-installer's `OpenRouter <family> filter ensure failed` lines for Web Tools,
Image Gen, Fusion and Direct Uploads keep the same contract without the cooldown: a
catalogue pass that keeps failing would otherwise write four traceback-bearing WARNINGs
an hour, per worker, for as long as a locked database stays locked. Each is a `WARNING`
the first time that worker sees that family fail with that exception class, and a
`DEBUG` — with the same message and the same traceback — on every pass after it.

What is *not* throttled on these paths: the person's own notice, which is emitted on
every refusal without exception, and the session log's fallback line, `Session log DB
staging returned no staged segment; falling back to a queued zip write`. The read-side
notices named above are the one exception on this list: each fires once per degraded
episode rather than once per refusal, and re-arms on the first read that reaches the
success arm, because the loader is called once per artifact group per turn and Open
WebUI neither dedupes nor rate-limits a notification frame. The write-side breaker
notice is still emitted on every refusal. That one is not
on a cooldown at all — every staging fault warns at `WARNING` — because a single turn
stages more than once (a tool loop or a hand-back stages the same turn again) and the
count of those lines is the only evidence a turn lost more than one segment. Each names
the `chat_id`, `message_id` and `request_id` it is about, so several faults in one turn
are told apart rather than merged into one. The zip write it announces runs on every one
of them: however many lines the outage has produced, each degraded segment is still
enqueued and still written to its own archive.

See also: [Session Log Storage](session_log_storage.md) and [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md).

---

## Testing

This repository includes tests for template behavior and error rendering. Prefer running the specific test modules first, then the full suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_error_handling.py tests/test_template_valve_restore.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```
