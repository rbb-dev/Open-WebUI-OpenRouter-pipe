# History Reconstruction & Context Replay

Open WebUI stores chat history as a list of heterogeneous message objects. OpenRouter’s Responses API expects a structured `input` array containing messages and (optionally) structured tool/reasoning artifacts.

This document describes how the pipe builds that `input` array, how it replays persisted artifacts referenced by hidden ULID markers, and how retention/pruning valves affect what is sent upstream.

> **Quick navigation:** [Docs Home](README.md) · [Persistence](persistence_encryption_and_storage.md) · [Multimodal](multimodal_ingestion_pipeline.md) · [Valves](valves_and_configuration_atlas.md)

---

## 1. Entry point: `transform_messages_to_input`

The core history conversion is implemented by `transform_messages_to_input(...)`.

Inputs (high level):
- `messages`: Open WebUI-style messages (each with `role` and `content`).
- Optional context for artifact replay:
  - `chat_id`
  - `openwebui_model_id`
  - `artifact_loader(chat_id, message_id, ulids)` (async)
- Retention/pruning:
  - `pruning_turns` (from `TOOL_OUTPUT_RETENTION_TURNS`)
  - `replayed_reasoning_refs` (for `PERSIST_REASONING_TOKENS="next_reply"` cleanup)
- Runtime context for multimodal conversion:
  - `user_obj`, `event_emitter`
  - `valves` (or defaults to `self.valves`)

Output:
- A list of input items (messages plus any replayed artifacts) suitable for OpenRouter’s Responses API.

---

## 2. System and developer messages

Messages with `role` of `system` or `developer` are preserved as separate message items:

- Content is converted into `input_text` blocks without merging or whitespace normalization.
- The pipe emits them as:

```json
{
  "type": "message",
  "role": "system",
  "content": [{ "type": "input_text", "text": "..." }]
}
```

---

## 3. User messages (content blocks → `input_*`)

User messages are converted into a single `type: "message"` item with a `content` list. The pipe transforms certain known block types; unknown block types are left unchanged.

### 3.1 Text
Open WebUI may provide user content as a string or as block objects. Text is normalized into:
- `{"type":"input_text","text":"..."}`

### 3.2 Images (vision gating + storage)
Image handling is described in detail in [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md). Key behaviors relevant to history reconstruction:

- Vision gating: if the target model is not vision-capable, the person's attachments and reused images are skipped and the pipe emits a status message saying so. Pictures a tool returns still go to the model, whatever it accepts, as in Open WebUI's own tool loop (section 5.4).
- Image forwarding policy:
  - `MAX_INPUT_IMAGES_PER_REQUEST` caps the images one of the person's messages forwards, its own or a reused one;
    pictures a tool returns are never capped.
  - `IMAGE_INPUT_SELECTION` controls fallback behavior:
    - `user_turn_only`: only user-attached images are forwarded.
    - `user_then_assistant`: if the user turn has no images, the pipe may reuse the most recent image already in the conversation - an assistant image extracted from Markdown image syntax, or one the user attached on an earlier turn - bounded by `IMAGE_REUSE_MAX_TURNS`. An image returned by a tool is never reused this way: it belongs to its tool round (section 5.4), and a tool round ends the window for pictures from before it — once a tool has run, nothing older is reused either, whether or not that round returned a picture. A round that asked you a question is not a media round and does not end the window, and neither does a picture the model shows you in its own reply to a round.
- An image the person attached is never written to Open WebUI storage, in any chat and by any request of a turn. A `data:` URL within `BASE64_MAX_SIZE_MB` is sent as it came apart from the scheme, which is lower-cased to `data:`, and one over the limit is not sent, the picture being skipped and the person told in a status on their latest message; a remote image is downloaded and its bytes sent inline, or its link is forwarded when it cannot be downloaded **unless the pipe's own address check refused the host**: a host that does not resolve, or resolves to a non-routable address, fails closed and is not sent, and the person sees `Images: skipped N (could not be fetched, so it was not sent).`; an Open WebUI file URL is read with the requester's access and inlined, so providers never need to fetch from your Open WebUI host. An image reused from an earlier turn is inlined as a `data:` URL, under a media type the pipe resolves from the bytes. Images the model generates are stored by the output path.

### 3.3 Files, audio, and video
The pipe includes transformer functions for:
- `input_file` / `file` → `input_file`
- `input_audio` / `audio` → `input_audio`
- `video_url` / `video` → `video_url`

The security/size/SSRF rules (including HTTPS-only defaults) are documented in [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md).

---

## 4. Assistant messages (plain text vs marker-based replay)

Assistant turns are handled in two modes:

### 4.1 Plain assistant messages (no markers)
If the assistant text does not contain any embedded markers, the pipe emits:

```json
{
  "type": "message",
  "role": "assistant",
  "content": [{ "type": "output_text", "text": "..." }]
}
```

### 4.2 Marker-based replay (ULID markers)
If the assistant text contains embedded marker lines, the pipe splits the text into:
- text segments (emitted as assistant `output_text` messages), and
- marker segments (used to replay persisted artifacts).

When a marker split produces several `output_text` items for one assistant message, `annotations` and `reasoning_details` go on the **last** of them, once each, and on no other item.

Marker detection and splitting is performed by helper functions (for example `contains_marker(...)` and `split_text_by_markers(...)`) and uses the marker format:

```text
[<20-char-ulid>]: #
```

Three families of line count as hidden marker lines, all of them valid CommonMark reference definitions and so
invisible in the browser:

```text
[<20-char-ulid>]: #                 the artifact reference — a replay marker
[P:<phase>]: #                       the phase label
[openrouter:v1:<kind>:<body>]: #     the kind-marker transport line — NOT a replay marker
```

Only the first is a **replay marker**: it is what a marker segment resolves, and what the loader is asked for. The
kind family carries the pipe's own transport state (video intent and relay disclosure blocks) and is *not* replayed
and *not* an artifact reference. A kind line is recognised as a hidden line, dropped from the text segments, and
never reaches the provider as assistant text; a message whose only markers are kind markers takes the marker branch
here and produces no `missing_artifact_markers` warning.

Tool **results** are stripped of hidden marker lines before the model sees them; tool **arguments** deliberately are
not, because a JSON argument value is not a line of the message and a caller may legitimately pass text that looks
like a marker — stripping it would corrupt the payload it means to send.

The strip is a **line** filter, so it removes a line by its whole content rather than by the marker inside it. A
marker-shaped line a user typed **on a line of its own** — in any of the three families above — is removed too, while
the same text inline is preserved. Only whole lines are affected; a marker that is part of a sentence is left alone.

For each marker segment:
- the pipe looks up the referenced persisted artifact payload (via `artifact_loader` when available),
- normalizes it to the schema expected by upstream (`normalize_persisted_item`),
- and appends it directly into the `input` array as a structured item.

**Artifact loader preconditions (important):**
- The pipe only attempts to load artifacts when all are present:
  - `artifact_loader`
  - `chat_id`
  - `openwebui_model_id`
  - at least one marker in the message

If any of these are missing, marker segments will not be replayed.

---

## 5. Replay filtering and pruning

### 5.1 Non-replayable tool artifact types
Some tool artifacts are intentionally never replayed back to the provider (to avoid wasting context window and to reduce provider-side errors). The pipe filters these by type during history reconstruction.

### 5.2 Orphaned function call pairs
When tool calls are persisted, the pipe attempts to keep tool call/request and tool output/response pairs consistent.

During replay, the pipe classifies persisted function call artifacts and may drop:
- `function_call` items with no matching output
- `function_call_output` items with no matching call

This prevents sending half of a tool interaction back to the model. A call that was still running when the user pressed Stop is exactly that: a call with no result. Dropping it is expected and is logged only at debug level.

### 5.3 Tool output pruning by turn age (`TOOL_OUTPUT_RETENTION_TURNS`)
When `TOOL_OUTPUT_RETENTION_TURNS` is set, the pipe computes turn indices across the conversation and treats messages older than the retention window as “old”.

For old turns, it can prune very large `function_call_output.output` strings by:
- preserving a head and tail,
- inserting a note indicating the output was pruned,
- and leaving markers intact.

When an output carries a picture, only its text is shortened; the picture stays.

This keeps replay payloads smaller while preserving recency and high-level context.

### 5.4 The pipe's own copy of each tool round

Every tool round the pipe runs itself, and every round of an OpenRouter server tool, is also stored by the pipe: the
call and its output as a pair, behind a hidden marker placed in the answer where the round happened. Four exceptions:
with results kept, a server tool whose own item OpenRouter takes back unchanged -- the advisor, the subagent, model
search -- is stored as that item instead, while every other server tool, including one the pipe does not know, keeps
the pair; image generation, whose picture is already part of the answer, is not stored; a
and a request that belongs to no chat message -- a call that carries no `chat_id`, the plain API route --
keeps its copy in memory for the length of that request only, keyed on the request id, and never writes it to the
database; the request ends and it is gone, and no marker line is added to the caller's response, so such a call's
records last for the request, not the conversation. A temporary chat
keeps nothing, so with tool cards off its rounds reach no later request. (In Open-WebUI tool mode the rounds of a
temporary chat's reply are held in memory until that reply ends or the provider refuses a call-back the pipe was waiting for, so Open WebUI's calls back after each round of tool
calls still hand them to the model; see [Persistence](persistence_encryption_and_storage.md).) The calls are written when
the round starts and each result when its call returns, and the round comes back to the model in call order on that turn. So in a streamed
reply, Stop keeps a round's calls before the first one still running. A call refused before it ran is answered as soon as it
is refused, and on the turn it is answered it is handed to the model in the order the round asked for it, because its
answer is produced before any queued call has run. A round stored with tool cards on is written as the calls are answered, so a
Continue replays a refused call's answer ahead of the results that were still running. A call the model sends malformed is answered as Open WebUI's own tool loop answers it on the same route and is kept like any other call; several argument objects sent back to back become one call each.

A call can also be kept in this copy and still be withheld from one request: when the replay budget is applied to that
turn, the result is replaced in `input` with a model-visible stub, and the tools whose results were replaced are named to
the person in a warning notification — once per call id, on every path that applies the budget except the task-model
adapter, which never sees the tool round. The copy above is untouched, so a later turn that fits still replays the full
result.

The copy carries the call's arguments and its full result, pictures included, whatever `PERSIST_TOOL_RESULTS`
says, as a shown card does in the message Open WebUI saves; that setting decides what later turns receive, not
whether the round is stored (see below). A call that
did not complete keeps its failure text; where that text alone would not read as a failure, it opens with
`Error: the tool call did not complete.` With no `ARTIFACT_ENCRYPTION_KEY` set, or with `ENCRYPT_ALL` off, the copy
is stored as plain JSON.

On replay each round reaches the model exactly once, except in a temporary chat, where a later turn gets a round
only through the card Open WebUI keeps for it in the browser, and none with cards off:

- Open WebUI hands back the rounds saved in the message -- the shown cards of a reply -- as ordinary tool messages.
  Where it has handed back a call, the pipe's copy of that call is dropped.
- A copy the pipe wrote for a round Open WebUI was never given -- with tool cards off -- is marked as such and always
  kept. Call ids alone cannot decide this: chats saved before the pipe made the ids it invents unique hold repeated
  ones, because the chat-completions route used to number calls without an id per request. This mark is what keeps the
  rule exact.
- After Stop in a streamed reply, Open WebUI saves as finished the cards of the calls before the first one still
  running (the pipe fills a round's cards in call order) and marks that call and every later call unfinished; its
  converter hands back only the finished calls, and the pipe's copy of them is dropped as a duplicate. With cards
  off, or where no card was saved, the pipe's copy carries the round's calls before the first one still running.
- Where the pipe replays OpenRouter's own item for a server tool unchanged (the advisor, the subagent or model
  search, with results kept), that item
  wins over the card pair Open WebUI saved for it.
- An earlier turn's results are withheld by the same rule whichever copy carries them: with `PERSIST_TOOL_RESULTS` off
  the model gets `{}` in place of the arguments and a placeholder result -- `[tool result not retained]`, or
  `[tool call failed; result not retained]` when the call did not complete. An `ask_user` round is the exception:
  its question and the person's typed answer are always handed over, since the answer is the person's own words.
  The stored row is left alone, so turning the setting back on hands the full result over again.
- An image returned by a tool comes back as a separate message right after the round's results ("Here are the
  images from the tool results above"): Open WebUI builds it from its own record, and the pipe builds the same
  message when its own copy carries the round, so the request is the same whatever the card switch says. It is
  part of that round's result: handed over in full where it sits, whatever the attachment limit, even when it is
  not the last message; withheld with the round on an earlier turn while results are not kept; never stored again;
  unlike an image the person attached, never reused on a later question; and it ends the reuse of any older
  picture.

The copy does not depend on reasoning. Tool rounds were accepted without the reasoning around them when measured with
Claude Opus 4.8 on `/responses` and `/chat/completions`, so the copy stays when reasoning is dropped from a request and outlives reasoning
cleanup. It is never published as an output item, so Open WebUI neither draws it nor runs it again.

---

---

## 6. Reasoning replay and `PERSIST_REASONING_TOKENS`

When replayed artifacts include reasoning items, the pipe can optionally record references in `replayed_reasoning_refs` so the caller can delete those artifacts after replay when reasoning retention is limited to a single turn. Under `next_reply`, the cleanup runs only on a generation that finished the reply: it must not have been cancelled or errored, and it must not have handed its tool calls back for another request to answer. The two ways a reply is not finished are both guarded by the same clause. A Continue keeps the rows of the message that request is still writing, so continuing an answer does not delete the reasoning of the generation it continues; a hand-back keeps them, because the request that comes back to answer the tool results is the one that will need them, and when that request arrives it is the one that deletes them; if it never arrives -- the user stops, or the sender errors -- the rows wait for the housekeeping sweep, because no generation finished the reply and `next_reply` has nothing to bind to. The tool-round copies of §5.4 are not reasoning and are not deleted with it.

System default is `PERSIST_REASONING_TOKENS="conversation"`; see [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the exact semantics and defaults.

A reasoning row keeps the `signature`, `format` and `encrypted_content` the provider sent, on every `/chat/completions` route, streamed or not: an unsigned row is one the next turn cannot replay, and on a later Anthropic turn that costs the whole span rather than the one block.

### 6.1 Where a replayed reasoning item goes

A model produces reasoning at a particular moment: before a tool call, after its result, or between two
sentences of an answer. Open WebUI stores the answer as text, so that position is lost unless the pipe records
it. Each persisted reasoning item therefore carries an anchor - which call it preceded or followed, or which
assistant message it sat before - and replay puts it back in that place rather than appending it. An ordinal at
or beyond the turn's message count places the block after the last message, in that turn: a continued turn whose
carried turn ends in reasoning produces exactly that one-past ordinal.

Anchors are **scoped to a turn**, where a turn is the region between user messages. Tool `call_id` values are
not guaranteed unique across a conversation: in chats saved before the pipe made the ids it invents unique, the
same id can appear in several turns, because the chat-completions adapters used to number calls per request.
Binding an anchor only within its own turn is what keeps a reasoning item from
attaching itself to an unrelated call with the same id. Open WebUI's own "Here are the images from the tool
results above" message, which follows a round's results and carries that round's pictures, is not a user message
for this purpose: it stays inside its round's turn on both the generating and the replaying side, and a thought
the model had after the round is replayed after it.

On `/chat/completions` a replayed reasoning item rides on the assistant message that carries the tool calls, as
`reasoning_details`, matching Open WebUI's `convert_output_to_messages(raw=True)`.

The anchors are internal ordering metadata, and they are stripped from every item the pipe sends — including a
turn-opener, not only reasoning — so none of the five keys in `REASONING_ANCHOR_KEYS` is ever part of a
request.

### 6.2 An answer continued across more than one request

"Continue response" adds a second generation to the same assistant message, and Open WebUI's own tool loop can
call the pipe several times within one turn. Ordinals are counted per request, so without care the continuation
would number its first call `0` again and its reasoning would bind to the first generation's call - placing two
thinking blocks side by side, the shape providers reject.

The pipe therefore offsets a continuation's ordinals by what the turn already contains: the calls and the
assistant messages the request carries before this generation starts. A turn that ends on reasoning gets one
further step, so the continuation's first block is placed after its own text rather than beside the block that
ended the previous generation. The offsets apply whether or not the Continue streams, since Open WebUI keeps the
earlier generation on a non-streamed Continue as it does on a streamed one. The two `/chat/completions` transports agree on that anchor whichever one carried the
generation: the non-streamed one flushes each round's reasoning before its tool calls, exactly as the
streamed one does, so a thought the model produced before a call is replayed before it on both.

The continuation's first output also has to bring its own line break. Open WebUI joins that output onto the
stored reply's last line. A hidden marker line that gains text after it is no longer a marker: its row stops
replaying, and the line either shows its id as text or, when the added text contains no space, hides the added text
along with the id. So when the stored reply ends on a marker line, the continuation's first model output starts on a
new line, whether it is text, a marker line or a generated picture. When the stored reply ends on ordinary text
instead, model text and a generated picture carry on the sentence, while a hidden marker line that comes first
starts a new paragraph, so it stays a line of its own. A card
or a refusal that opens a continuation - the provider's error card when a model refuses prefill, the breaker's
refusal, a busy or startup notice, the card saying the API key is missing, or the notice that the model catalog
could not load when the Continue is not streamed - starts a block of its own on any Continue; on a streamed Continue
the catalog notice shows as Open WebUI's error instead, and the stored reply stays as it was. A reply written by
internal Fusion is not continued: a Continue on one leaves the reply as it was, and a notification says that Continue
is not available for Fusion replies and that regenerating runs Fusion again.

---

## 7. Failure modes (what happens when artifacts are missing)

- If the artifact loader fails (DB errors, network issues), the pipe logs a warning and continues without replaying artifacts for that assistant message.
- If an individual marker cannot be resolved to a payload (for example after key rotation or cleanup), the pipe logs a warning and skips that artifact. In a temporary chat a later turn's markers resolve to nothing by design, since the pipe keeps none of its rows, and are logged only at debug level. A call that carries no `chat_id` never gets markers at all, so there is nothing for it to resolve.

Operational implications:
- Conversations may still render in the UI, but upstream requests may lack some historical tool/reasoning context.
- If you rely on long-lived replayability, validate your retention and key rotation procedures in [Persistence, Encryption & Storage](persistence_encryption_and_storage.md).
