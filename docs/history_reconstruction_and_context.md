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
  - `__request__`, `user_obj`, `event_emitter`
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

- Vision gating: if the target model is not vision-capable, image blocks are skipped and the pipe emits a status message indicating attachments were ignored.
- Image forwarding policy:
  - `MAX_INPUT_IMAGES_PER_REQUEST` caps images forwarded per request.
  - `IMAGE_INPUT_SELECTION` controls fallback behavior:
    - `user_turn_only`: only user-attached images are forwarded.
    - `user_then_assistant`: if the user turn has no images, the pipe may reuse the most recent image already in the conversation - an assistant image extracted from Markdown image syntax, or one the user attached on an earlier turn - bounded by `IMAGE_REUSE_MAX_TURNS`. An image returned by a tool is never reused this way: it belongs to its tool round (section 5.4).
- Images attached to the current turn are re-hosted into Open WebUI storage when a storage context is available; an image reused from an earlier turn is inlined as a `data:` URL instead, under a media type the pipe resolves from the bytes. Where a storage context resolves, the block sent upstream carries the bytes, so providers never need to fetch from your Open WebUI host; a request made without one - API automation, for instance - keeps the original payload. Re-hosting on this path covers only what the user attached; images the model generates are stored by the output path.

### 3.3 Files, audio, and video
The pipe includes transformer functions for:
- `input_file` / `file` → `input_file`
- `input_audio` / `audio` → `input_audio`
- `video_url` / `video` → `video_url`

The security/size/SSRF rules (including HTTPS-only defaults) and re-hosting behaviors are documented in [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md).

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

Marker detection and splitting is performed by helper functions (for example `contains_marker(...)` and `split_text_by_markers(...)`) and uses the marker format:

```text
[<20-char-ulid>]: #
```

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
call and its output as a pair, behind a hidden marker placed in the answer where the round happened. Three exceptions:
with results kept, a server tool whose own item OpenRouter takes back unchanged -- the advisor, the subagent -- is
stored as that item instead; image generation, whose picture is already part of the answer, is not stored; and a
request that belongs to no chat message, such as a direct API call, stores no copy at all. The calls are written when
the round starts and each result when its call returns, in call order. So in a streamed reply, Stop keeps a round's
calls before the first one still running. A call refused before it ran, such as one naming an unknown tool, is answered
at once and kept wherever it sits. A call the model sends without a name or without arguments is answered only after
the round and is not kept.

The copy carries the call's arguments and its full result, pictures included, whatever `PERSIST_TOOL_RESULTS`
says, as a shown card does in the message Open WebUI saves; that setting decides what later turns receive, not
whether the round is stored (see below). A call that
did not complete keeps its failure text; where that text alone would not read as a failure, it opens with
`Error: the tool call did not complete.` With no `ARTIFACT_ENCRYPTION_KEY` set, or with `ENCRYPT_ALL` off, the copy
is stored as plain JSON.

On replay each round reaches the model exactly once:

- Open WebUI hands back the rounds saved in the message -- the shown cards of a reply -- as ordinary tool messages.
  Where it has handed back a call, the pipe's copy of that call is dropped.
- A copy the pipe wrote for a round Open WebUI was never given -- with tool cards off -- is marked as such and always
  kept. Call ids alone cannot decide this: chats saved before the pipe made the ids it invents unique hold repeated
  ones, because the chat-completions route used to number calls without an id per request. This mark is what keeps the
  rule exact.
- A round Open WebUI saved unfinished, because the user pressed Stop, is not handed back by Open WebUI's converter, so
  the pipe's copy carries the round's calls before the first one still running.
- Where the pipe replays OpenRouter's own item for a server tool unchanged (the advisor, with results kept), that item
  wins over the card pair Open WebUI saved for it.
- An earlier turn's results are withheld by the same rule whichever copy carries them: with `PERSIST_TOOL_RESULTS` off
  the model gets `{}` in place of the arguments and a placeholder result -- `[tool result not retained]`, or
  `[tool call failed; result not retained]` when the call did not complete. An `ask_user` round is the exception:
  its question and the person's typed answer are always handed over, since the answer is the person's own words.
  The stored row is left alone, so turning the setting back on hands the full result over again.
- An image returned by a tool comes back from Open WebUI as a separate message right after the round's results
  ("Here are the images from the tool results above"). It is part of that round's result: handed over where it
  sits, even when it is not the last message; withheld with the round on an earlier turn while results are not
  kept; and never stored again, or reused on a later question, the way an image the person attached is.

The copy does not depend on reasoning. Tool rounds were accepted without the reasoning around them when measured with
Claude Opus 4.8 on `/responses` and `/chat/completions`, so the copy stays when reasoning is dropped from a request and outlives reasoning
cleanup. It is never published as an output item, so Open WebUI neither draws it nor runs it again.

---

---

## 6. Reasoning replay and `PERSIST_REASONING_TOKENS`

When replayed artifacts include reasoning items, the pipe can optionally record references in `replayed_reasoning_refs` so the caller can delete those artifacts after replay when reasoning retention is limited to a single turn. Under `next_reply`, a Continue is the case to watch: the cleanup that runs at the end of a request keeps the rows of the message that request is still writing, so continuing an answer does not delete the reasoning of the generation it continues. The tool-round copies of §5.4 are not reasoning and are not deleted with it.

System default is `PERSIST_REASONING_TOKENS="conversation"`; see [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the exact semantics and defaults.

### 6.1 Where a replayed reasoning item goes

A model produces reasoning at a particular moment: before a tool call, after its result, or between two
sentences of an answer. Open WebUI stores the answer as text, so that position is lost unless the pipe records
it. Each persisted reasoning item therefore carries an anchor - which call it preceded or followed, or which
assistant message it sat before - and replay puts it back in that place rather than appending it.

Anchors are **scoped to a turn**, where a turn is the region between user messages. Tool `call_id` values are
not guaranteed unique across a conversation: in chats saved before the pipe made the ids it invents unique, the
same id can appear in several turns, because the chat-completions adapters used to number calls per request.
Binding an anchor only within its own turn is what keeps a reasoning item from
attaching itself to an unrelated call with the same id. Open WebUI's own synthetic "Here are the images from
the tool results above" message counts as a user message for this purpose, which splits the region at the same
point on both the generating and the replaying side.

### 6.2 An answer continued across more than one request

"Continue response" adds a second generation to the same assistant message, and Open WebUI's own tool loop can
call the pipe several times within one turn. Ordinals are counted per request, so without care the continuation
would number its first call `0` again and its reasoning would bind to the first generation's call - placing two
thinking blocks side by side, the shape providers reject.

The pipe therefore offsets a continuation's ordinals by what the turn already contains: the calls and the
assistant messages the request carries before this generation starts. A turn that ends on reasoning gets one
further step, so the continuation's first block is placed after its own text rather than beside the block that
ended the previous generation. The offsets apply whether or not the Continue streams, since Open WebUI keeps the
earlier generation on a non-streamed Continue as it does on a streamed one.

The continuation's first output also has to bring its own line break. Open WebUI joins that output onto the
stored reply's last line. A hidden marker line that gains text after it is no longer a marker: its row stops
replaying, and the line either shows its id as text or, when the added text contains no space, hides the added text
along with the id. So when the stored reply ends on a marker line, the continuation's first model output starts on a
new line, whether it is text, a marker line or a generated picture. When the stored reply ends on ordinary text
instead, that output carries on the sentence. A card
or a refusal that opens a continuation - the provider's error card when a model refuses prefill, the breaker's
refusal, a busy or startup notice, or a configuration notice when the Continue is not streamed - starts a block of
its own on any Continue; on a streamed Continue a configuration notice shows as Open WebUI's error instead, and the
stored reply stays as it was. A reply written by internal Fusion is not continued: a Continue on one ends with a
notice saying that Continue is not available for Fusion replies and that regenerating runs Fusion again. The notice
starts a block of its own.

---

## 7. Failure modes (what happens when artifacts are missing)

- If the artifact loader fails (DB errors, network issues), the pipe logs a warning and continues without replaying artifacts for that assistant message.
- If an individual marker cannot be resolved to a payload (for example after key rotation or cleanup), the pipe logs a warning and skips that artifact.

Operational implications:
- Conversations may still render in the UI, but upstream requests may lack some historical tool/reasoning context.
- If you rely on long-lived replayability, validate your retention and key rotation procedures in [Persistence, Encryption & Storage](persistence_encryption_and_storage.md).
