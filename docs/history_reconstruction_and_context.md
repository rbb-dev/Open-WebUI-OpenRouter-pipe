# History Reconstruction & Context Replay

Open WebUI stores chat history as a list of heterogeneous message objects. OpenRouter’s Responses API expects a structured `input` array containing messages and (optionally) structured tool/reasoning artifacts.

This document describes how the pipe builds that `input` array, how it replays persisted artifacts referenced by hidden ULID markers, and how retention/pruning valves affect what is sent upstream.

> **Quick navigation:** [Docs Home](README.md) · [Persistence](persistence_encryption_and_storage.md) · [Multimodal](multimodal_ingestion_pipeline.md) · [Valves](valves_and_configuration_atlas.md)

---

## 1. Entry point: `transform_messages_to_input`

The core history conversion is implemented by `Pipe.transform_messages_to_input(...)`.

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
    - `user_then_assistant`: if the user turn has no images, the pipe may reuse the most recent assistant-generated image URLs extracted from Markdown image syntax.
- Remote/data URL images are re-hosted into Open WebUI storage and/or inlined as `data:` URLs as needed so providers do not need to fetch from your Open WebUI host directly.

### 3.3 Files, audio, and video
The pipe includes transformer functions for:
- `input_file` / `file` → `input_file`
- `input_audio` / `audio` → `input_audio`
- `video_url` / `video` → `video_url`

The security/size/SSRF rules and re-hosting behaviors are documented in [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md).

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
- normalizes it to the schema expected by upstream (`_normalize_persisted_item`),
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

This prevents sending half of a tool interaction back to the model.

### 5.3 Tool output pruning by turn age (`TOOL_OUTPUT_RETENTION_TURNS`)
When `TOOL_OUTPUT_RETENTION_TURNS` is set, the pipe computes turn indices across the conversation and treats messages older than the retention window as “old”.

For old turns, it can prune very large `function_call_output.output` strings by:
- preserving a head and tail,
- inserting a note indicating the output was pruned,
- and leaving markers intact.

This keeps replay payloads smaller while preserving recency and high-level context.

---

## 6. Reasoning replay and `PERSIST_REASONING_TOKENS`

When replayed artifacts include reasoning items, the pipe can optionally record references in `replayed_reasoning_refs` so the caller can delete those artifacts after replay when reasoning retention is limited to a single turn.

System default is `PERSIST_REASONING_TOKENS="conversation"`; see [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the exact semantics and defaults.

---

## 7. Failure modes (what happens when artifacts are missing)

- If the artifact loader fails (DB errors, network issues), the pipe logs a warning and continues without replaying artifacts for that assistant message.
- If an individual marker cannot be resolved to a payload (for example after key rotation or cleanup), the pipe logs a warning and skips that artifact.

Operational implications:
- Conversations may still render in the UI, but upstream requests may lack some historical tool/reasoning context.
- If you rely on long-lived replayability, validate your retention and key rotation procedures in [Persistence, Encryption & Storage](persistence_encryption_and_storage.md).

