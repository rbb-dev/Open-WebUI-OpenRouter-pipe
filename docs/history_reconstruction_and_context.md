# history reconstruction & context replay

**file:** `docs/history_reconstruction_and_context.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:1104-2350`, marker helpers at `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:8132-8258`

Open WebUI stores messages as a heterogeneous list of dicts. The OpenRouter Responses API expects a tightly structured `input[]` array with blocks such as `input_text`, `input_image`, `response_text`, and `tool_result`. This document describes how the pipe bridges the two, how persisted artifacts are recovered via ULID markers, and how pruning/retention knobs influence the final payload.

> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## 1. message normalization pipeline

1. **System / developer roles**. Messages whose `role` is `system` or `developer` are buffered in `pending_instructions`. They are concatenated and prepended to the next user turn so the provider receives a single instruction block rather than disjoint snippets. The backlog resets whenever a user message is emitted.
2. **User turns**. `_transform_messages_to_input` walks every content block in the message:
   * Strings become `{"type": "input_text", "text": ...}` blocks.
   * Image blocks flow through `_to_input_image` (see `docs/multimodal_ingestion_pipeline.md`). The helper enforces `MAX_INPUT_IMAGES_PER_REQUEST`, honors `IMAGE_INPUT_SELECTION` (user-only vs. user-then-assistant fallback), and rewrites every remote or base64 payload into an internal `/api/v1/files/...` URL before inlining it as a `data:` URI.
   * File/audio/video blocks call `_to_input_file`, `_to_input_audio`, or `_to_input_video` respectively. Each helper runs SSRF guards, size checks, retry logic, and storage uploads before emitting the Responses-style block.
   * The resulting list of blocks is appended to `openai_input` as `{"role": "user", "content": [...]}`, preceded by the buffered instructions when present.
3. **Assistant turns without markers**. If the assistant message body contains no hidden markers, the code emits a simple `{ "role": "assistant", "content": [{"type": "output_text", "text": ...}] }` block. This preserves any inline markdown (including citations) and primes the next request"s context.
4. **Assistant turns with markers**. When `_contains_marker` detects `[XXXXXXXXXXXXXXX]: #` lines, the message is split into literal segments and marker segments. Each marker corresponds to a persisted artifact stored by `_persist_artifacts`. The loader fetches missing payloads (DB or Redis) and reconstructs the original tool results, reasoning traces, images, or files inline. Plain text before/after markers is preserved verbatim so the user sees exactly what the assistant typed.

---

## 2. ULID marker mechanics

| Concept | Description |
| --- | --- |
| Marker format | Each persisted artifact is tagged with `[{ULID}]: #`. The bracketed line is invisible in the UI but survives inside the raw assistant text so the pipe can rehydrate artifacts later. |
| ULID generation | `generate_item_id()` composes 16 Crockford base32 characters of timestamp followed by 4 characters of randomness. This keeps sort order stable and makes it trivial to compare age across artifacts. |
| Serialization helpers | `_serialize_marker`, `_iter_marker_spans`, `split_text_by_markers`, and `_extract_marker_ulid` scan assistant text, pull markers out, and provide offsets for reassembly without touching unrelated text. |
| Artifact loader hook | `_transform_messages_to_input` accepts `artifact_loader(chat_id, message_id, ulids)`. The pipe passes `_load_artifacts_for_message` here so persisted payloads come from Redis or SQLAlchemy. Tests can inject fakes. |
| Replay tracking | `replayed_reasoning_refs` collects `(chat_id, artifact_id)` tuples whenever reasoning artifacts are reinserted. If `PERSIST_REASONING_TOKENS` is `"next_reply"`, the caller deletes those rows once the assistant finishes the following turn so secrets do not linger longer than intended. |
| Tool pruning | `_prune_tool_output` shortens large `function_call_output` payloads once they fall outside the `TOOL_OUTPUT_RETENTION_TURNS` window. The helper keeps the head/tail, inserts a note inside the transcript, and logs the removal so token usage stays predictable. |
| Schema normalization | `_normalize_persisted_item` guarantees every stored artifact has the fields the Responses API expects (`id`, `status`, `call_id`, JSON-stringified `arguments`, etc.). Function-call artifacts without a valid payload are filtered out before replay. |
| Orphan detection | `_classify_function_call_artifacts` matches `function_call` entries with their corresponding `function_call_output`. Orphaned calls or outputs are ignored so the model never receives half of a tool interaction. |

---

## 3. retention knobs

| Valve / setting | Impact |
| --- | --- |
| `PERSIST_REASONING_TOKENS` (system + user valve) | Controls how long reasoning artifacts remain replayable: `"disabled"` stores nothing, `"next_reply"` deletes markers after a single replay, `"conversation"` keeps them until the chat is cleared. |
| `PERSIST_TOOL_RESULTS` | When `False`, tool outputs are never stored. The assistant still references tools in real time, but regenerations cannot replay them. Use sparingly. |
| `TOOL_OUTPUT_RETENTION_TURNS` | Limits the number of recent turns whose tool outputs are sent verbatim. Older turns are pruned using `_prune_tool_output`, keeping transcripts short without losing the high-level summary inserted into the text. |
| `MAX_INPUT_IMAGES_PER_REQUEST` | Caps how many `input_image` blocks go out per request. When exceeded the pipe emits a warning status and drops the oldest attachments from that user turn. |
| `IMAGE_INPUT_SELECTION` | `"user_turn_only"` ensures only current-turn images are forwarded. `"user_then_assistant"` (default) lets the pipe fall back to the most recent assistant-generated images (detected via markdown image links) when the user did not attach new ones. |

---

## 4. error handling & user feedback

* **Inline emitters**. `_emit_status` is called whenever an attachment is skipped, a download fails, or a marker could not be resolved. Users see progress messages like "📥 downloading remote image..." or warnings such as "⚠️ failed to reload stored reasoning chain".
* **Graceful degradation**. When `_to_input_*` raises, the helper catches the exception, logs the stack, emits a warning, and returns an empty block (e.g., `{ "type": "input_image", "image_url": "" }`). The model receives structurally valid input and the chat continues.
* **Artifact holes**. If a marker is found but the DB/Redis lookup returns nothing (e.g., key rotation), the pipe logs a warning and substitutes a short `response_text` block describing the missing artifact. This makes data loss obvious to end users.
* **Retention mismatches**. If a user requests `PERSIST_REASONING_TOKENS="conversation"` but the system valve forces `disabled`, the user override is ignored and a status message explains why. This prevents confusing "I set it but nothing was replayed" reports.

---

## 5. adding new content types

1. Extend `_transform_messages_to_input` with a helper tailored to the new block type (follow the existing `_to_input_*` naming convention).
2. Include size guards, SSRF protections, and storage fallback logic; most of it can reuse `_parse_data_url`, `_download_remote_url`, `_upload_to_owui_storage`, and `_inline_internal_file_url`.
3. Document the new type here and in `docs/multimodal_ingestion_pipeline.md` so developers know how it is persisted and replayed.
4. If the new type creates persisted artifacts, ensure `_serialize_marker` can represent it and that `_load_artifacts_for_message` knows how to rehydrate the payload.

By keeping history reconstruction deterministic and well documented, the manifold can persist huge reasoning/tool chains without bloating UI transcripts or leaking sensitive data.
