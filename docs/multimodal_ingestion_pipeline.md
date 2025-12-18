# multimodal intake pipeline

**file:** `docs/multimodal_ingestion_pipeline.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:1104-2100`, `_download_remote_url`, `_parse_data_url`, `_inline_internal_file_url`

This document is the canonical reference for every image, file, audio, and video pathway that flows through the OpenRouter Responses pipe. Use it whenever you touch `_to_input_image`, `_to_input_file`, `_to_input_audio`, `_to_input_video`, download helpers, or storage fallbacks.

> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## 1. design goals

1. **Zero data loss** -- every inbound payload (remote URL, base64 blob, Open WebUI file reference) is persisted inside Open WebUI storage before being forwarded to OpenRouter. Chats stay replayable months later.
2. **Bounded resources** -- every helper enforces MB caps (`REMOTE_FILE_MAX_SIZE_MB`, `BASE64_MAX_SIZE_MB`, `VIDEO_MAX_SIZE_MB`) and chunked buffering (`IMAGE_UPLOAD_CHUNK_BYTES`) so workers stay responsive even under dozens of concurrent uploads.
3. **Secure by default** -- SSRF guards reject private-network hosts, fallback storage accounts stay least-privileged, and warnings surface immediately when operators override defaults.
4. **UX feedback** -- emitters broadcast download/upload progress so users know why a response is waiting on a large attachment.

---

## 2. storage context resolution

| Helper | Purpose |
| --- | --- |
| `_resolve_storage_context(request, user_obj)` | Returns `(request, user)` for uploads. Falls back to `_ensure_storage_user()` when the chat lacks an authenticated user (API automations, system triggers). |
| `_ensure_storage_user()` | Creates/loads the dedicated service account defined by `FALLBACK_STORAGE_{EMAIL,NAME,ROLE}`. New accounts receive a random `oauth_sub` so they cannot log in interactively. Warns if you configure a privileged role. |
| `_upload_to_owui_storage()` | Streams bytes into Open WebUI"s `/api/v1/files` endpoint with `process=False`. Returns the internal file URL (e.g., `/api/v1/files/XYZ`). Errors are logged and surfaced via `_emit_error`. |

Every multimodal helper first calls `_resolve_storage_context`. If storage cannot be resolved (e.g., synthetic tests), the helper degrades gracefully by skipping uploads and issuing a warning.

---

## 3. image handling

| Feature | Behavior |
| --- | --- |
| **Supported inputs** | `image_url` dicts, nested Open WebUI blocks, markdown image links extracted from assistant text (fallback mode). |
| **Data URLs** | `_parse_data_url` validates MIME type + size (`BASE64_MAX_SIZE_MB`), decodes the payload, and uploads it to storage. The final Responses block uses the internal URL converted via `_inline_internal_file_url`. |
| **Remote URLs** | `_download_remote_url` streams bytes with retry/backoff (`REMOTE_DOWNLOAD_*` valves), SSRF checks, and a hard MB cap. Successful downloads are uploaded to storage, ensuring future turns do not depend on third-party hosts. |
| **Internal URLs** | `/api/v1/files/...` links are streamed in configurable chunks (`IMAGE_UPLOAD_CHUNK_BYTES`) and re-encoded as `data:` URLs so OpenRouter never touches the Open WebUI host directly. |
| **Selection policy** | `MAX_INPUT_IMAGES_PER_REQUEST` + `IMAGE_INPUT_SELECTION` limit which images accompany a request. When the user supplies no images and the selection mode is `user_then_assistant`, the pipe reuses the most recent assistant-generated image URLs captured from markdown. |
| **Status messaging** | `_emit_status` posts `StatusMessages.IMAGE_BASE64_SAVED`, `IMAGE_REMOTE_SAVED`, etc., so operators know when uploads finish or fail. |

---

## 4. file handling

### Supported block fields

* `file_id`: already stored in Open WebUI. Passed through untouched.
* `file_data`: base64 string or `data:` URL. Always saved to storage when `SAVE_FILE_DATA_CONTENT=True` (default) so inline blobs do not bloat transcripts.
* `file_url`: remote URL. Only downloaded when `SAVE_REMOTE_FILE_URLS=True` to avoid surprise storage usage.
* `filename` and `mime_type`: preserved if provided; otherwise derived from headers or defaults.

### Flow

1. Extract every available field (nested dicts + top-level keys).
2. If `file_data` is a `data:` URL: validate size, decode, upload, and rewrite `file_url` to the internal path. The inline `file_data` field is dropped from the outgoing payload to save bandwidth.
3. If `file_data` is an HTTP(S) URL: treat it like a remote download so the final request references an internal storage path.
4. If `file_url` is remote and `SAVE_REMOTE_FILE_URLS` is true, download + re-host just like `file_data`.
5. Emit a `type: 
"input_file"` block containing whichever combination of `file_id`, `file_url`, `filename`, and `mime_type` remain relevant.
6. If anything fails, the helper emits an inline warning and returns a minimal block so the request stays valid.

---

## 5. audio handling

* Audio is only accepted as base64/data URLs per OpenRouter"s spec. `_to_input_audio` rejects remote URLs outright so users know to upload instead.
* Supported MIME types map to `wav` or `mp3` (see `_AUDIO_MIME_MAP`). Invalid types produce a warning + status event while leaving the text content untouched.
* Size validation reuses `_validate_base64_size` so large clips fail before decoding.
* Audio payloads are kept inline (no storage upload yet) because OpenRouter expects `input_audio` blocks. If you need durable storage, wrap the helper so it stores the file and injects a marker referencing the persisted blob.

---

## 6. video handling

| Feature | Behavior |
| --- | --- |
| Block normalization | `_to_input_video` accepts both `video_url` and the legacy `video` key, emitting a Chat Completions-style `{"type":"input_video"}` block. |
| Data URLs | Enforce `VIDEO_MAX_SIZE_MB` before decoding, then upload to storage so the final URL is internal. |
| Remote URLs | Allowed when they pass SSRF checks. Unlike images/files, videos are not re-hosted automatically to avoid huge storage bills. The helper emits a warning so operators know why an external URL was passed through. |
| Status updates | Long downloads emit progress/status lines via `_emit_status`, and failures fall back to text warnings. |

---

## 7. remote download subsystem

`_download_remote_url` is shared across image/file helpers and enforces:

* **Retry policy**: `REMOTE_DOWNLOAD_MAX_RETRIES`, `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS`, and `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS`. Retries cover network errors, HTTP 5xx, and HTTP 408/425/429.
* **Streaming**: Responses stream in chunks with precise byte accounting. The transfer aborts as soon as the real size or `Content-Length` exceeds `REMOTE_FILE_MAX_SIZE_MB` (or Open WebUI"s `FILE_MAX_SIZE` if smaller).
* **SSRF protection**: `_is_safe_url` resolves every host to IPv4/IPv6 addresses and rejects loopback, RFC1918, link-local, multicast, and reserved ranges.
* **MIME normalization**: e.g., `image/jpg` → `image/jpeg` so downstream validators stay consistent.
* **Status hooks**: When an `event_emitter` is provided, downloads broadcast start/finish/abort statuses.

> **Security note**: See [Security and Encryption](security_and_encryption.md#ssrf-protection) for compliance context, security procedures, and key rotation policies related to SSRF protection and other security measures.

---

## 8. configuration summary

| Valve | Default | Effect |
| --- | --- | --- |
| `REMOTE_FILE_MAX_SIZE_MB` | 50 | Applies to remote downloads and inline image/file payloads. Auto-clamps to Open WebUI"s `FILE_MAX_SIZE` when RAG uploads are enabled. |
| `BASE64_MAX_SIZE_MB` | 50 | Checked before decoding base64 data URLs. Prevents runaway memory usage. |
| `VIDEO_MAX_SIZE_MB` | 100 | Caps video data URL size. Remote links are not downloaded but are still validated before streaming. |
| `SAVE_FILE_DATA_CONTENT` | True | Forces all `file_data` payloads into storage for durability. |
| `SAVE_REMOTE_FILE_URLS` | False | When true, remote `file_url` entries are also downloaded+stored; when false they"re passed through untouched. |
| `MAX_INPUT_IMAGES_PER_REQUEST` | 5 | Hard limit on how many `input_image` blocks a single request can contain. |
| `IMAGE_INPUT_SELECTION` | `user_then_assistant` | Governs whether old assistant images can be reused when the user omits attachments. |
| `IMAGE_UPLOAD_CHUNK_BYTES` | 1 MB | Memory ceiling for inlining Open WebUI files as `data:` URLs. |

---

## 9. troubleshooting checklist

1. **"Image was ignored"** → verify the target model advertises `vision=true` in the catalog and that `MAX_INPUT_IMAGES_PER_REQUEST` was not exceeded.
2. **"Download blocked"** → check logs for SSRF warnings or size-limit errors; adjust `REMOTE_FILE_MAX_SIZE_MB` / `BASE64_MAX_SIZE_MB` only if you understand the cost.
3. **"Uploads fail for automations"** → confirm the fallback storage user exists and that `FALLBACK_STORAGE_ROLE` is not a forbidden value in your Open WebUI deployment.
4. **"Videos never play back"** → remember that videos are not re-hosted yet; ensure the remote URL stays reachable or implement optional storage similar to images.

Use this doc alongside `docs/history_reconstruction_and_context.md` to understand how multimodal content is replayed later. Together they guarantee OpenRouter receives the richest possible context without overloading Open WebUI.
