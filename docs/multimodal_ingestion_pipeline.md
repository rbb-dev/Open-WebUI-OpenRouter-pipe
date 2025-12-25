# Multimodal Intake Pipeline

This document describes how the pipe transforms Open WebUI message content into OpenRouter-compatible multimodal blocks (images, files, audio, and video), including storage behavior, SSRF protections, and size limits.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Security](security_and_encryption.md) · [History/Replay](history_reconstruction_and_context.md)

---

## What this pipeline does

For user messages that include multimodal content, the pipe normalizes content blocks into the structures it sends to OpenRouter.

At a high level:

- **Images** are converted to Responses-style `input_image` blocks and (when sourced from remote URLs or data URLs) are **re-hosted** into Open WebUI storage.
- **Files** are converted to Responses-style `input_file` blocks and may be re-hosted into Open WebUI storage depending on valve configuration (defaults re-host common cases).
- **Audio** is converted to Responses-style `input_audio` blocks and must be **base64/data URL** (remote URLs are rejected).
- **Video** is passed using Chat Completions-style `video_url` blocks (the Responses API does not provide a dedicated `input_video` block). Videos are **not** downloaded or re-hosted by the pipe; the pipe applies basic validation and SSRF checks for remote URLs.

---

## Storage context resolution (uploads to Open WebUI)

When the pipe needs to re-host content into Open WebUI storage, it resolves a storage context `(request, user)`:

- If a real Open WebUI request/user context exists, the pipe uploads via Open WebUI’s file APIs.
- If a user context is missing (for example some automations), the pipe can fall back to a dedicated “storage owner” identity configured by:
  - `FALLBACK_STORAGE_EMAIL`
  - `FALLBACK_STORAGE_NAME`
  - `FALLBACK_STORAGE_ROLE`

If storage cannot be resolved (for example in tests or synthetic contexts), the pipe degrades by skipping uploads and returning minimal valid blocks.

---

## Images (`image_url` → `input_image`)

### Inputs accepted
- Open WebUI content blocks containing `type: "image_url"` with:
  - a nested object `{ "image_url": { "url": "..." } }`, or
  - a string `{ "image_url": "..." }`.

### Storage behavior (important)
- If the image is provided as a **data URL** (`data:image/...;base64,...`), the pipe validates size, decodes, and uploads it to Open WebUI storage. The outgoing `input_image.image_url` references the internal Open WebUI file URL.
- If the image is provided as a **remote URL** (`http://` or `https://`), the pipe downloads it (with retries/limits/SSRF protection), uploads it to Open WebUI storage, and references the internal URL.
- If the image is already an **Open WebUI file URL** (for example `/api/v1/files/...`), the pipe streams it and inlines it as a `data:` URL to avoid requiring OpenRouter to fetch from your Open WebUI host.

This image re-hosting behavior is intentionally “always on” for data URLs and remote URLs to avoid chat history bloat and to preserve replayability.

### Limits and selection
- `MAX_INPUT_IMAGES_PER_REQUEST` limits how many images will be forwarded.
- `IMAGE_INPUT_SELECTION` controls whether the pipe can fall back to recent assistant images when the current user turn has no attachments.

---

## Files (`input_file` / `file` → `input_file`)

### Fields accepted
The file transformer extracts and forwards the following Responses-compatible fields when present:

- `file_id` (already in Open WebUI storage)
- `file_data` (base64/data URL, or a URL-like string depending on upstream)
- `file_url` (URL to a file)
- `filename`

### Re-hosting behavior (defaults)
The pipe can re-host both `file_data` and `file_url` into Open WebUI storage:

- When `SAVE_FILE_DATA_CONTENT=True` (default), and `file_data` is:
  - a `data:` URL, it is decoded and stored; the outgoing block will prefer `file_url` pointing to internal storage and will clear `file_data`.
  - an `http://` or `https://` URL, it is downloaded and stored; the outgoing block will set `file_url` to internal storage and clear `file_data`.
- When `SAVE_REMOTE_FILE_URLS=True` (default), and `file_url` is:
  - a `data:` URL, it is decoded and stored; the outgoing block rewrites `file_url` to the internal storage URL.
  - an `http://` or `https://` URL, it is downloaded and stored; the outgoing block rewrites `file_url` to the internal storage URL.

If you want to reduce storage growth and accept the tradeoff that chat replay depends on third-party URLs, set `SAVE_REMOTE_FILE_URLS=False`.

---

## Audio (`input_audio` / `audio` → `input_audio`)

OpenRouter audio inputs require base64-encoded audio, and the pipe enforces that:

- Remote URLs (`http://` / `https://`) are rejected and replaced with an empty `input_audio` block.
- Data URLs (`data:audio/...;base64,...`) are accepted if valid and within size limits.
- Raw base64 strings are accepted if valid.

Supported formats are normalized to `mp3` or `wav` based on MIME hints when available; unknown types default to `mp3`.

**Limitation:** The pipe does not currently re-host audio into Open WebUI storage; audio remains inline in the request it sends upstream.

---

## Video (`video_url` / `video` → `video_url`)

### Block format
The pipe uses Chat Completions-style `video_url` blocks:

```json
{
  "type": "video_url",
  "video_url": { "url": "https://example.com/video.mp4" }
}
```

### Validation and SSRF behavior
- Data URLs (`data:video/...;base64,...`) are accepted only if their estimated decoded size is at or below `VIDEO_MAX_SIZE_MB`.
- YouTube URLs are allowed (and may only work on certain model/provider combinations).
- Remote URLs (`http://` / `https://`) that are not Open WebUI file URLs are checked by the SSRF guard when `ENABLE_SSRF_PROTECTION=True`.

**Limitation:** Videos are not downloaded or re-hosted by the pipe; it passes the URL (or data URL) through after applying basic checks. If you need durability for video content, you must ensure the URL remains reachable or implement a storage policy outside this pipe.

---

## Remote downloads (`_download_remote_url`)

Remote downloads are used for images and for files when re-hosting is enabled. The downloader enforces:

- **Protocols:** `http://` and `https://` only.
- **SSRF protection:** blocks private/internal address targets when enabled.
- **Retry/backoff:** retries on network errors and transient HTTP statuses (`>=500` and `408/425/429`), with exponential backoff controlled by valves.
- **Size limits:** enforced via `REMOTE_FILE_MAX_SIZE_MB` (and may be further capped to Open WebUI’s configured upload limits when available).

---

## Configuration summary (key valves)

| Valve | Default (verified) | What it controls |
| --- | --- | --- |
| `ENABLE_SSRF_PROTECTION` | `True` | Blocks remote downloads to private/internal network ranges. |
| `REMOTE_DOWNLOAD_MAX_RETRIES` | `3` | Retry attempts for remote downloads. |
| `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS` | `5` | Initial retry delay (exponential backoff). |
| `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS` | `45` | Max total retry time budget for one download. |
| `REMOTE_FILE_MAX_SIZE_MB` | `50` | Size cap for remote downloads (images/files) and related payload guards. |
| `SAVE_FILE_DATA_CONTENT` | `True` | Re-host `file_data` content into Open WebUI storage to avoid transcript bloat. |
| `SAVE_REMOTE_FILE_URLS` | `True` | Re-host remote/data URLs in `file_url` into Open WebUI storage. |
| `BASE64_MAX_SIZE_MB` | `50` | Base64 payload size guard before decoding. |
| `IMAGE_UPLOAD_CHUNK_BYTES` | `1048576 (1 MiB)` | Chunk size used when inlining Open WebUI-hosted images as `data:` URLs. |
| `MAX_INPUT_IMAGES_PER_REQUEST` | `5` | Maximum images forwarded per request. |
| `IMAGE_INPUT_SELECTION` | `user_then_assistant` | Image selection policy when the user attaches no images. |
| `VIDEO_MAX_SIZE_MB` | `100` | Size guard for base64 video data URLs. |

For the complete list, see [Valves & Configuration Atlas](valves_and_configuration_atlas.md).

---

## Troubleshooting checklist

1. Image attachments are ignored: confirm the selected model supports vision and that `MAX_INPUT_IMAGES_PER_REQUEST` is not exceeded.
2. “Download blocked” / “URL blocked”: check SSRF controls (`ENABLE_SSRF_PROTECTION`) and confirm the URL is not internal/private.
3. Large payload failures: review `REMOTE_FILE_MAX_SIZE_MB`, `BASE64_MAX_SIZE_MB`, and `VIDEO_MAX_SIZE_MB`.
4. Video inputs do not work: provider/model support varies; the pipe does not re-host video and cannot force upstream providers to accept the URL.

