# Multimodal Intake Pipeline

This document describes how the pipe transforms Open WebUI message content into OpenRouter-compatible multimodal blocks (images, files, audio, and video), including storage behavior, SSRF protections, HTTPS-only defaults for remote URLs, and size limits.

If you are specifically trying to **bypass Open WebUI RAG for chat uploads** and forward uploads as **native OpenRouter attachments**, see:
- [OpenRouter Direct Uploads (bypass OWUI RAG)](openrouter_direct_uploads.md)

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Security](security_and_encryption.md) · [History/Replay](history_reconstruction_and_context.md)

---

## What this pipeline does

For user messages that include multimodal content, the pipe normalizes content blocks into the structures it sends to OpenRouter.

At a high level:

- **Images** are converted to Responses-style `input_image` blocks. A `data:` URL within `BASE64_MAX_SIZE_MB` is sent as it came apart from the scheme, which is lower-cased to `data:`, and one over it is not sent, the picture being skipped and the person told in a status on their latest message; a remote image is downloaded and sent inline (or forwarded as its link when it cannot be downloaded, unless the pipe's own address check refused the host), a picture the pipe refuses to forward is dropped and reported rather than sent, and an Open WebUI file URL is read with the requester's access and sent inline.
- **Files** are converted to Responses-style `input_file` blocks and forwarded as they came: a `data:` URL or link in `file_data` or `file_url` goes out unchanged, and the pipe never downloads a file link. Inline data over `BASE64_MAX_SIZE_MB` is the exception: it is not sent, and the person sees "Files: skipped …" for their latest message, as with pictures. An Open WebUI file URL is read with the requester's access and sent inline.
- **Audio** is converted to Responses-style `input_audio` blocks and must be **base64/data URL** (remote URLs are rejected).
- **Video** is passed using Chat Completions-style `video_url` blocks (the Responses API does not provide a dedicated `input_video` block). Videos are **not** downloaded or re-hosted by the pipe; the pipe applies basic validation and SSRF checks for remote URLs.

Nothing a person attaches is written to Open WebUI storage by the pipe, in any chat and by any request of a turn.

---

## Storage context resolution (uploads to Open WebUI)

This input path never writes to Open WebUI storage. Open WebUI already converts a saved chat's stored pictures back to base64 before each model call, so a copy the pipe made would only be read straight back. Copies an earlier release made of a person's attachments stay where they are: ordinary files owned by that person (or by the fallback account below), listed in their Files. Nothing reads them, and the pipe does not delete them.

The pipe stores only what a model generates, and for that it resolves a storage context `(request, user)`:

- If a real Open WebUI request/user context exists, the pipe uploads via Open WebUI’s file APIs.
- If a user context is missing (for example some automations), a picture a model generates is stored under a dedicated “storage owner” identity configured by:
  - `FALLBACK_STORAGE_EMAIL`
  - `FALLBACK_STORAGE_NAME`
  - `FALLBACK_STORAGE_ROLE`

  The resolved account is memoised, but the memo is keyed on the email it was derived from: an unchanged email reuses the same account, and a changed one re-resolves on the next request that needs a storage owner.

If storage cannot be resolved (for example in tests or synthetic contexts), the pipe skips the upload.

---

## Reading Open WebUI files (storage gateway)

When the pipe needs to **read** an already-stored Open WebUI file (for example an `/api/v1/files/...` URL to inline as a `data:` URL, or an `input_file.file_id`), every read routes through a single backend-agnostic gateway: `OwuiFileGateway` in `storage/owui_files.py`.

For each read the gateway:

- **Authorizes** the requester against the file (owner, admin, or Open WebUI’s `has_access_to_file`), failing closed if access is denied.
- **Routes the read through Open WebUI’s `Storage` provider**, so it works on any backend — `local`, `s3`, `gcs`, or `azure` (this fixes [issue #46](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/issues/46), where cloud-backed files were previously unreadable). `FileModel.path` is treated as an opaque storage key; the local provider's read is a no-op (no remote download).
- **Size-gates** the content against `BASE64_MAX_SIZE_MB`.
- **Copies every real OWUI file read to a request-owned private temp** (regardless of provider) before any bytes are encoded, then cleans the temp up.

For cloud/unknown backends, a file whose declared `meta['size']` is missing or invalid is **refused by default** to avoid an unbounded download. The global `ALLOW_UNKNOWN_SIZE_CLOUD_READS` Valve opts back in (the download is still capped by `BASE64_MAX_SIZE_MB`); it is a download-safety gate, not an authorization bypass. See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for its exact default and semantics.

---

## Images (`image_url` → `input_image`)

### Inputs accepted
- Open WebUI content blocks containing `type: "image_url"` with:
  - a nested object `{ "image_url": { "url": "..." } }`, or
  - a string `{ "image_url": "..." }`.

### What is sent (important)

The pipe never writes an image the person attached to Open WebUI storage: not for a saved, channel or temporary chat, and not for any request of a turn (its first answer, a Regenerate, each further model answering it, a Continue, or Open WebUI's calls back after each round of tool calls). Every request sends the image as the message carries it:
- A **data URL** (`data:image/...;base64,...`) within `BASE64_MAX_SIZE_MB` is sent as it came apart from the scheme, which is lower-cased to `data:`; one over the limit is not sent, the picture is skipped and the person sees `"Images: skipped N (…)."` on their latest message. One that fails validation is dropped and reported, never sent unvalidated. The `;base64` marker is matched case-insensitively, as the data-URL standard requires, so `;BASE64,` is a spelling of the same thing and the URL is still forwarded with its own spelling intact. A token-free `data:` URL — the word `base64` in a payload, or a parameter that merely starts with those letters — is not base64 and is refused, in every spelling.
- A **remote URL** (`https://`) is downloaded (with retries/limits/SSRF protection) and its bytes are sent upstream as a `data:` URL; one that cannot be downloaded is forwarded as its link **unless the pipe's own address check refused the host**: a host that does not resolve, or resolves to a non-routable address, fails closed and is not sent, and the person sees `Images: skipped N (could not be fetched, so it was not sent).` A public link the pipe merely failed to *download* (404, timeout, slow host) is forwarded as before, so OpenRouter may fetch it. A failed download costs one further address check on top of the one the download already ran, so a picture the pipe cannot fetch is refused after two address checks, and a turn carrying N such pictures waits for up to 2 x N x `ADDRESS_CHECK_SECONDS` (5s each) of checking, one picture after another. A download that returns no bytes counts as a failed download, so it is forwarded as its link rather than shipped as a zero-byte `data:` URL. One larger than `BASE64_MAX_SIZE_MB` once downloaded is not sent. Plain `http://` is disabled by default and requires explicit allowlisting.
- An **Open WebUI file URL** (for example `/api/v1/files/...`) is streamed with the requester's access and inlined as a `data:` URL to avoid requiring OpenRouter to fetch from your Open WebUI host.

An image the pipe **reuses** from an earlier turn is inlined as a `data:` URL too. A remote picture a saved chat reuses is cached in the worker's memory for the life of the process so later turns fetch it once; in a temporary chat it is re-fetched per request and kept nowhere. Bytes that carry a recognisable image signature decide its media type, whatever the source declared; where they carry none, the declaration decides. A reuse is also dropped when the SSRF gate, re-run for it on every send, now refuses the URL, so tightening `ENABLE_SSRF_PROTECTION` or editing a host allowlist takes effect on the next turn rather than the next fetch. That re-run costs one address resolution and no download; a URL it then refuses is fetched once more through the already-gated path, which runs its own address check, so a refusal pays two resolutions. A resolution that does not finish inside the address-check budget counts as a refusal, so a slow nameserver costs a re-fetch rather than a permission. Nothing has to be refused for the re-check to be paid for: a request reusing `MAX_INPUT_IMAGES_PER_REQUEST` pictures pays one blocking resolution per reused picture, one after another (5 by default, 20 at the ceiling). Otherwise a reuse is dropped only when the type settled on this way is not an image type - so a payload declaring an image type whose bytes the pipe does not recognise (BMP and TIFF among them) is forwarded under its declaration, and one the pipe could not fetch at all is dropped. An `http(s)` reuse that the pipe could not fetch is reported as "could not be fetched, so it was not sent" rather than "could not be fetched, so its type could not be established": the fetch fails before any type gate runs, so the type-gate wording no longer describes it, and the type-gate cause now covers non-`http(s)` schemes only. A dropped reuse is reported to the person with the same `Images: skipped N (…)` status as a refused attachment, on the turn that triggered it. Nothing on this input path writes a file - images the model *generates* are stored separately, and that is an output-side behavior.

### Limits and selection
- `MAX_INPUT_IMAGES_PER_REQUEST` caps how many images one of the person's messages forwards, counting a picture reused from earlier in the conversation. Pictures a tool returns are never cut by this limit. Whenever a tool round's result reaches the model, all of its pictures go with it. A picture the pipe refuses is not sent, and the person is told, as for one they attached: the refusal is reported on the turn whose round carries the picture, which for a tool's pictures is the round being answered, not the person's next question.
- `IMAGE_INPUT_SELECTION` controls whether the pipe can fall back to the most recent image already in the conversation - the model's or the user's - when the current user turn has no attachments.
- `IMAGE_REUSE_MAX_TURNS` bounds how long that image stays available, so a long text conversation stops resending a picture nobody is discussing.

---

## Files (`input_file` / `file` → `input_file`)

### Fields accepted
The file transformer extracts and forwards the following Responses-compatible fields when present:

- `file_id` (already in Open WebUI storage)
- `file_data` (base64/data URL, or a URL-like string depending on upstream)
- `file_url` (URL to a file)
- `filename`

### What is sent
Every request forwards `file_data` and `file_url` as they came, except inline data over the size limit, for every chat and every request of a turn, and the pipe writes no file the person attached to Open WebUI storage:

- Inline data (a `data:` URL in either field, or raw base64 in `file_data`) up to `BASE64_MAX_SIZE_MB` is sent unchanged.
- Larger inline data is not sent. Without a `file_id` the file is dropped, and the person sees "Files: skipped …" for their latest message, as with pictures; when the block has a `file_id`, only the oversized field is dropped and the `file_id` is sent.
- An `https://` link in either field is sent unchanged for the provider to fetch; the pipe does not download it. Plain `http://` is disabled by default and requires explicit allowlisting: a cleartext link whose host is not allowlisted has its **whole block dropped**, and the person sees "Files: skipped …", the same as an oversized file.
- An Open WebUI file URL in either field becomes a `file_id`, which is read with the requester's access and sent inline as `file_data`.

---

## Audio (`input_audio` / `audio` → `input_audio`)

OpenRouter audio inputs require base64-encoded audio, and the pipe enforces that:

- Remote URLs (`http://` / `https://`) are rejected and replaced with an empty `input_audio` block.
- Data URLs (`data:audio/...;base64,...`) are accepted if valid and within size limits; one over `BASE64_MAX_SIZE_MB` is not sent at all, the block is dropped, the person is told in a status on their latest message, and it is logged as a size refusal rather than an encoding failure. "Valid" is enforced rather than assumed: the payload must be the base64 alphabet with RFC 2045 line wrapping and nothing else, so a payload of stray characters is refused instead of decoding to silence. The same parser serves pictures and audio, so the `;base64` marker is matched case-insensitively here too: `;BASE64,` is the same marker, spelled differently, and is accepted rather than refused as "Audio input must be base64-encoded audio data".
- Raw base64 strings are accepted if valid, and refused by size the same way.

Supported formats are normalized to `mp3` or `wav` based on MIME hints when available; unknown types default to `mp3`.

The pipe never writes audio to Open WebUI storage; it stays inline in the request it sends upstream.

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
- Remote URLs (`https://` by default; `http://` only when allowlisted) that are not Open WebUI file URLs are checked by the SSRF guard when `ENABLE_SSRF_PROTECTION=True`.

**Limitation:** Videos are not downloaded or stored by the pipe; it passes the URL (or data URL) through after applying basic checks. If you need durability for video content, you must ensure the URL remains reachable or implement a storage policy outside this pipe.

---

## Remote downloads (`_download_remote_url`)

Remote downloads are used for picture links: one in the conversation, or one a model returns for a picture it generated. The pipe never downloads a file link. The downloader enforces:

- **Protocols:** `https://` only by default; `http://` requires `ALLOW_INSECURE_HTTP` + allowlisting.
- **SSRF protection:** blocks private/internal address targets when enabled.
- **Retry/backoff:** retries on network errors and transient HTTP statuses (`>=500` and `408/425/429`), with exponential backoff controlled by valves.
- **Size limits:** enforced via `REMOTE_FILE_MAX_SIZE_MB`, and — when RAG is enabled and an admin cap is stored — additionally held to the value the Open WebUI admin last saved under **Admin → Settings → Documents → Max Upload Size**, the same number Open WebUI applies to the same file, read from its own store so a change takes effect without a restart. Normally the lower of the two wins; the one exception is this valve left at its default of 50, where a larger admin cap wins instead, clipped to the pipe’s own 500 MB ceiling. A cleared admin box lifts Open WebUI’s cap, not the valve’s: left at its 50 MB default the valve still refuses anything over 50 MB.

---

## Configuration summary (key valves)

| Valve | Default (verified) | What it controls |
| --- | --- | --- |
| `ENABLE_SSRF_PROTECTION` | `True` | Fetches only addresses that are provably globally routable, so private/internal ranges, carrier-grade NAT and IPv6 site-local are all refused. HTTPS-only defaults still apply even if SSRF protection is disabled. |
| `ALLOW_INSECURE_HTTP` | `False` | Allow plaintext HTTP remote URLs when explicitly enabled. HTTP is disabled by default. |
| `ALLOW_INSECURE_HTTP_HOSTS` | `""` | Comma-separated list of hosts or host:port entries allowed for plaintext HTTP. Exact match only (no wildcards). Empty means no HTTP allowed. |
| `REMOTE_DOWNLOAD_MAX_RETRIES` | `3` | Retry attempts for remote downloads. |
| `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS` | `5` | Initial retry delay (exponential backoff). |
| `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS` | `45` | Max total retry time budget for one download. Also caps any single retry wait, including one a server asked for in a `Retry-After` header. |
| `REMOTE_FILE_MAX_SIZE_MB` | `50` | Size cap for downloading a picture from a link (a file link is never downloaded) and related payload guards. |
| `BASE64_MAX_SIZE_MB` | `50` | Base64 payload size guard before decoding. |
| `IMAGE_UPLOAD_CHUNK_BYTES` | `1048576 (1 MiB)` | Chunk size used when inlining Open WebUI-hosted images as `data:` URLs. |
| `MAX_INPUT_IMAGES_PER_REQUEST` | `5` | Maximum images one of the person's messages forwards, counting a reused picture; pictures a tool returns are not counted. |
| `IMAGE_INPUT_SELECTION` | `user_then_assistant` | Image selection policy when the user attaches no images. |
| `IMAGE_REUSE_MAX_TURNS` | `3` | How many turns an earlier image stays available for reuse. |
| `VIDEO_MAX_SIZE_MB` | `100` | Size guard for base64 (`data:`) videos and for stored videos re-read to extract frames. |

For the complete list, see [Valves & Configuration Atlas](valves_and_configuration_atlas.md).

---

## Troubleshooting checklist

1. Image attachments are ignored: confirm the selected model supports vision and that `MAX_INPUT_IMAGES_PER_REQUEST` is not exceeded.
2. “Download blocked” / “URL blocked”: check SSRF controls (`ENABLE_SSRF_PROTECTION`) and confirm the URL is not internal/private.
3. Large payload failures: review `REMOTE_FILE_MAX_SIZE_MB`, `BASE64_MAX_SIZE_MB`, and `VIDEO_MAX_SIZE_MB`.
4. Video inputs do not work: provider/model support varies; the pipe does not download or store video and cannot force upstream providers to accept the URL.
