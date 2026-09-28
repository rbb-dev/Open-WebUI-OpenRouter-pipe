# Multimodal Intake Pipeline

This document describes how the pipe transforms Open WebUI message content into OpenRouter-compatible multimodal blocks (images, files, audio, and video), including storage behavior, SSRF protections, HTTPS-only defaults for remote URLs, and size limits.

If you are specifically trying to **bypass Open WebUI RAG for chat uploads** and forward uploads as **native OpenRouter attachments**, see:
- [OpenRouter Direct Uploads (bypass OWUI RAG)](openrouter_direct_uploads.md)

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Security](security_and_encryption.md) · [History/Replay](history_reconstruction_and_context.md)

---

## What this pipeline does

For user messages that include multimodal content, the pipe normalizes content blocks into the structures it sends to OpenRouter.

At a high level:

- **Images** are converted to Responses-style `input_image` blocks. A `data:` URL within `BASE64_MAX_SIZE_MB` is sent with its declared type and parameters preserved, its payload de-wrapped and validated once, and one over it is not sent, the picture being skipped and the person told in a status on their latest message; a remote image is downloaded, and its media type is resolved from the bytes before it is sent inline (or forwarded as its link when it cannot be downloaded, unless the pipe's own address check refused the host), a downloaded payload that settles on a non-image type is not inlined but refused and reported,  a link the pipe cannot resolve into an image is refused on the inline leg with an `Images: skipped N (not a link the pipe can resolve into an image)` status, a picture the pipe refuses to forward is dropped and reported rather than sent, and an Open WebUI file URL is read with the requester's access and sent inline.
- **Files** are converted to Responses-style `input_file` blocks and forwarded as they came: a `data:` URL or link in `file_data` or `file_url` goes out unchanged, and the pipe never downloads a file link. Inline data over `BASE64_MAX_SIZE_MB` is the exception: it is not sent, and the person sees "Files: skipped …" for their latest message, as with pictures. A block carrying none of `file_id`, `file_data` or `file_url` — a filename on its own, or an empty `file` part — is also skipped rather than forwarded, because Open WebUI's own converter drops such a part (for the keys it recognises) and because a part that declares only its type is accepted by OpenRouter and then unreadable. The pipe keeps a `file_url` that Open WebUI's converter drops, which is a deliberate superset of its rule. An Open WebUI file URL is read with the requester's access and sent inline.
- **Audio** is converted to Responses-style `input_audio` blocks and must be **base64/data URL** (remote URLs are rejected).
- **Video** is passed as `input_video` blocks, with `video_url` a bare string, on a turn that resolves to `/responses`; a turn that resolves to `/chat/completions` keeps the Chat Completions-style `video_url` block, whose `video_url` is an object. The two shapes are not interchangeable, and each endpoint is sent the one it reads. Videos are **not** downloaded or re-hosted by the pipe; the pipe applies basic validation and SSRF checks for remote URLs.

Nothing a person attaches is written to Open WebUI storage by the pipe, in any chat and by any request of a turn.

### What a user turn is allowed to carry

Every user turn the pipe dispatches carries at least one usable block. Four shapes are dropped rather than sent:

- a text block that is **only whitespace**;
- an `input_file` with no `file_id`, `file_data` or `file_url`;
- a media block whose wrapper is present but whose payload is not — an `input_image` / `image_url` whose `url` is `""`, an `input_audio` whose `data` is `""` inside a non-empty `{ "data": …, "format": … }` wrapper, a `video_url` whose `url` is `""`.

The media cases are decided on the **payload**, not on the wrapper. `{"url": ""}` and `{"data": "", "format": "mp3"}` are non-empty objects, so a plain truthiness test passes exactly the blocks being dropped; the payload itself is what has to be there. These are all the pipe's own conversion failures — the block never existed to be sent — which is why the empty `data`/`url` wrapper is not a thing Open WebUI produces and the pipe does not forward it as one. The reason a block was dropped this way is reported on the same turn, in the `Images:` or `Files:` status line, whether or not the turn also carried text. A payload-less media block beside real text is **dropped**, not forwarded: the status says "skipped", and nothing the status calls skipped is put in front of the provider.

A whitespace text block beside a block that *is* usable is **kept**: only a wholly void turn is emptied. So a person who sends a real picture and a blank box gets both blocks, in the order they wrote them.

What an emptied turn carries instead depends on what was dropped:

- A turn that was **blank text with no attachment at all** carries `[The user sent an empty message.]`. It states the fact in the same bracketed register as the attachment note and tells the model nothing false.
- A turn that was **blank text beside a block that carried no payload**, or that was **only an attachment that could not be converted**, carries `[An attached item was not sent: <reason>.]` — one note for both. A person who attached something that did not go out did not send an empty message, whatever else was on the turn, so the two are no longer told apart. Open WebUI's own empty-block rule (`strip_empty_content_blocks`, `utils/misc.py`) drops the blank text here and invents nothing; the note is this pipe adding what did not go out, in the register it already uses for exactly that.

A turn whose `content` is absent or `None` is not one of these: it carried no block to drop, so it is forwarded as an empty list, unchanged. `content: ""` is the third spelling of the same thing and is forwarded as an empty list too, which is why it is named separately from `content: []` below: a caller that sent an empty *block list* carried a block list, and that turn takes the empty-message line, while an empty *string* carried nothing at all.

The line a turn carries in place of its content is a *statement to the model*, and the
pipe reads these turns back. The image adapter derives its prompt from the transformed
body, so a turn that carried nothing reads there as that literal sentence. It must not be
taken for one: the adapter refuses before sending any turn whose only user text is that
line, exactly as it does for a blank message, so a turn where the person said nothing is
not sent as a generation. The video adapter reads the raw body rather than the transformed
one, so it never sees the line and must not grow this guard: a refactor that unified the
two adapters would silently re-open the image case.

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
- Open WebUI's own native `type: "image"` blocks, `{ "type": "image", "url": "..." }`, are accepted and
  converted to the same `input_image` block, with a `detail` on the native block honoured. Open WebUI's
  live chat path does not write this shape into a message, but the chat-completions gateway forwards
  `messages` verbatim and third-party filters and pipes write it directly, so it arrives. Before this it
  was passed through unchanged and the literal string `image` reached the provider as a content type.
- A block of a type the transformer does not recognise is still passed through unchanged.

### What is sent (important)

The pipe never writes an image the person attached to Open WebUI storage: not for a saved, channel or temporary chat, and not for any request of a turn (its first answer, a Regenerate, each further model answering it, a Continue, or Open WebUI's calls back after each round of tool calls). Every request sends the image as the message carries it:
- A **data URL** (`data:image/...;base64,...`) within `BASE64_MAX_SIZE_MB` is sent with the declared type and parameters preserved; one over the limit is not sent, the picture is skipped and the person sees `"Images: skipped N (…)."` on their latest message. Whitespace inside the payload is removed before sending, so an RFC 2045 line-wrapped body arrives at the provider as one line, and the payload is validated once, before it is sent: one that is not decodable as base64 — including a URL-safe-base64 payload, whose alphabet is not the one this data-URL form declares — is refused rather than forwarded, with the same `Images: skipped N (not decodable as base64)` status. A data URL carrying no payload at all — an empty body, or one that is only whitespace — is refused the same way, rather than forwarded as a header with nothing after it. The `;base64` marker is matched case-insensitively, as the data-URL standard requires, so `;BASE64,` is a spelling of the same thing and the URL is still forwarded with its own spelling intact. A token-free `data:` URL — the word `base64` in a payload, or a parameter that merely starts with those letters — is not base64 and is refused, in every spelling.
- A **remote URL** (`https://`) is downloaded (with retries/limits/SSRF protection) and its bytes are sent upstream as a `data:` URL; one that cannot be downloaded is forwarded as its link **unless the pipe's own address check refused the host**: a host that does not resolve, or resolves to a non-routable address, fails closed and is not sent, and the person sees `Images: skipped N (could not be fetched, so it was not sent).` A public link the pipe merely failed to *download* (404, timeout, slow host) is forwarded as before, so OpenRouter may fetch it. A failed download costs one further address check on top of the one the download already ran, so a picture the pipe cannot fetch is refused after two address checks, and a turn carrying N such pictures waits for up to 2 x N x `ADDRESS_CHECK_SECONDS` (5s each) of checking, one picture after another. A download that returns no bytes counts as a failed download, so it is forwarded as its link rather than shipped as a zero-byte `data:` URL. One larger than `BASE64_MAX_SIZE_MB` once downloaded is not sent. The media type is resolved **from the bytes** on this leg exactly as on the reuse leg, and the declaration only decides when the bytes identify nothing: a payload that settles on a non-image type is not inlined but refused, with `Images: skipped N (not identifiable as an image).` on the turn that carried it, and the picture is not sent. The size check is applied first, so a payload refused for being too large is reported as too large and never as unidentifiable. Plain `http://` is disabled by default and requires explicit allowlisting, and a picture refused for it is reported the same way as every other picture refusal: one `Images: skipped N (…)` line on the turn that carried it, naming both valves so the person can undo the refusal. A turn whose only content was that picture is not dropped either; it goes to the model as a `[An attached item was not sent: …]` placeholder naming the reason.
- An **Open WebUI file URL** (for example `/api/v1/files/...`) is streamed with the requester's access and inlined as a `data:` URL to avoid requiring OpenRouter to fetch from your Open WebUI host.

An image the pipe **reuses** from an earlier turn is inlined as a `data:` URL too. A remote picture a saved chat reuses is cached in the worker's memory for the life of the process so later turns fetch it once; in a temporary chat it is re-fetched per request and kept nowhere. Bytes that carry a recognisable image signature decide its media type, whatever the source declared; where they carry none, the declaration decides. That is one rule, not two: a remote picture attached to a turn and one reused from an earlier turn are typed by the same gate on the same terms, and a payload that settles on a non-image type is refused on either leg with the same `Images: skipped N (not identifiable as an image).` A reuse is subject to `BASE64_MAX_SIZE_MB` per picture, the same ceiling an attachment is; a picture over it is refused and named. The residual is the same on both legs: an `image/*` declaration the bytes do not corroborate is forwarded under the declaration, BMP and TIFF among them. A reuse is also dropped when the SSRF gate, re-run for it on every send, now refuses the URL, so tightening `ENABLE_SSRF_PROTECTION` or editing a host allowlist takes effect on the next turn rather than the next fetch. That re-run costs one address resolution and no download; a URL it then refuses is fetched once more through the already-gated path, which runs its own address check, so a refusal pays two resolutions. A resolution that does not finish inside the address-check budget reaches no verdict, so a slow nameserver never buys a permission. On this leg it also never costs a re-fetch: the entry survives a check that never finished and the picture is served from it, because the question here is not whether a fetch may start but whether the bytes already in hand are still permitted, and no answer is not a denial of them. Contention can cause that same absent verdict as a stall: the checks run on a private bounded pool, so a check that cannot start inside its budget — one queued behind other checks because the pool is full — reaches no verdict either, and only a check that completed and refused the address evicts the entry. A download has no bytes to keep, so on that path a check with no verdict still sends nothing. What the private pool buys is that only address checks can pay that price, never media, thumbnails, file reads or Open WebUI's own endpoints. Nothing has to be refused for the re-check to be paid for: a request reusing `MAX_INPUT_IMAGES_PER_REQUEST` pictures pays one blocking resolution per reused picture, one after another (5 by default, 20 at the ceiling). Otherwise a reuse is dropped only when the type settled on this way is not an image type - so a payload declaring an image type whose bytes the pipe does not recognise (BMP and TIFF among them) is forwarded under its declaration, and one the pipe could not fetch at all is dropped. An `http(s)` reuse that the pipe could not fetch is reported as "could not be fetched, so it was not sent" rather than "could not be fetched, so its type could not be established": the fetch fails before any type gate runs, so the type-gate wording no longer describes it, and the type-gate cause now covers non-`http(s)` schemes only. A dropped reuse is reported to the person with the same `Images: skipped N (…)` status as a refused attachment, on the turn that triggered it. So is a picture refused for being served over plain `http://`; the status line names the reason, and a turn whose only content was that picture goes to the model as a `[An attached item was not sent: …]` placeholder rather than as an empty turn. Nothing on this input path writes a file - images the model *generates* are stored separately, and that is an output-side behavior.

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
- A block with no `file_id`, no `file_data` and no `file_url` is dropped for the same reason and with the same status: a `filename` is a label, not a source, and the part would otherwise go out declaring only its type.
- A link the HTTP security policy refuses is dropped, and the person sees "Files: skipped …", the same as an oversized file; when the block also has a `file_id`, the refused field alone is dropped and the `file_id` is sent, with no status.
- An `https://` link in either field is sent unchanged for the provider to fetch; the pipe does not download it. Plain `http://` is disabled by default and requires explicit allowlisting: a cleartext link whose host is not allowlisted has its **whole block dropped**, and the person sees "Files: skipped …", the same as an oversized file.
- An Open WebUI file URL in either field becomes a `file_id`, which is read with the requester's access and sent inline as `file_data`.

---

## Audio (`input_audio` / `audio` → `input_audio`)

OpenRouter audio inputs require base64-encoded audio, and the pipe enforces that:

- Remote URLs (`http://` / `https://`) are rejected and replaced with an empty `input_audio` block.
- Data URLs (`data:audio/...;base64,...`) are accepted if valid and within size limits; one over `BASE64_MAX_SIZE_MB` is not sent at all, the block is dropped, the person is told in a status on their latest message, and it is logged as a size refusal rather than an encoding failure. "Valid" is enforced rather than assumed: the payload must be the base64 alphabet with RFC 2045 line wrapping and nothing else, so a payload of stray characters is refused instead of decoding to silence. The same parser serves pictures and audio, so the `;base64` marker is matched case-insensitively here too: `;BASE64,` is the same marker, spelled differently, and is accepted rather than refused as "Audio input must be base64-encoded audio data".
- Raw base64 strings are accepted if valid, and refused by size the same way.

The pipe passes a recognised audio format through unchanged, so `wav`, `mp3`, `flac`, `m4a`, `ogg`, `aiff`, `aac`, `pcm16` and `pcm24` all reach OpenRouter as themselves; anything unrecognised — including a `webm` container, which OpenRouter documents on neither endpoint — is normalized to `mp3`. Only native audio attachments sent directly are checked against `Direct responses audio format allowlist` before routing; a hand-built `input_audio` block is not.

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
- Data URLs (`data:video/...;base64,...`) are accepted only if their size is at or below `VIDEO_MAX_SIZE_MB`. A base64 payload's size is its decoded size; a token-free payload's is its own length, which is the actual size rather than an estimate.
- YouTube URLs are allowed (and may only work on certain model/provider combinations).
- Remote URLs (`https://` by default; `http://` only when allowlisted) that are not Open WebUI file URLs are checked by the SSRF guard when `ENABLE_SSRF_PROTECTION=True`.
- For **video** only, the test for an Open WebUI file URL is `names_an_owui_file_path`, and it is on the **path**, not on the spelling: any URL whose path names `/api/v1/files/<id>`, whatever its scheme, host or letter-case (`https://owui.example.com/api/v1/files/abc/content`, `//host/api/v1/files/abc/content`, `HTTPS://OWUI.EXAMPLE.COM/API/V1/FILES/abc/content`), hard-fails with `RequiredInternalFileError` instead of being forwarded. The classifier used everywhere else, `is_internal_file_url`, stays relative-only -- an absolute URL is external to it -- and widening it would change the picture, file and audio paths it gates, so the video site uses this second predicate. Because the pipe has no configured base URL for Open WebUI to compare a host against, the test is host-agnostic: a foreign host that happens to serve that path is refused as one, which is what Open WebUI's own image router does for the same situation. A URL `urlsplit` cannot parse is judged on its raw text, so a malformed one that names the path still hard-fails rather than being reported as a broken video and forwarded.

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
