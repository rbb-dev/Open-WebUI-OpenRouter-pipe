# OpenRouter Direct Uploads (bypass Open WebUI RAG for chat uploads)

Open WebUI supports uploading files into a chat. By default, Open WebUI often treats those uploads as **RAG / Knowledge** inputs (extracting text, embedding, retrieving, etc.).

The **OpenRouter Direct Uploads** integration is a toggleable Open WebUI *filter* that can instead forward eligible chat uploads to OpenRouter as **direct multimodal inputs** (files/audio/video), bypassing Open WebUI RAG for those uploads.

This is useful when:
- You want the **model** to directly “see” the uploaded content (PDFs for native document understanding, audio/video understanding, etc.).
- You do **not** want Open WebUI to embed or chunk the upload.
- You want per-chat control (single toggle) and per-user modality control (simple valves).

> **Scope note:** this filter applies to **chat uploads** (`files[]` in the Open WebUI request payload).  
> Image understanding is handled by the pipe’s normal multimodal intake (`image_url` → `input_image`). See: [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md).

---

## Quick start

### Admin (recommended defaults)
- Keep `AUTO_INSTALL_DIRECT_UPLOADS_FILTER=true` so the pipe installs/updates the companion filter in Open WebUI’s Functions DB.
- Keep `AUTO_ATTACH_DIRECT_UPLOADS_FILTER=true` so the toggle appears only on models where direct files/audio/video are supported.
- Configure the filter’s **admin valves** (size limits and allowlists) to match your environment and risk tolerance.

### User (per chat)
1. Enable the **Direct Uploads** toggle in the Integrations menu.

2. In the filter settings (“knobs”), enable one or more modality valves:
   - `DIRECT_FILES`
   - `DIRECT_AUDIO`
   - `DIRECT_VIDEO`
3. Upload supported files in the chat as usual.

---

## UI surfaces (where toggles/valves live)

There are **two layers** of control:

1) **Integrations menu switch (per chat)**  
   - Single on/off toggle: **Direct Uploads**

2) **User valves (per user; edited via filter settings)**  
   - `DIRECT_FILES`
   - `DIRECT_AUDIO`
   - `DIRECT_VIDEO`

Admins configure size limits and allowlists via the filter’s **admin valves** (see below).

---

## What gets diverted (and what does not)

When the filter is enabled and a user turns on one or more modality valves:

- **Diverted** (direct upload path):
  - Uploads that match the relevant **MIME allowlist** and **size limits**
  - Audio/video uploads only when their MIME/format matches the filter’s allowlists

- **Not diverted** (normal Open WebUI path, fail-open):
  - Uploads not allowlisted (example: `.docx`, `.xlsx` when your allowlist is only PDF/text)
  - Knowledge-base style uploads marked `legacy: true`
  - Anything missing an Open WebUI file `id`
  - Audio in a `webm` container, which OpenRouter documents on neither endpoint, so it stays on the normal Open WebUI path whatever the allowlists say

Fail-open is intentional: unsupported types should continue to behave like “normal Open WebUI” (RAG/Knowledge) rather than breaking chat uploads.

---

## Model compatibility (how we detect "supports files/audio/video")

This integration does **not** rely on Open WebUI's `file_upload` capability flag (deployments often enable `file_upload` broadly for UI reasons).

Instead, the pipe maintains an internal capability map in each model's metadata:

- `model.info.meta.openrouter_pipe.capabilities.file_input`
- `model.info.meta.openrouter_pipe.capabilities.audio_input`
- `model.info.meta.openrouter_pipe.capabilities.video_input`

These are used for:
- Deciding whether to auto-attach the filter (`AUTO_ATTACH_DIRECT_UPLOADS_FILTER`)
- Validating user toggles at runtime

### File input: enabled by default (blocklist approach)

**Extensive testing revealed that most models support direct file uploads**, even though OpenRouter's model catalog (`architecture.input_modalities`) only declares ~53 models as supporting "file" input.

| Metric | Value |
|--------|-------|
| Models tested | 287 |
| Successful file processing | 239 (83.3%) |
| Models needing blocklist | 29 |

Rather than gating on incomplete upstream metadata, **`file_input` is now enabled by default** for all models except those in a known-incompatible blocklist. The blocklist names model *families*, not individual catalog rows: an entry blocks every variant form of that model, because a variant is the same weights behind a different provider.

The blocklist (`open_webui_openrouter_pipe/models/blocklists.py`) includes:
- Models that explicitly reject file input (HTTP 400)
- Guard/classifier models not designed for chat
- Models that claim they cannot process files
- Models with broken/empty responses when given files

One entry therefore covers the model under every spelling OpenRouter publishes it as: a routing suffix (`:nitro`, `:floor`, `:exacto`, `:online`), a catalog suffix (`:free`, `:batch`, `:thinking`, `:extended`), any combination of them, and a dated `-YYYY-MM-DD` snapshot. A `:free` twin of a blocklisted model is the same model, so it loses its Direct Uploads switch too. A `~…-latest` **alias row** counts as another spelling: an alias resolves through its `alias_target` to the model it names, and a blocklisted model withholds Direct Uploads from that alias too. (Every other capability of the alias row is still answered from the alias row itself — see [Zero Data Retention](openrouter_zdr.md), which draws the same one hop for the same reason.) No shipped catalogue publishes such an alias today; the hop is here so the blocklist is exhaustive over spellings rather than over today's snapshot.

The blocklist also covers the capability Open WebUI acts on directly. A blocklisted model publishes `file_upload: false` alongside the missing `file_input`, so Open WebUI does not offer it the chat file tools — `view_file`, `query_chat_files`, `grep_chat_files` and `list_chat_files` — which it injects when a model accepts uploads but its Workspace row has `file_context` off. The cover is live only where that row's `file_context` is off: Open WebUI treats an absent capability as on, so on a stock install the model row is left alone and the tools are offered as they are for any other model.

### Audio and video input

For `audio_input` and `video_input`, the pipe still relies on OpenRouter's declared `architecture.input_modalities`, as these modalities are less commonly supported and require explicit provider enablement

---

## Data flow (filter → metadata marker → pipe injection)

### 1) Filter diversion (inlet)

When the filter diverts an upload, it:
- Removes the diverted items from the request `files[]` list **and** from `metadata.files`, so Open WebUI won’t treat them as knowledge inputs. The video filter honours the removal: it is the one pipe filter that reads attachments, and it will not put a diverted file back into the request or into `video_generation.input_references`.
- Records lightweight references (file IDs + hints) under:
  - `__metadata__["openrouter_pipe"]["direct_uploads"]`

### Interaction with Open WebUI “File Context” (OWUI 0.7.x+)

Open WebUI has a per-model capability toggle called **File Context** (Models → Advanced Settings → Capabilities). When enabled, Open WebUI will:
- Extract content from uploaded files / knowledge items
- Run retrieval as needed
- Inject the resulting context into the conversation (prompt) before the request reaches the model provider

This is great for classic OWUI RAG flows, but it is the opposite of what “Direct Uploads” is for: Direct Uploads is meant to forward the original file bytes to OpenRouter as **real multimodal inputs** (documents/audio/video), without OWUI pre-extracting and injecting text.

#### The important implementation detail (why we touch both `files[]` and `metadata.files`)

In OWUI, the File Context handler reads **`metadata.files`** (not `files[]`) when deciding what to process for injection. However, OWUI also rebuilds `metadata.files` from the request `files[]` after inlet filters have run.

That means: if a diversion filter only edits `metadata.files`, OWUI can later overwrite it from the unmodified `files[]`, and the File Context injector may still run on the diverted uploads.

To make Direct Uploads behave consistently, the companion filter therefore:
- Removes diverted items from **both** `files[]` and `metadata.files`
- Leaves **retained** (non-diverted) items in place so OWUI can still handle them normally

#### What you should expect (behavior matrix)

- **Direct Uploads OFF**
  - **File Context ON** → OWUI may extract/retrieve and inject file context into messages (classic OWUI RAG behavior).
  - **File Context OFF** → OWUI skips automatic file extraction/injection; only raw attachment metadata remains.

- **Direct Uploads ON**
  - Diverted uploads are removed from `files[]`/`metadata.files`, so OWUI File Context will not inject their extracted content (even if File Context is enabled for the model).
  - Diverted uploads are forwarded to OpenRouter as direct inputs by the pipe (see next sections).
  - A diverted upload that Open WebUI has already processed — a document its RAG pass extracted, a voice note its speech-to-text pass transcribed — is forwarded as the **stored file's own bytes** under its recorded type, and the extraction is never substituted for them. The text is served only for a record Open WebUI stored no file for.
  - Any non-diverted uploads (unsupported type, allowlist mismatch, user valve off, etc.) remain on the normal OWUI path and may still be processed by File Context when enabled.

### 2) Pipe injection (the outer request and every internal-Fusion stage)

The pipe reads those references and:
- Loads bytes from Open WebUI storage by file `id` — once per turn, and reuses those bytes for every stage: the outer request, each internal-Fusion panel member, the judge, the judge's repair pass and the synthesis. The `data:` URL rendered from those bytes is memoised with them, one per attachment. The format sniffing and the Direct Audio Format Allowlist check are *not* memoised: they run on every stage.
- Injects the attachment(s) into the **last user message** in the outgoing request, once per stage (the outer request, each internal-Fusion panel member, the judge, the judge's repair pass and the synthesis). The duplicate check that keeps an already-sent attachment from being injected twice is by block identity for the three kinds the pipe itself injects (`file`, `input_audio`, `video_url`) -- a file id and filename, a format with the payload's length and hash, a url with the url's length and hash -- and never by walking an attachment's bytes; a block of any other kind on that message, an inline picture in particular, contributes no key at all, because the pipe appends none and the question cannot be answered yes.

For safety/portability, OpenRouter never receives an internal Open WebUI URL, unconditionally. Any internal `/api/v1/files/<id>/content` reference is inlined to a `data:<mime>;base64,...` payload before sending upstream -- and "internal" is decided on the **path**, not on the spelling, so an absolute form such as `https://your-owui-host/api/v1/files/<id>/content` (with a userinfo, a `?token=`, any letter-case, or any other encoding) is the same reference and is resolved from local storage rather than fetched or forwarded. A reference that names the path but yields no readable id is refused with `RequiredInternalFileError` before anything is sent, so no such URL leaves the worker in any form. The `<mime>` there is the recorded type of the stored file where there is one, and otherwise one derived from the file's own name, lowercased with its parameters stripped and checked against the media-type grammar: a recorded type that is not a media type produces no `data:` head at all, and the reference is refused with the `Direct Upload Issue` card rather than forwarded under a defaulted or truncated type.

### 3) Tasks do not receive direct uploads

Open WebUI may run “task model” requests during chat flows (query/title/tags/follow-up generation — for example when Web Search is enabled).

To avoid expensive prompt bloat and unintended content leakage, the pipe:
- Does **not** inject direct uploads into task requests
- Still allows the subsequent **main chat** request to receive the attachments normally

---

## Endpoint selection: `/responses` vs `/chat/completions`

OpenRouter exposes multiple OpenAI-compatible endpoints. For direct uploads, the pipe selects an endpoint based on what must be sent.

### Summary rules

- **Direct files / documents**
  - On **`/responses`**, the pipe emits Responses-style `type:"input_file"` blocks.
  - On **`/chat/completions`**, the pipe emits Chat-style `type:"file"` blocks and inlines the bytes as `file.file_data` (data URL).

- **Direct video** → requires **`/chat/completions`** (current implementation). The `data:` head is grammar-checked before it is built: a declared type that is not a media type is refused and named on the `Direct Upload Issue` card, not sent with a malformed head.

- **Direct audio**
  - Eligible for **`/responses`** when the resolved audio format (sniffed from the bytes, else the declared one) is in `DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST` (default: `wav,mp3`)
  - Otherwise routes to **`/chat/completions`**

The pipe does not trust upstream file metadata: it re-sniffs audio containers and then applies `DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST` for routing. The sniff is deliberately narrow where it claims a container of its own — an ISO base-media box is only labelled `m4a` when its major brand is one that names MPEG-4 audio (`M4A `, `F4A `, `M4B `). Any other brand, known or not, is left unlabelled, exactly like a header the pipe cannot read, and the declared format governs. Nothing is refused for that: the gate is unchanged, and a resolved format outside the nine the pipe sends natively is still refused (unless the operator has listed it in `DIRECT_AUDIO_FORMAT_ALLOWLIST`) and named on the `Direct Upload Issue` card.

### Conflict behavior (no silent degradation)

If a request includes any direct uploads that require `/chat/completions` (video, or audio formats not eligible for `/responses`), the pipe routes the whole request to `/chat/completions` and forwards direct files using `type:"file"` blocks there.

The only hard stop is when an **admin forces `/responses`** for a model but the request requires `/chat/completions`. In that case, the pipe:
- Stops the request
- Emits a templated, user-friendly error (`ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE`)

---

## Limits and allowlists (admin configuration)

Direct uploads are intentionally gated. There are **two kinds** of limits:

1) **Filter valves (admin)**
   - Per-modality max size
   - Total max size across diverted attachments
   - MIME and format allowlists

2) **Pipe safety limits (admin)**
   - The pipe must inline internal Open WebUI file references into base64 data URLs
   - Inlining is bounded by the pipe valve `BASE64_MAX_SIZE_MB`

If a diverted upload exceeds limits, the request fails with a clear error instead of silently falling back. An attachment is measured only when its own modality’s user valve is on; otherwise it is handed straight back to Open WebUI.

---

## Valves reference

### Pipe valves (admin)

These are configured on the **pipe** function in Open WebUI (Admin → Functions → pipe → Valves).

| Valve | Default (verified) | Purpose / notes |
| --- | --- | --- |
| `AUTO_ATTACH_DIRECT_UPLOADS_FILTER` | `True` | Auto-enable the OpenRouter Direct Uploads filter in each compatible model’s Advanced Settings (`filterIds`), so the switch appears only where it can work. A pass that could not install the filter, because Open WebUI refused the write, is not a decision to detach: the switch stays exactly where it was and the install is tried again at the next catalog fetch. |
| `AUTO_INSTALL_DIRECT_UPLOADS_FILTER` | `True` | Auto-install / auto-update the companion filter function into Open WebUI’s Functions DB (recommended with auto-attach). Turning this off retires the rows the pipe installed for it - switched off, not deleted, so their settings survive - and turning it back on brings them back; a copy an admin installed by hand carries no such record and is left alone. A row an earlier install of this pipe wrote — the pipe function was renamed or re-created, so its record names an id Open WebUI no longer loads as a pipe — is retired too. |
| `BASE64_MAX_SIZE_MB` | `50` | Upper bound for inlining Open WebUI internal file URLs into base64 data URLs. |

For the full list, see: [Valves & Configuration Atlas](valves_and_configuration_atlas.md).

### Companion filter valves (admin)

These are configured on the **OpenRouter Direct Uploads** filter function (Admin → Functions → filter → Valves).

| Valve | Default (verified) | Purpose / notes |
| --- | --- | --- |
| `DIRECT_TOTAL_PAYLOAD_MAX_MB` | `50` | Maximum total size (MB) across all diverted direct uploads in a single request. |
| `DIRECT_FILE_MAX_UPLOAD_SIZE_MB` | `50` | Maximum size (MB) for a single diverted direct file upload. |
| `DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB` | `25` | Maximum size (MB) for a single diverted direct audio upload. |
| `DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB` | `20` | Maximum size (MB) for a single diverted direct video upload. |
| `DIRECT_FILE_MIME_ALLOWLIST` | `application/pdf,text/plain,text/markdown,application/json,text/csv` | Comma-separated MIME allowlist for diverted direct generic files. Non-allowlisted types are fail-open (left on normal OWUI RAG/Knowledge path). The pattern is matched with `fnmatch` against the declared type, so a wildcard admits declared values that are not media types at all; an attachment whose type is not a media type is refused before the request is sent. |
| `DIRECT_AUDIO_MIME_ALLOWLIST` | `audio/*` | Comma-separated MIME allowlist for diverted direct audio files. |
| `DIRECT_VIDEO_MIME_ALLOWLIST` | `video/mp4,video/mpeg,video/quicktime,video/webm` | Comma-separated MIME allowlist for diverted direct video files. The pattern is matched with `fnmatch` against the declared type, so a wildcard admits declared values that are not media types at all; such an attachment is not sent at all -- it is refused before the request leaves the pipe and the turn carries the `Direct Upload Issue` card. |
| `DIRECT_AUDIO_FORMAT_ALLOWLIST` | `wav,mp3,aiff,aac,ogg,flac,m4a,pcm16,pcm24` | Comma-separated audio format allowlist (derived from filename/MIME). Listing a format lets a direct audio upload through even when it is outside the nine the pipe sends natively; naming an undocumented one here diverts the clip onto the pipe's path, and the pipe then leaves it out of the request rather than relabelling it `mp3` -- the clip comes back with a note on the chat's status line saying an audio clip was in a format the pipe will not rename. So listing a format the pipe does not send does not make it sendable; it makes the refusal visible. Only the formats listed here are diverted; a `webm` container is never diverted, listed or not, and stays on Open WebUI's path; a cleared value diverts no audio at all. |
| `DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST` | `wav,mp3` | Comma-separated audio formats eligible for `/responses` `input_audio.format`. |

A stored value from an older version of the pipe that no longer fits its field falls back to that field's default rather than failing the request, and every other stored value on the same row is kept.

### Companion filter user valves (per-user)

These appear in the filter’s user-facing “knobs” UI and control what gets diverted natively.

| Valve | Default (verified) | Purpose / notes |
| --- | --- | --- |
| `DIRECT_FILES` | `False` | Divert eligible chat file uploads and forward them as direct document inputs. |
| `DIRECT_AUDIO` | `False` | Divert eligible audio uploads and forward them as direct audio inputs. |
| `DIRECT_VIDEO` | `False` | Divert eligible video uploads and forward them as direct video inputs (routes via `/chat/completions`). |
| `DIRECT_PDF_PARSER` | `"Native"` | Selects the OpenRouter PDF parsing engine for PDF uploads (requires `DIRECT_FILES` enabled). |

---

## PDF parser selection

Direct PDF uploads can use OpenRouter’s PDF processing engines. The **PDF Parser** user valve controls which engine is requested.

Options (UI label → OpenRouter engine ID):
- `Native` → `native`
- `PDF Text` → `pdf-text`
- `Mistral OCR` → `mistral-ocr`

Notes:
- The parser choice is applied only when a **PDF** upload is diverted to OpenRouter.
- `DIRECT_FILES` must be enabled; otherwise PDFs stay on the normal OWUI RAG path and this valve has no effect.
- See OpenRouter’s PDF Inputs guide for details: https://openrouter.ai/docs/guides/overview/multimodal/pdfs

## Troubleshooting

### “Direct Upload Issue” error

This is emitted by the pipe (not OpenRouter) when direct uploads can’t be applied safely.

Common causes:
- File exceeds size limits (the size checks apply only to a modality whose user valve is on)
- The uploaded file's dict carries no `size`
- The MIME type is allowlisted but the model or provider still rejects it (see "Provider says unsupported MIME/type" below)
- An attachment declares a type that is not a media type, so the pipe will not forward it
- Open WebUI storage object could not be loaded by ID
- Admin enforced an incompatible endpoint override (forced `/responses` but the request requires `/chat/completions`)

### “Provider says unsupported MIME/type”

This comes from the upstream provider/model. Even if your allowlist permits a type, a specific provider may reject it.

Best practice:
- Keep allowlists conservative (PDF + plain text is a good starting point for “documents”).
- Let non-allowlisted types fall back to normal Open WebUI RAG/Knowledge behavior.

### “Where did my upload go?”

If you enable direct uploads and your upload disappears from the RAG/Knowledge path, that is expected:
- The filter removes diverted items from `files[]` so Open WebUI doesn’t treat them as knowledge inputs.
- The pipe injects the upload into the model request as a direct multimodal block.

If you don’t see it reaching the model:
- Confirm the filter toggle is enabled in the Integrations menu for that chat.
- Confirm the relevant user valve is enabled (Files/Audio/Video).
- Confirm the MIME type is allowlisted and the file is within size limits.

If direct uploads are enabled but the selected model does not support a required modality (file/audio/video), the filter will **fail open**:
- The upload stays on the normal Open WebUI path (RAG/Knowledge), and
- The pipe emits a warning notification that direct uploads were not applied for those attachments.

The capability check is what runs first, so these are the five paths on which an attachment is handed back instead of measured; a missing `size` is not one of them, because an entry that was never going to be diverted is never measured, while one that is diverted without a usable size is refused with the `Direct Upload Issue` card:
- **valve off** — the modality’s `DIRECT_*` user valve is off,
- **modality unsupported by the model**,
- **MIME not allowlisted** for that modality, and
- **audio only** — the audio format could not be inferred from the name or content type, and
- **audio `webm`** — the container is one OpenRouter documents on neither endpoint, so it is not diverted even when the allowlist names it.

### Debug logging (useful strings)

When `OPENAI` log level is set to debug in Open WebUI, the pipe logs:
- `Injecting direct uploads into chat request ...`
- `Ignoring direct uploads for task request ...`
