# OpenRouter Video Intent Classifier

The pipe runs a small classifier model before each video generation to figure out what the user actually wants. Without it, the OpenRouter `/videos` endpoint is single-shot and stateless — every request only knows about the latest user message, so a follow-up like "change colour to black" produces an unrelated video instead of a recoloured version of the previous one.

## Why this exists

The OpenRouter `/videos` endpoint takes `prompt`, optional `frame_images`, and optional `input_references`, then returns a video. It does not know about prior turns in a chat. If the user's previous turn produced a video of a cat and they then say "change colour to black", the model has no idea what the cat looked like — it just makes a new video about a black cat (or a black anything).

Other UIs work around this by automatically attaching a frame from the prior video as a reference. The pipe does the same thing now: a small task model reads the chat history, the latest message, and any attachments, then decides what visual reference (if any) to wire into the request before it's submitted.

## What it costs, and how the classifier is found

Every video follow-up that is not short-circuited makes **one small task-model call**. It is bounded by `VIDEO_INTENT_TIMEOUT_S` (default `8`) and is skipped entirely on a fresh chat with no prior turns and no attachments (`VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT`, default `True`), so the call is not wasted where there is nothing to classify against. It is a billable call at whatever the chosen Task Model's OpenRouter rate is, which is why the timeout, the skip valve and `VIDEO_INTENT_MAX_CALLS_PER_CHAT` all exist.

The classifier's model id is read from Open WebUI's **config table** — `task.model.default` and `task.model.external`, the two settings behind Settings → Tasks. A host that populates the older `app.state.config.TASK_MODEL` / `TASK_MODEL_EXTERNAL` shape instead is still read, as a fallback. This matters because Open WebUI 0.11.4 assigns no `app.state.config` at all, so a resolver that read only that attribute answered nothing on a stock host and every video follow-up degraded open with the classifier silently inert — no warning, just a plain text-to-video with no cross-turn context.

When the task model keeps failing, `classifier_failed` is set, the **60-second** intent breaker opens (so the next turns skip the call instead of retrying quietly forever), and a warning plus a toast say so. See "Failure modes" below.

## How it works

```
user turn ─► VideoGenerationAdapter.generate
                │
                ▼
            short-circuits (help / resume / empty / first-turn-no-attachment)
                │
                ▼
            VIDEO_INTENT_ENABLED?
                │
            ┌───┴───┐
           yes      no ─► send only latest user message (no context, no questions)
            │
            ▼
       resolve_intent (task model + JSON schema)
                │
            ┌───┴────────────────────┐
            ▼                        ▼
       clarification needed?    intent + frame_plan
            │                        │
       emit question            materialise frame_plan
       (return)                 (extract frames, upload thumbnails)
                                     │
                                     ▼
                                 inject into video_meta["frame_images"]
                                     │
                                     ▼
                                 render Intent Disclosure Block
                                     │
                                     ▼
                                 submit to /videos with frames + cleaned prompt
```

The classifier returns one of five **intents**:

- `text_to_video` — fresh generation; no prior context wired
- `image_to_video` — user-attached image as anchor
- `modify_prior_video` — re-render the previous video with a change ("make it black")
- `continue_prior_video` — temporal extension ("continue", "what happens next")
- `ambiguous` — needs a clarifying question

For every non-trivial intent, the classifier produces a `frame_plan` array (max 4 entries). Each entry says: *here's the source* (uploaded attachment / prior video first frame / prior video last frame / prior video at timestamp T), *here's the target* (first_frame / last_frame / input_reference), and *here's the index*. The pipe extracts the actual frame, uploads it as an OWUI image, and injects it into the request.

The `index` of an uploaded attachment is its position in the user's attachment list — the order they appear in the chat, counting pictures, clips and audio alike. Every attached picture is listed, whether or not the Frames dropdown claimed it as a keyframe: a turn with three pictures and `first_last` selected still reports indices 0, 1 and 2, not 0 and 2. An `index` that names a picture the dropdown did not claim is promoted into the frame slot the entry asks for.

The `attachments` array in the classifier's payload is that list: one entry per attachment the turn carries, each with `index` (its position, 0-based and gap-free), `kind` (`image` / `video` / `other`), `mime_type`, `id`, `name` and `size`, ordered by `index`. A picture the Frames dropdown claimed as a keyframe and a picture it demoted to a reference both appear once, each at the position the user attached it, and neither is listed twice because it occupies both channels. What the pipe does not send is the bytes and not the role the picture plays downstream — the classifier is told *what* the user attached, and it is the classifier's `frame_plan` that says what to do with it. Under `frame_mode="none"` the array is empty by design: the dropdown sends the pictures nowhere, so there is nothing for the classifier to refer to.

## Intent Disclosure Block

When `frame_plan` is non-empty, the assistant message includes an **Intent Disclosure Block** rendered before the video appears:

```
🎬 Modifying previous video — using its first frame as anchor.

![ref](/api/v1/files/THUMB/content)

Prompt: "a black cat walking through tall grass"

[generated video appears below]
```

The block is wrapped in hidden markdown markers (`[openrouter:v1:intent_block_start]: #` … `[openrouter:v1:intent_block_end]: #`) so the next turn's classifier can strip it before parsing — the thumbnail won't be mistaken for a fresh image input.

If the user wants to abort because they see something wrong (e.g. the wrong reference video was picked up), they hit Open WebUI's stop button and the `/videos` call is cancelled before the paid request goes out.

## When a model can't visually modify a previous video

The classifier's `modify_prior_video` intent (triggered by prompts like "change colour to black") emits a `frame_plan` entry with `target="input_reference"` — meaning: *use the prior frame as a style/content reference, not as a hard pixel anchor.* This is the only target type that lets the underlying video model repaint or transform the frame.

Most current OpenRouter video models (Seedance, Veo, Kling, Wan, …) only support `first_frame` and `last_frame` — hard anchors that lock the output to the exact input pixels. None of them currently advertise `input_reference` support in the catalog.

When the selected model cannot honor `input_reference`, the pipe **does not** stop to ask the user — it degrades open, and the paid `/videos` call still proceeds:

- The validator drops the `input_reference` frame entry (recorded as the `dropped_input_reference_no_frame_support_for_model` downgrade). If that leaves a `modify_prior_video` intent with no usable frames, the intent is downgraded to `text_to_video` (`modify_prior_video_dropped_to_text_no_frame_support`) and the request runs as a plain text-to-video generation from the classifier's rewritten prompt.
- There is no catalog branch to take. The catalog's `supported_frame_images` is a fixed two-value enum (`first_frame`, `last_frame`), so it cannot name `input_reference` and that branch is never taken. Where a model entry does carry `input_modalities`, the pipe withholds a reference whose kind that model does not declare, with a notice; where it does not, the pipe sends the reference and cannot know whether the model read it.

Either way there is no confirmation prompt and no `1`/`2` question; the downgrade is surfaced only in the disclosure block and the telemetry `downgrades` list.

## Frame extraction at arbitrary timestamps

The classifier supports `prior_video_at_timestamp` with `timestamp_seconds`. Examples the system prompt recognises:

- *"use the frame at 5 seconds"*
- *"from the 5-second mark"*
- *"at 0:30"*

The pipe validates the requested timestamp against how long the previous video's **picture** runs, which is the video stream's own length where the host can measure it and the container's length otherwise — a clip muxed with a longer audio bed is as long as the audio, not as long as the frames. A length the probe could not measure at all leaves the pipe unable to call a request an overshoot, so the seek is made as asked and a frame is substituted only if the seek itself comes back empty; the disclosure block then says the moment could not be read rather than that it was past the end. When the requested time is measured as past the end, the pipe downgrades to the frame `Reused frame position` selected and surfaces a note in the disclosure block:

> ⚠️ The requested time was past the end of the previous video; used its last frame instead.

A seek that misses the last decodable frame is retried against the end of the file with a wider window (1s, then 5s, then 30s), and a damaged tail is retried against wider windows before the frame is given up on, so a tail of up to 30s still yields a frame at up to three times the normal extraction time. A window that hits damage inside an otherwise readable file widens before the frame is given up on; an input ffmpeg cannot open at all fails on the first window, because no wider window will read it either.

Before a `first_frame` extraction decodes anything, the pipe reads the source's declared frame size from the container header and refuses to decode a source over the 25-megapixel pixel budget; the frame is then re-acquired through ffmpeg at the 1920-wide ceiling. That header read costs roughly one extra container open on every `first_frame` extraction, and it is the difference between allocating a few kilobytes and allocating the whole decoded frame (180 MB on a 10000×3000 source) for a source the budget already refuses.

## Configuration valves (admin)

All admin-scoped on the global `Valves` model. User-tunable per-chat versions of four of these are also exposed on each video model's filter UserValves (see "User-tunable settings" below).

| Valve | Type | Default | Purpose |
|---|---|---|---|
| `VIDEO_INTENT_ENABLED` | `bool` | `True` | Master switch. When False, the classifier is bypassed entirely; only the latest user message is sent to the video model. |
| `VIDEO_INTENT_TASK_MODEL_MODE` | `internal` / `external` | `external` | Which of Open WebUI's two Task Models to use as the classifier, as configured in Open WebUI's admin Task Model settings. `internal` reads the local-model setting; `external` reads the API-model one. |
| `VIDEO_INTENT_TASK_MODEL_FALLBACK` | `none` / `other_task_model` | `other_task_model` | Failure fallback strategy. `none` returns only the primary task model; `other_task_model` also tries the other (internal/external) Task Model. With neither configured the classifier is skipped, with one warning logged per chat. |
| `VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT` | `bool` | `True` | Skip the classifier when the chat has no prior turns and no attachments — there is nothing to classify against, so the call is wasted. Turn off if you want clarifying questions on first-turn ambiguous prompts. |
| `VIDEO_INTENT_MAX_CLARIFICATIONS` | `0`–`3` | `1` | Per-session cap on consecutive clarifying questions. `0` disables the clarification loop entirely. |
| `VIDEO_INTENT_FRAME_EXTRACTION_INDEX` | `first` / `last` | `last` | Default frame to extract from a prior video when the requested frame is unavailable. |
| `VIDEO_INTENT_TIMEOUT_S` | `int` | `8` | Hard timeout (seconds) on the classifier call. On breach, the pipe falls back to sending only the latest user message — the paid video request still proceeds. |
| `VIDEO_INTENT_CONFIRM_MODE` | `always` / `on_reference` / `low_confidence` / `never` | `on_reference` | When to surface the confirmation footer. `on_reference` confirms only when a prior video's frame is reused or more than one frame is combined; a lone attached image does not trigger it. |
| `VIDEO_INTENT_MAX_CALLS_PER_CHAT` | `int` | `0` (unlimited) | Cost guard. `0` = unlimited. Admin sets a positive integer to enforce a per-chat ceiling. |
| `VIDEO_INTENT_MAX_CALLS_PER_USER_DAY` | `int` | `0` (unlimited) | Cost guard. `0` = unlimited. Admin sets a positive integer to enforce a per-user-per-day ceiling. Tallying is O(1) per classifier call however many distinct users have already called today: yesterday's keys are dropped once per day, not once per call. |
| `VIDEO_INTENT_LOG_DECISIONS` | `bool` | `False` | Log the per-turn classification summary (intent, confidence, language, frame counts, latency, fallback/failure flags, hashed chat id) at INFO instead of DEBUG; always written, level-only. Excludes the verbatim prompt and the model's free-text reason. |

## User-tunable settings (per-model filter UserValves)

When admin `VIDEO_INTENT_ENABLED=True`, each video model's companion filter exposes four UserValves so individual users can override the admin defaults for their own chats. They appear in the OWUI filter settings panel alongside the existing video knobs (`VIDEO_DURATION`, `VIDEO_ASPECT_RATIO`, etc.).

| User valve | Shown as | Effect |
|---|---|---|
| `VIDEO_INTENT_ENABLED` | `Reuse previous videos` | Per-user opt-out. When off, this user's video chats bypass the classifier even though it's on globally. |
| `VIDEO_INTENT_MAX_CLARIFICATIONS` | `Clarifying question limit` | Override the cap on consecutive clarifying questions for this user. |
| `VIDEO_INTENT_FRAME_EXTRACTION_INDEX` | `Which frame to use from previous video` | Pick which frame the user prefers when the request is ambiguous about first vs last. |
| `VIDEO_INTENT_CONFIRM_MODE` | `Show what was reused` | Control when the disclosure footer appears for this user (always / on reference / on low confidence / never). |

"Shown as" is the label above the control in the filter settings panel and the
name the model's `help` panel lists it under; the field name is what the
generated filter source calls it.

When admin `VIDEO_INTENT_ENABLED=False`, the four user fields do not appear in the filter UI at all — the next time the pipe rebuilds filters (i.e. on the next `pipes()` refresh) the filter source is regenerated without them. The switch is enforced at request time regardless: it is re-read live on every request and treated as a floor, so a filter row installed while it was on (which keeps pushing `video_intent` into request metadata) cannot re-enable a classifier the operator has switched off.

## Anti-overasking guardrails

The classifier is biased to act, not ask. Clarifying questions only fire when:

- The user references "it"/"that"/"the previous one" but **multiple prior videos** exist with no positional cue, AND
- The candidate options would produce **meaningfully different outputs**, AND
- The user has not explicitly opted out (e.g., "just do it", "your call", "you decide").

The classifier obeys explicit wiring instructions ("use the previous video as first frame", "use the frame at 5 seconds") without asking. One-word continuations like "more", "again", "encore" trigger continue-prior-video with a default last-frame anchor, no question.

## Telemetry

When `VIDEO_INTENT_LOG_DECISIONS=True`, every turn on which the classifier runs emits a structured INFO log line (bypassed turns emit nothing). Keys:

| Key | Meaning |
|---|---|
| `intent_mode` | `text2video` / `image2video_attached` / `image2video_priorframe` / `clarify` (or the raw `intent` value when a frame plan has neither prior-video nor uploaded-attachment sources) |
| `intent` | `text_to_video` / `image_to_video` / `modify_prior_video` / `continue_prior_video` / `ambiguous` |
| `confidence` | `high` / `medium` / `low` |
| `language` | classifier-detected language tag (`en`, `it`, …) |
| `frame_plan_size` | count of entries in the (post-validation) frame plan |
| `clarification_emitted` | bool |
| `task_model_latency_ms` | classifier latency |
| `task_model_fallback_triggered` | bool |
| `classifier_failed` | bool — TRUE when the task-model orchestration itself failed (timeout / parse error / auth / quota) and the pipe returned a synthesized fallback result. Operators grep this to find degrade-open turns. |
| `failure_reason` | string — `"<ExceptionClass>: <message>"` when `classifier_failed=true`, otherwise empty. |
| `prior_video_frame_extracted` | bool — TRUE iff the pipe actually extracted at least one prior-video frame; FALSE when the classifier asked for one but the request was blocked (e.g. by the modify-fallback gate) or extraction failed |
| `prior_video_frames_extracted_count` | int — number of prior-video frames successfully extracted and uploaded |
| `prior_video_frames_requested_count` | int — number of `prior_video_*` source entries in the classifier's frame_plan (what was asked for; distinct from what actually ran) |
| `frames_retargeted_count` | int — number of uploaded-attachment frames whose `kind` was rewritten per the classifier's instruction (e.g. when the user said "use this as the last frame" and the auto-attach filter had defaulted it to `first_frame`). Normal success behaviour; not a downgrade. |
| `downgrades_count` | number of validator downgrades (capability mismatches, timestamp overshoots, dropped entries, etc.) |
| `discarded_plan` | bool — true when the validator threw out the classifier's plan entirely (e.g. due to explicit-attachment precedence) |

## Failure modes

Every failure path in the classifier returns a fallback result equivalent to "no classifier ran": `intent=text_to_video`, `frame_plan=[]`, `prompt=<latest user text>`. The video call still fires; it just doesn't carry cross-turn context. Failures handled:

- **Task model returns invalid JSON** → one corrective retry per candidate; on second failure, fall through to the next candidate; if all candidates fail, fallback.
- **Task model timeout** (>`VIDEO_INTENT_TIMEOUT_S`) → fallback.
- **Frame extraction fails** (corrupt prior video, unsupported codec) → drop that frame_plan entry, append a downgrade note to the disclosure block, continue with other entries; if every entry fails, send text-only. A clip whose tail is damaged but which still decodes is not this case: the end-seek ladder widens its window until one reads, and the frame that comes back is labelled as the nearest decodable one, not as a frame from past the end.
- **Thumbnail upload fails** → disclosure block omits that thumbnail and says so on a ⚠️ line ("A preview picture for this frame could not be stored."); the `frame_images` entry still ships. A thumbnail that cannot be *made* is recorded the same way, with "…could not be made." Every per-entry outcome carries the plan position as well as the source index, so two entries asked of the same prior video record two separate codes and read as two separate ⚠️ lines rather than one duplicated.
- **User cancels mid-classification** → cancellation propagates up; `/videos` is never submitted.
- **Model can't honor `input_reference` for modify intent** → the validator drops the reference frame and downgrades the intent to `text_to_video`; the paid call proceeds as text-to-video with no confirmation prompt. See "When a model can't visually modify a previous video" above.

The **first** classifier infrastructure failure per chat surfaces a notification toast: *"Intent inference unavailable; using simple text-to-video."* Subsequent failures within the same chat are silent (logged at DEBUG). The rule covers both failure branches — a classifier that reported failure, and a classifier call that raised — and holds whether or not the emit itself succeeded.

The "already notified" record is a bounded window of the most recent **300** failing chats, oldest evicted first. It is not valve-gated — it fills on the shipped configuration — and the bound is why a chat that falls out of the window may be shown the toast a second time. In practice eviction is not expected within months of continuous, total classifier failure: a failure arms a 60-second process-global breaker, so the window gains at most one entry per minute, i.e. roughly five hours of unbroken failure just to fill it and five more before the first eviction. Eviction happens on the add path only, so the classifier hot path's membership check stays constant-time. The real cost of the bound is that the attribute is an insertion-ordered mapping rather than a `set`.

**Diagnostic log lines** for the toast emission path (search these when the toast doesn't appear as expected):

- `video_intent classifier_failed=True; reason=<...>; breaker tripped` — WARNING, fires every time a classifier infrastructure failure is detected.
- `first-failure toast emitted (chat_key=<...>)` — INFO, confirms the toast was sent to the OWUI event emitter.
- `first-failure toast suppressed (chat already notified)` — DEBUG, expected on the 2nd+ failure in the same chat.
- `first-failure toast suppressed: event_emitter is None` — DEBUG, fires when OWUI didn't pass an emitter (rare; indicates an upstream integration issue). It fires **once per request** for the life of an emitter-less chat, not once per chat, because no notice is consumed on this path. Nobody was there, so the chat's one notice is **not** consumed: the next request that does have an emitter still warns.
- `first-failure toast emission raised (suppressed): <exc>` — WARNING, `classifier_failed` branch only, fires if the event_emitter call itself raised. Pipe continues; the chat is latched and its one warning is spent, so later failures in it are silent. `video_intent classifier failed (degrade-open)` is the enclosing handler's own log line, and is the one to grep for this path.
- The raise branch's emit sits in a bare `contextlib.suppress` and logs nothing of its own. The only record for that path is `video_intent classifier failed (degrade-open)`, written **before** the toast is attempted, so it says the classifier raised, not that the toast was lost.

## Rollback

- **Site-wide kill switch**: set admin `VIDEO_INTENT_ENABLED=False`. The classifier is bypassed for every user on every path; the pipe restores its pre-classifier behaviour with no code redeploy. This switch is admin-only: it is read live on every request, so a stale installed filter row cannot re-enable it.
- **Per-user opt-out**: with the admin switch on, a user can disable the classifier just for their own chats via the filter UserValve (`VIDEO_INTENT_ENABLED` on the per-model video filter). Useful when an individual user prefers raw control. This applies to the three non-master settings and to the per-chat opt-out only; the master switch itself is never user-settable.

## Notes for operators

- Frame extraction uses PIL+imageio first, with ffmpeg subprocess as fallback. The `imageio-ffmpeg` package ships its own ffmpeg binary, so there is no system dependency to install.
- Thumbnails are 256×256 JPEG, generated at intent-resolution time (after the classifier returns, before submission). The 256×256 is the canvas, not the picture: the content is letterboxed onto a white canvas with its aspect ratio preserved and centred, and the bars outside it are white. Storage cost: roughly 10–20 KB per thumbnail.
- Filter-injected `frame_images` (user explicitly attached an image) take precedence over classifier output. Plan entries that reference an uploaded attachment the filter claimed as a frame are kept and applied, so phrases like "use this as the last frame" still work. A plan entry that names an attachment the filter claimed as a **style reference** — a clip, an audio file, or an image past the first — is also kept, but it changes nothing: that reference already goes to the model as a reference, not as a frame, so there is no frame to re-target. Plan entries that reference prior videos are dropped when an explicit attachment is present.
- Attachments are collected into one flat list for the classifier: `frame_images` first, then `input_references`, order preserved within each source list, and the index is 0-based across the flat output. Each entry carries a `kind` of `image`, `video` or `other`; a `frame_images` entry is always `image`, and an `input_references` entry is classified by its `content_type` family (`video/*` → `video`, `image/*` → `image`, anything else → `other`). The family is resolved into a per-entry name, never back into the source-list loop variable, so a reference cannot inherit the previous reference's family.
