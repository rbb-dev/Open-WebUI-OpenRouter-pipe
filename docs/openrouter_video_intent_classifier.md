# OpenRouter Video Intent Classifier

The pipe runs a small classifier model before each video generation to figure out what the user actually wants. Without it, the OpenRouter `/videos` endpoint is single-shot and stateless — every request only knows about the latest user message, so a follow-up like "change colour to black" produces an unrelated video instead of a recoloured version of the previous one.

## Why this exists

The OpenRouter `/videos` endpoint takes `prompt`, optional `frame_images`, and optional `input_references`, then returns a video. It does not know about prior turns in a chat. If the user's previous turn produced a video of a cat and they then say "change colour to black", the model has no idea what the cat looked like — it just makes a new video about a black cat (or a black anything).

Other UIs work around this by automatically attaching a frame from the prior video as a reference. The pipe does the same thing now: a small task model reads the chat history, the latest message, and any attachments, then decides what visual reference (if any) to wire into the request before it's submitted.

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

When the selected model's `supported_frame_images` is non-empty AND does not include `input_reference`, the pipe **blocks the paid `/videos` call** and asks the user instead:

```
⚠️ This video model can't visually modify a previous video — it would just produce the same video, ignoring your change request.

Want me to generate as text-only using a rewritten prompt that describes the change?

**Please type a single digit:**
**1** — Yes, generate as text-only
**2** — Cancel
```

The user replies `1` (also accepted silently: `yes`, `y`, `ok`, `okay`, `proceed`, `go`, `do it`) or `2` (also: `no`, `n`, `cancel`, `skip`, `stop`, `abort`). Anything else loops with `❌ "<input>" isn't 1 or 2. Try again.` There is no retry cap.

**On proceed**: the pipe replays the classifier's rewritten prompt (e.g. "a black cat sitting on green grass") as a plain text-to-video request — no frame wiring. The classifier does NOT re-run on the user's `1`; the saved prompt is recovered from hidden markers embedded in the question turn.

**On cancel**: the pipe emits `Cancelled. Type a new request when ready.` No `/videos` call.

**When this gate does NOT fire**:
- Model's `supported_frame_images` includes `input_reference` (model can honor the request) → straight through, no question.
- Model's `supported_frame_images` is empty / unknown → optimistic pass-through, no question (we only block on models we KNOW won't honor it).
- Intent is `continue_prior_video` (uses `last_frame` anchoring, which actually works on these models) → no question.

## Frame extraction at arbitrary timestamps

The classifier supports `prior_video_at_timestamp` with `timestamp_seconds`. Examples the system prompt recognises:

- *"use the frame at 5 seconds"*
- *"from the 5-second mark"*
- *"at 0:30"*

The pipe validates the requested timestamp against the actual video duration. If the requested time exceeds the video length, it gracefully downgrades to the last frame and surfaces a note in the disclosure block:

> ⚠️ Requested frame at 30s but the previous video is only 4s. Using last frame instead.

## Configuration valves (admin)

All admin-scoped on the global `Valves` model. User-tunable per-chat versions of four of these are also exposed on each video model's filter UserValves (see "User-tunable settings" below).

| Valve | Type | Default | Purpose |
|---|---|---|---|
| `VIDEO_INTENT_ENABLED` | `bool` | `True` | Master switch. When False, the classifier is bypassed entirely; only the latest user message is sent to the video model. |
| `VIDEO_INTENT_TASK_MODEL_MODE` | `internal` / `external` | `external` | Which Open WebUI Task Model to use as the classifier. `internal` reads `TASK_MODEL`; `external` reads `TASK_MODEL_EXTERNAL`. |
| `VIDEO_INTENT_TASK_MODEL_FALLBACK` | `none` / `other_task_model` | `other_task_model` | Failure fallback strategy. `none` returns only the primary task model; `other_task_model` also tries the other (internal/external) Task Model. |
| `VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT` | `bool` | `True` | Skip the classifier when the chat has no prior turns and no attachments — there is nothing to classify against, so the call is wasted. Turn off if you want clarifying questions on first-turn ambiguous prompts. |
| `VIDEO_INTENT_MAX_CLARIFICATIONS` | `0`–`3` | `1` | Per-session cap on consecutive clarifying questions. `0` disables the clarification loop entirely. |
| `VIDEO_INTENT_FRAME_EXTRACTION_INDEX` | `first` / `last` | `last` | Default frame to extract from a prior video when the requested frame is unavailable. |
| `VIDEO_INTENT_TIMEOUT_S` | `int` | `8` | Hard timeout (seconds) on the classifier call. On breach, the pipe falls back to sending only the latest user message — the paid video request still proceeds. |
| `VIDEO_INTENT_CONFIRM_MODE` | `always` / `on_reference` / `low_confidence` / `never` | `on_reference` | When to surface the confirmation footer. `on_reference` confirms only when reusing a prior video frame or attached image. |
| `VIDEO_INTENT_MAX_CALLS_PER_CHAT` | `int` | `0` (unlimited) | Cost guard. `0` = unlimited. Admin sets a positive integer to enforce a per-chat ceiling. |
| `VIDEO_INTENT_MAX_CALLS_PER_USER_DAY` | `int` | `0` (unlimited) | Cost guard. `0` = unlimited. Admin sets a positive integer to enforce a per-user-per-day ceiling. |
| `VIDEO_INTENT_LOG_DECISIONS` | `bool` | `False` | Log full intent classification JSON at INFO level. Off by default to avoid leaking prompts into shared logs. |

## User-tunable settings (per-model filter UserValves)

When admin `VIDEO_INTENT_ENABLED=True`, each video model's companion filter exposes four UserValves so individual users can override the admin defaults for their own chats. They appear in the OWUI filter settings panel alongside the existing video knobs (`VIDEO_DURATION`, `VIDEO_ASPECT_RATIO`, etc.).

| User valve | Effect |
|---|---|
| `VIDEO_INTENT_ENABLED` | Per-user opt-out. When off, this user's video chats bypass the classifier even though it's on globally. |
| `VIDEO_INTENT_MAX_CLARIFICATIONS` | Override the cap on consecutive clarifying questions for this user. |
| `VIDEO_INTENT_FRAME_EXTRACTION_INDEX` | Pick which frame the user prefers when the request is ambiguous about first vs last. |
| `VIDEO_INTENT_CONFIRM_MODE` | Control when the disclosure footer appears for this user (always / on reference / on low confidence / never). |

When admin `VIDEO_INTENT_ENABLED=False`, the four user fields do not appear in the filter UI at all — the next time the pipe rebuilds filters (i.e. on the next `pipes()` refresh) the filter source is regenerated without them.

## Anti-overasking guardrails

The classifier is biased to act, not ask. Clarifying questions only fire when:

- The user references "it"/"that"/"the previous one" but **multiple prior videos** exist with no positional cue, AND
- The candidate options would produce **meaningfully different outputs**, AND
- The user has not explicitly opted out (e.g., "just do it", "your call", "you decide").

The classifier obeys explicit wiring instructions ("use the previous video as first frame", "use the frame at 5 seconds") without asking. One-word continuations like "more", "again", "encore" trigger continue-prior-video with a default last-frame anchor, no question.

## Telemetry

When `VIDEO_INTENT_LOG_DECISIONS=True`, every classifier turn emits a structured INFO log line. Keys:

| Key | Meaning |
|---|---|
| `intent_mode` | `text2video` / `image2video_attached` / `image2video_priorframe` / `clarify` / `bypass_skipped` / `bypass_disabled` |
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
- **Frame extraction fails** (corrupt prior video, unsupported codec) → drop that frame_plan entry, append a downgrade note to the disclosure block, continue with other entries; if every entry fails, send text-only.
- **Thumbnail upload fails** → disclosure block omits that thumbnail; the `frame_images` entry still ships.
- **User cancels mid-classification** → cancellation propagates up; `/videos` is never submitted.
- **Model can't honor `input_reference` for modify intent** → the pipe blocks the paid call and asks the user "1 = text-only, 2 = cancel" instead of silently producing an unchanged video. See "When a model can't visually modify a previous video" above.

The **first** classifier infrastructure failure per chat surfaces a notification toast: *"Intent inference unavailable; using simple text-to-video."* Subsequent failures within the same chat are silent (logged at WARNING).

**Diagnostic log lines** for the toast emission path (search these when the toast doesn't appear as expected):

- `video_intent classifier_failed=True; reason=<...>; breaker tripped` — WARNING, fires every time a classifier infrastructure failure is detected.
- `first-failure toast emitted (chat_key=<...>)` — INFO, confirms the toast was sent to the OWUI event emitter.
- `first-failure toast suppressed (chat already notified)` — DEBUG, expected on the 2nd+ failure in the same chat.
- `first-failure toast suppressed: event_emitter is None` — WARNING, fires when OWUI didn't pass an emitter (rare; indicates an upstream integration issue).
- `first-failure toast emission raised (suppressed): <exc>` — WARNING, fires if the event_emitter call itself raised. Pipe continues; user gets no toast for this turn.

## Rollback

- **Site-wide kill switch**: set admin `VIDEO_INTENT_ENABLED=False`. The classifier is bypassed for every user; the pipe restores its pre-classifier behaviour with no code redeploy.
- **Per-user opt-out**: a user can disable the classifier just for their own chats via the filter UserValve (`VIDEO_INTENT_ENABLED` on the per-model video filter). Useful when an individual user prefers raw control.

## Notes for operators

- Frame extraction uses PIL+imageio first, with ffmpeg subprocess as fallback. The `imageio-ffmpeg` package ships its own ffmpeg binary, so there is no system dependency to install.
- Thumbnails are 256×256 JPEG, generated at intent-resolution time (after the classifier returns, before submission). Storage cost: roughly 10–20 KB per thumbnail.
- Filter-injected `frame_images` (user explicitly attached an image) take precedence over classifier output. Plan entries that reference uploaded attachments are kept (so phrases like "use this as the last frame" still work); plan entries that reference prior videos are dropped when an explicit attachment is present.
