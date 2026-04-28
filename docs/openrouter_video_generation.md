# OpenRouter Video Generation

This pipe exposes OpenRouter's eleven async video-generation models as
selectable chat models in Open WebUI. You pick a video model in the chat
header (just like any other LLM), type a prompt, and the pipe submits a
job, polls until completion, downloads the generated video into Open WebUI
file storage, and renders it inline with a `<video>` tag.

The feature is on by default (`ENABLE_VIDEO_GENERATION=True`). If you want
to disable it, set that valve to `False` in Admin → Functions → OpenRouter
pipe → Valves.

## Table of contents

- [Quickstart](#quickstart)
- [The 11 video models](#the-11-video-models)
- [Per-model deep dive](#per-model-deep-dive)
- [Per-model parameter reference](#per-model-parameter-reference)
- [Filter UserValve identifiers (master reference)](#filter-uservalve-identifiers-master-reference)
- [The chat filter UI (UserValves)](#the-chat-filter-ui-uservalves)
- [The `help` command](#the-help-command)
- [Frame images and image-to-video](#frame-images-and-image-to-video)
- [Multimodal references (Wan 2.7)](#multimodal-references-wan-27)
- [Provider passthrough](#provider-passthrough)
- [Pricing and cost display](#pricing-and-cost-display)
- [Output rendering and message format](#output-rendering-and-message-format)
- [Resume, recovery, and disconnect resilience](#resume-recovery-and-disconnect-resilience)
- [Concurrency limits](#concurrency-limits)
- [Configuration valves (admin)](#configuration-valves-admin)
- [Errors and troubleshooting](#errors-and-troubleshooting)
- [Architecture overview](#architecture-overview)
- [Phase 0 probe](#phase-0-probe)
- [Limitations and non-goals](#limitations-and-non-goals)

---

## Quickstart

### For end users

1. Open a chat in Open WebUI.
2. In the model picker, choose any model whose name starts with the
   provider's name (e.g. `Google: Veo 3.1 Lite`, `OpenAI: Sora 2 Pro`,
   `Alibaba: Wan 2.7`). Video models look like normal chat models — they
   are not in a separate menu.
3. (Optional) Open the Integrations menu (puzzle-piece icon below the
   prompt input). The matching `Veo 3.1 Lite` (or whichever model) filter
   should already be toggled on. This is the **per-model filter** that
   exposes parameter knobs for that model.
4. (Optional) Click the per-model filter's settings icon to set
   per-message overrides — duration, aspect ratio, resolution, audio,
   negative prompt, etc.
5. (Optional) Attach one or two images. The first image becomes the
   `first_frame`, and if you attach two and the model supports
   `last_frame`, the second becomes the closing frame.
6. Type your prompt and press send.
7. The chat shows a status line ("Submitting", "Polling", "Downloading",
   "Generated in 35.2s · $0.40"). Generation typically takes 30s–4min
   depending on the model and duration.
8. The final message renders an inline video player. Click play.

### Model-specific help in chat

Typing the literal word `help` (no other text) into a chat against any
video model returns a research-grounded model-specific help blurb covering:

- What the model is best known for
- Output capabilities (durations, aspect ratios, resolutions, frames, audio, seed)
- Every filter knob exposed for this model and what it does
- 3–4 tips and pitfalls
- Live pricing rates pulled from the OpenRouter catalog

This is the fastest way to learn a model without leaving the chat. Try
it on each of the 11 models — the answers are different for every one.

### For administrators

Out-of-the-box defaults are sensible for most deployments:

```
ENABLE_VIDEO_GENERATION = True
AUTO_INSTALL_VIDEO_FILTERS = True       # creates per-model filter rows
AUTO_ATTACH_VIDEO_FILTERS = True        # attaches each filter to its model
AUTO_DEFAULT_VIDEO_FILTERS = True       # filter is on-by-default per chat
MAX_CONCURRENT_VIDEO_GENS = 2           # global cap per pipe process
MAX_CONCURRENT_VIDEO_GENS_PER_USER = 2  # per-user cap
```

If the per-model filters do not appear in the Integrations menu, check:
- `AUTO_INSTALL_VIDEO_FILTERS` and `AUTO_ATTACH_VIDEO_FILTERS` are both
  `True`.
- The pipe has been called at least once with a logged-in user (the
  filters install during `pipes()` warmup).
- OpenRouter Admin → Functions lists 11+ entries named ` Veo 3.1 Lite`,
  ` Seedance 2.0`, etc. (note the leading space — that's intentional, see
  [The chat filter UI](#the-chat-filter-ui-uservalves)).

**Access control for non-admin users.** Video models are inserted PRIVATE
by default per the standard `NEW_MODEL_ACCESS_CONTROL` valve (default
`admins`). Non-admin users will not see video models in the picker until
an admin explicitly grants access via Admin → Models → [video model row]
→ Access. The pipe's auto-attach and auto-default behaviour fully
prepares the model row beforehand (filter wired, defaulted on, ready to
generate), so the per-model access grant is the only manual step
required. This is intentional policy — video generation is expensive
enough that operators usually want admin-curated access.

**Auto-default re-assert.** The per-model filter is re-defaulted to
enabled on every catalog metadata sync (typically every pipe `pipes()`
call). If you manually disable a video filter for a chat, the next sync
will re-default it. Set `AUTO_DEFAULT_VIDEO_FILTERS=False` to opt out
of the re-assert.

See [Configuration valves](#configuration-valves-admin) for the full list
(19 video-related valves).

---

## The 11 video models

| Model id | Display name | Best for | Audio | Seed | Frames | Cost rate |
|----------|--------------|----------|:----:|:----:|:------:|-----------|
| `google/veo-3.1` | Google: Veo 3.1 | Flagship hero shots; best prompt adherence; native synchronised audio with ~120ms lip-sync; up to 4K. | ✅ | ✅ | first + last | ~$0.40/s with audio |
| `google/veo-3.1-fast` | Google: Veo 3.1 Fast | Drafting/iteration of Veo 3.1 quality at ~60% lower cost; A/B-testing concepts; image-to-video. | ✅ | ✅ | first + last | ~$0.10/s without audio, $0.12/s with |
| `google/veo-3.1-lite` | Google: Veo 3.1 Lite | Cheapest Veo tier; high-volume / batch / consumer-app integrations; same speed as Fast at half the cost. | ✅ | ✅ | first + last | $0.03/s @ 720p without audio |
| `kwaivgi/kling-video-o1` | Kling: Video O1 | Cinematic film-grade clips, character/identity consistency, physics-aware human motion. No deterministic seed. | ✅ | ❌ | first + last | $0.0896/s |
| `minimax/hailuo-2.3` | MiniMax: Hailuo 2.3 | State-of-the-art human physics and emotional micro-expressions; fluid + cloth + fire dynamics. **Silent — no audio.** | ❌ | ❌ | first only | $0.0817/s |
| `alibaba/wan-2.7` | Alibaba: Wan 2.7 | Multimodal reference control (up to 5 ref videos + image grids), lip-sync across languages, FLF2V. Tuned for character-led narrative. | ✅ | ✅ | first + last | $0.10/s |
| `alibaba/wan-2.6` | Alibaba: Wan 2.6 | Cheaper Wan tier with multi-shot storyboarding, 24fps, dialogue + lip-sync, shot_type cinematography. **First-frame only.** | ✅ | ✅ | first only | $0.04–$0.15/s by mode and resolution |
| `bytedance/seedance-1-5-pro` | ByteDance: Seedance 1.5 Pro | First Dual-Branch DiT with native unified video+audio, multilingual lip-sync, widest size matrix (21 dimensions). | ✅ | ✅ | first + last | Token-based, ~$1.20–$2.40 / M tokens |
| `bytedance/seedance-2.0` | ByteDance: Seedance 2.0 | Universal Reference (text + 9 images + 3 video/audio), best character consistency for branded/series content. | ✅ | ✅ | first + last | Token-based, $0.000007/token |
| `bytedance/seedance-2.0-fast` | ByteDance: Seedance 2.0 Fast | Speed-optimised Seedance 2.0; ~30% cheaper; 480p/720p only; ideal for drafts and bulk pipelines. | ✅ | ✅ | first + last | $0.0000056/token |
| `openai/sora-2-pro` | OpenAI: Sora 2 Pro | Physics-accurate motion + world-state persistence across multi-shot sequences. Longest clips (up to 20s). **Text-only — no frame images.** | ✅ | ❌ | none | $0.30/s @ 720p, $0.50/s @ 1080p |

Pick model selection rules of thumb:

- **Speed + cost matters most** → Veo 3.1 Lite (cheapest), Seedance 2.0 Fast (token-priced bulk).
- **Hero shot for client work** → Veo 3.1 (full) or Seedance 2.0.
- **Multi-shot story with consistent characters** → Wan 2.7 or Seedance 2.0.
- **Dialogue / lip-sync from a reference voice** → Wan 2.7 (audio passthrough), Seedance 1.5 Pro.
- **Physics realism / human motion** → Sora 2 Pro, Hailuo 2.3.
- **Longest clip** → Sora 2 Pro (20s).
- **No audio needed (cheapest path)** → Hailuo 2.3, or set `Audio = off` on Veo Lite.

---

## Per-model deep dive

This section is the same content the in-chat `help` command renders, in
written form. Skip to a model that matches your use case, or read all
eleven to get a feel for the catalog. Every paragraph here is grounded
in the OpenRouter catalog metadata + targeted public research on each
model's reputation, papers, and signature features.

### Google: Veo 3.1

> **id**: `google/veo-3.1`

Google DeepMind's flagship video model, positioned for production-quality
output where visual fidelity is the priority — commercial deliverables,
hero shots, and cinematic sequences. Its standout trait is jointly-diffused
native audio: dialogue, SFX, and ambience are generated alongside the
video in a single pass with lip-sync within roughly 120ms. Compared to
Fast and Lite, the full tier delivers sharper motion, stronger prompt
adherence, finer texture/lighting detail, and access to 4K output, at a
higher cost per second. It also leads MovieGenBench evaluations on
overall preference and prompt-following accuracy.

**Tips & pitfalls**

- Write prompts like a shot list: structure as Camera/Lens, Subject,
  Action, Environment, Lighting, Style, Audio — Veo 3.1 responds far
  better to film-industry vocabulary than conversational prose.
- Specify audio explicitly. If you leave dialogue, SFX, or ambience
  undefined, Veo defaults to rushed reads, mismatched ambience, or
  unwanted on-screen subtitles — quote dialogue with `(no subtitles)`
  to suppress captions.
- Known weak spots: in-video text rendering is unreliable, hands and
  limbs can warp, multi-subject scenes drift, and exact object counts
  break down past ~15 items. Use a negative prompt covering "no
  warping, no duplicate limbs, no face distortion, no floating objects"
  and prefer "a small group" over hard numbers.
- Keep one dominant action per 8-second clip; conflicting simultaneous
  actions destabilise physics. For longer narratives, generate separate
  clips and stitch with last-frame conditioning rather than overloading
  one prompt.

### Google: Veo 3.1 Fast

> **id**: `google/veo-3.1-fast`

The speed-and-cost-optimised tier of Veo 3.1, generating 4-, 6-, or
8-second clips up to 4K with native synchronised audio at roughly 2× the
speed and a fraction of the price of full Veo 3.1. Editor blind tests put
quality within ~1–8% of the full tier while costs run ~60% lower, making
it the workhorse choice for drafting, A/B-testing creative concepts,
batch ad and social content, and image-to-video work where dialogue and
SFX must land in sync.

**Tips & pitfalls**

- Front-load one clear shot: cinematography + subject + action +
  context + style/audio in plain prose, one idea per clip — multi-subject
  scenes and crowded actions remain a weak spot.
- Use first/last frame for controlled transitions: describe the
  transformation between the two stills (e.g. "camera arcs 180°,
  lighting shifts cool→warm"), not the stills themselves; mismatched
  aspect/lighting causes identity pops.
- Audio is generated from prompt cues — describe ambient sound,
  dialogue, and SFX explicitly, otherwise you get generic ambience;
  turning audio off both saves cost and avoids out-of-sync lip movement
  on talking heads.
- At 8s, faces and logos can drift partway through. Lock with reference
  text, reuse a seed when iterating, and use the negative prompt to
  exclude common failures ("no text overlays, no extra fingers, no
  warped logos").

### Google: Veo 3.1 Lite

> **id**: `google/veo-3.1-lite`

Google's most cost-effective Veo 3.1 tier, positioned for high-volume
video applications and rapid iteration where cost-per-clip is the
deciding factor. Unlike Veo 3.1 Fast, it does not sacrifice generation
speed for the lower price — it matches Fast's latency at less than half
the cost — making it the go-to pick for batch pipelines, social
automation, and consumer-app integrations. Tradeoffs: a hard cap at
1080p (no 4K), no video extension, and slightly less polished visual
fidelity, but it retains native synchronised audio.

**Tips & pitfalls**

- Upgrade to Veo 3.1 (full) when you need a final hero cut, 4K output,
  or video extension — Lite caps at 1080p and cannot extend an existing
  clip.
- Lite is tuned for "Cinematic Control" prompts — explicit camera
  directives like "slow pan", "low-angle tilt", and named lighting
  setups land more reliably than vague mood descriptors.
- For complex multi-subject scenes or fine character consistency,
  expect more retries than the full tier — generate a small batch with
  different seeds rather than over-engineering one prompt.
- Download outputs immediately: Google retains generated video URIs for
  only ~2 days before they expire.

### Kling: Video O1

> **id**: `kwaivgi/kling-video-o1`

Kuaishou's Kling Video O1 is best known for cinematic, film-grade output
with strong character and subject consistency, "director-like memory"
that locks identities across shots, and physics-aware human motion
(weight, momentum, fabric, water) that holds up better than most peers.
It shines at reference-driven workflows — mixing characters/props
across multi-shot sequences — and at previsualisation, marketing assets,
and short narrative clips where camera language (tracking, push-in,
aerial) matters.

**Tips & pitfalls**

- Write like a director, not a tagger: lead with camera (wide / slow
  dolly-in / tracking) and motivate the camera move narratively — Kling
  responds to cinematic intent more than object lists.
- Be explicit about motion physics and end state: describe how a body
  or fabric moves and how the shot resolves; vague motion or missing
  end-states cause stalls and rubbery limbs.
- Use the negative prompt as guardrails (e.g. "blurry text, extra
  fingers, warped face") rather than burying don'ts in the main prompt
  — Kling honours negatives well.
- No seed control is exposed, so don't expect bit-exact repeats; lock
  look via reference frames (first/last) and tight prompt language
  instead, and avoid on-screen text (Kling renders text poorly).

### MiniMax: Hailuo 2.3

> **id**: `minimax/hailuo-2.3`

MiniMax's flagship video model, best known for state-of-the-art human
physics and character motion — fluid full-body choreography, accurate
limb tracking, and lifelike facial micro-expressions that read as
genuine emotion rather than uncanny animation. It excels at
physics-heavy dynamics (rigid body, fluids, cloth, fire) where most
rivals fall apart, and renders cinematic 1080p output with strong camera
control and stylisation across photoreal, anime, illustration, and
ink-wash looks.

**Tips & pitfalls**

- **No audio**: Hailuo 2.3 is silent — you must add dialogue, SFX, and
  music in post; the catalog confirms `generate_audio=false`.
- **First-frame only**: 2.3 dropped last-frame conditioning that 2.0
  had, so you can anchor the opening still but cannot pin the ending —
  plan motion to flow forward from the first frame.
- Hailuo rewards specific physical and emotional direction (e.g.
  "tight smile turning to laughter," "cloth catches the wind, then
  settles") far more than other models — vague prompts under-use its
  physics strengths.
- `prompt_optimizer` rewrites/expands your prompt for better adherence;
  turn it off only when you have a deliberately precise prompt.
  `fast_pretreatment` speeds up that step at a small quality cost —
  useful for batch runs, otherwise leave at default.

### Alibaba: Wan 2.7

> **id**: `alibaba/wan-2.7`

Alibaba Tongyi Lab's flagship multimodal video model, unifying text,
image, audio, and video conditioning in a single 27B-parameter Diffusion
Transformer with Flow Matching. Its standout capability is true
multi-reference control: lock subject identity, vocal timbre, props, and
visual style across new scenes by feeding up to five reference videos
plus reference image grids. It also adds last-frame anchoring (FLF2V),
native audio-synced lip generation across languages, and a "Thinking
Mode" planner that improves coherence on dialogue- and character-led
shots — at the cost of weaker fast-motion physics than Seedance 2.0.

**Tips & pitfalls**

- Pick the right reference channel: use the images array to lock
  appearance, wardrobe, props (it works like a 9-image storyboard grid
  in 2.7); use the video / videos passthrough for motion style, camera
  language, or vocal timbre transfer.
- For talking-head and dialogue clips, supply an audio reference —
  Wan 2.7's automatic lip-sync matches mouth shapes to the supplied
  speech in the target language, a headline upgrade over 2.6.
- Wan 2.7 is tuned for character-led, narrative content; for fast
  sports/action shots its physics still trails Seedance 2.0 and Runway
  Gen-4, so add explicit motion verbs and a negative prompt against
  blur/morphing.
- Wan 2.7's instruction-following changed vs 2.6, so prompts calibrated
  on 2.6 may drift; lean on `prompt_extend = on` when prompts are
  short, but turn it off when you've already written a precise
  multi-shot storyboard.

### Alibaba: Wan 2.6

> **id**: `alibaba/wan-2.6`

Alibaba's most feature-rich video generation model (Dec 2025), supporting
10+ unified visual creation capabilities (text-to-video, image-to-video,
reference-to-video, voiceover, action generation, role-play, editing) on
a 14B-parameter MoE architecture. Best known for affordable multi-shot
1080p @ 24fps generation with synchronised native audio (multi-speaker
dialogue, lip-sync, voice/music conditioning) and intelligent multi-shot
narrative storyboarding that holds character and lighting consistency
across cuts. Pick 2.6 over 2.7 when you want the cheaper, well-tuned
generation pipeline; pick 2.7 only if you specifically need last-frame
control, 9-grid input, instruction-based editing, or stronger physics.

**Tips & pitfalls**

- **First-frame ONLY**: Wan 2.6 supports `first_frame` image
  conditioning but has no `last_frame`. To define both endpoints of a
  clip, you must upgrade to Wan 2.7 — don't try to fake it through
  prompts.
- `shot_type` controls camera framing/composition (e.g. values like
  `medium_to_closeup`), used for cinematic shot intent; this knob was
  removed in 2.7. For multi-shot scripts, write scene-timed segments in
  the prompt itself.
- Use `enable_prompt_expansion` (LLM-based prompt rewriter) for short
  or terse prompts — it adds cinematographic detail "for free" without
  consuming your budget; turn it OFF when you've already crafted a
  long, precise prompt.
- Audio reference files must be 3–30s, WAV/MP3, max 15 MB; clips longer
  than the video get truncated and shorter clips leave a silent tail.
  Two-speaker dialogue tends to collapse to one dominant voice —
  generate single-speaker clips and composite.

### ByteDance: Seedance 1.5 Pro

> **id**: `bytedance/seedance-1-5-pro`

ByteDance's first foundation model to natively generate video and audio
in a single unified pass, using a 4.5B-parameter Dual-Branch Diffusion
Transformer with a cross-modal joint module that locks phonemes to
visemes and physics events to audio spikes at millisecond precision.
Pick 1.5 Pro over Seedance 2.0 when you want the older,
production-validated audio-visual workflow at materially lower cost; it
offers 1080p output, the wider 4–12s duration window, and reliable
multilingual lip-sync (Mandarin, English, Japanese, Korean, Spanish, plus
dialects).

**Tips & pitfalls**

- Audio doubles the bill: `video_tokens` with audio is ~2× without
  audio, so toggle Audio off for silent B-roll, layout passes, or
  anything you'll dub later.
- Use 1.5 Pro for short, repeatable clips with simple camera work and
  known-good prompts; switch to 2.0 only when you need richer multimodal
  references, 2K output, or longer 15s shots — 1.5 Pro caps at 1080p
  and 12s.
- Long durations drift: 4–6s clips stay on-model, but 10–12s shots show
  face drift, color shift, and continuity errors — chain shorter shots
  with `last_frame` anchors and consistent character descriptions.
- `last_frame` is a directional guide, not a pixel-perfect target —
  pick an end frame with framing and lighting close to the start frame,
  or you'll get jumpy transitions in the final second.

### ByteDance: Seedance 2.0

> **id**: `bytedance/seedance-2.0`

ByteDance's flagship multimodal video model, best known for "locked"
character consistency — preserving faces, clothing, accessories, and
small subject details across the duration of a clip and across
multi-shot generations. It stands out for its "Universal Reference"
system that accepts text plus up to 9 images and 3 video/audio clips in
a single generation, letting you direct composition, camera movement,
and character actions from reference assets at once. Unlike 2.0 Fast
(speed over quality) and 1.5 Pro (limited to text + first/last frame),
the full 2.0 variant is the production-quality choice with native audio
(dialogue, ambience, SFX) and multi-shot story coherence.

**Tips & pitfalls**

- Reach for full Seedance 2.0 (not Fast) when you need production
  drafts where identity preservation matters — branded characters,
  story-led scenes, or repeatable creative formats — and accept the
  longer render in exchange for tighter facial/clothing fidelity.
- Long durations (12–15s) still drift more than short ones; for the
  most stable identity, anchor with a reference image AND a clear text
  description of the subject, and prefer 6–10s clips for hero shots.
- The multimodal reference workflow is the headline feature — use
  images for style/identity, video clips for motion/camera language,
  audio for pacing — but keep references coherent; conflicting
  references degrade consistency more than helping it.
- ByteDance gates real-person reference features and identity
  verification due to IP/likeness concerns; expect occasional refusals
  on celebrity or copyrighted-character prompts.

### ByteDance: Seedance 2.0 Fast

> **id**: `bytedance/seedance-2.0-fast`

ByteDance's speed-and-cost-optimised variant of the Seedance 2.0 family,
built on the same unified multimodal architecture but using distillation
and accelerated sampling to cut generation time at roughly 30–33% lower
cost than standard Seedance 2.0. Best known for cinematic 480p/720p
output with native audio synchronised in a single pass, support for
text-to-video, image-to-video with first/last frame control, and
multimodal reference-to-video, plus very wide aspect-ratio coverage
including 21:9 cinematic and 9:21. OpenRouter bills it via video tokens,
so cost scales with pixels × seconds.

**Tips & pitfalls**

- Use Fast for drafting, prompt iteration, and bulk pipelines; switch
  to standard Seedance 2.0 for hero shots — Fast trades a small amount
  of motion refinement and detail for ~33% lower cost.
- Token math means doubling resolution or duration roughly multiplies
  cost; a 720p 10s clip costs far more than a 480p 5s draft, so iterate
  small first.
- This OpenRouter listing does not expose `negative_prompt` and caps at
  720p — for 1080p or text-prompted negatives you need the standard
  2.0 model or another provider.
- `watermark` toggles the visible provider/ByteDance branding overlay
  on the returned MP4, and `req_key` is ByteDance/Volcengine ModelArk's
  internal model-routing identifier — leave both at defaults unless
  your provider explicitly tells you otherwise.

### OpenAI: Sora 2 Pro

> **id**: `openai/sora-2-pro`

OpenAI's flagship video model, best known for physics-accurate motion
(gravity, momentum, fluid dynamics, object permanence — e.g. a missed
basketball realistically rebounds off the backboard) paired with
natively synchronised audio: dialogue, sound effects, and ambient audio
are predicted alongside the frames rather than dubbed in, so footsteps
land on the correct frame and lip-sync stays tight. Its standout
differentiator is world-state persistence across multi-shot sequences —
characters, props, and spatial relationships stay consistent across
cuts, enabling cohesive short-form storytelling. In this catalog it
also offers the longest clips of any video model, up to 20 seconds at
full 1080p.

**Tips & pitfalls**

- **Text-to-video only here**: this catalog entry has no
  `frame_image` support, so you can't seed it with a start/end image —
  drive the result entirely from prompt language.
- 20-second durations and 1080p are unique strengths but render slow —
  community tests report 2–5 minutes for a 20s clip and much longer at
  peak, so prefer 4–8s 720p for iteration and reserve 16–20s 1080p for
  finals.
- Plays to its strengths on physics, motion weight, lighting, and
  ambient/dialogue audio; struggles with on-screen text, brand logos,
  fine hand details, and highly choreographed multi-character action —
  don't ship as-is for client deliverables that depend on legible text.
- `Quality` and `Style` are passthrough hints OpenRouter forwards; the
  OpenAI Videos API itself doesn't expose a discrete quality enum
  (resolution drives the tier), so treat them as soft hints rather than
  guaranteed switches.

---

## Per-model parameter reference

This section enumerates exactly which filter knobs each model exposes,
based on the OpenRouter catalog at the time of writing.
The chat-filter UI auto-hides knobs the model does not support, so this table
is also the spec for what you can change per-message.

### Google: Veo 3.1 / Veo 3.1 Fast / Veo 3.1 Lite

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 4, 6, 8 | Cost scales per second. |
| Aspect ratio | Literal | 16:9, 9:16 | Native composition (no crop). |
| Resolution | Literal | 720p, 1080p, 4K (full + Fast); 720p, 1080p (Lite) | 4K SKU is roughly 2× the 1080p rate on the full tier. |
| Size | Literal | from `supported_sizes` in catalog | Exact pixel dimensions; used when you need a specific canvas. |
| Frames | Literal | auto / none / first_only / first_last | first/last requires both images attached. |
| Negative prompt | str | free text | Routed via `negativePrompt` passthrough. |
| Audio (`generate_audio`) | Literal | model_default / on / off | Off cuts price ~50% but loses signature joint-diffusion soundtrack. |
| Seed | int | 0 = model default; otherwise 32-bit integer | Same prompt + seed yields a near-identical clip. |
| Person generation | Literal | "" / allow_all / allow_adult / dont_allow | Safety gate. EU/UK/CH/MENA only allow `allow_adult`. |
| Conditioning scale | float | 0.0 = default; 0.0–1.0 | Bias toward reference images vs text prompt. |
| Enhance prompt | Literal | model_default / on / off | Auto-rewrite prompt (officially Veo 2 only on Vertex; provider may ignore). |
| Provider options JSON | str | raw JSON object keyed by provider slug | Escape hatch — see [Provider passthrough](#provider-passthrough). |

### Kling: Video O1

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 5, 10 | Linear pricing. |
| Aspect ratio | Literal | 16:9, 9:16, 1:1 | Square output uses 720×720. |
| Resolution | Literal | 720p only | Single tier. |
| Size | Literal | 1280×720, 720×1280, 720×720 | |
| Frames | Literal | auto / none / first_only / first_last | first/last for controlled transitions. |
| Negative prompt | str | free text | Hard guardrails — Kling honours these strongly. |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native ambient audio. |
| Provider options JSON | str | raw JSON | |

**No seed knob** — Kling's catalog says `seed: false`. Lock visual identity via reference frames + prompt language, not bit-exact replay.

### MiniMax: Hailuo 2.3

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 6, 10 | |
| Aspect ratio | Literal | 16:9 only | |
| Resolution | Literal | 1080p only | Native HD. |
| Size | Literal | 1920×1080 | Single canvas. |
| Frames | Literal | auto / none / first_only | **No `first_last`** — Hailuo 2.3 dropped last-frame support. |
| Provider options JSON | str | raw JSON | |
| Prompt optimizer | Literal | model_default / on / off | MiniMax server-side prompt rewriter. |
| Fast pretreatment | Literal | model_default / on / off | Quicker optimiser pass; small quality cost. |

**No audio knob** (`generate_audio: false`). **No seed knob** (`seed: null`). **No negative prompt.**

### Alibaba: Wan 2.7

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 2, 3, 4, 5, 6, 7, 8, 9, 10 | Widest 1-second granularity. |
| Aspect ratio | Literal | 16:9, 9:16, 1:1, 4:3, 3:4 | |
| Resolution | Literal | 720p, 1080p | No 4K. |
| Size | Literal | 10 dimensions | Including 1440×1080 broadcast-safe. |
| Frames | Literal | auto / none / first_only / first_last | FLF2V interpolates between two keyframes. |
| Negative prompt | str | free text | Useful against fast-motion artefacts. |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native audio with multi-language lip-sync. |
| Seed | int | 0 / 32-bit int | Multi-shot continuity uses this. |
| Provider options JSON | str | raw JSON | |
| Audio reference URL | str | URL | Voice timbre + lip motion conditioning (Wan-2.7-r2v). |
| Last image URL | str | URL | Anchor closing frame via passthrough. |
| Reference video URL | str | URL | Single reference video (motion/camera/vocal style transfer). |
| Reference videos JSON | str | JSON array | Up to 5 reference videos in one call. |
| Reference images JSON | str | JSON array | 9-image structured grid for identity/wardrobe/props. |
| Prompt extend | Literal | model_default / on / off | Wan prompt rewriter. |
| Ratio | str | provider-specific string | Non-standard aspect string passthrough. |

### Alibaba: Wan 2.6

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 5, 10 | OpenRouter SKU caps at 10s. |
| Aspect ratio | Literal | 16:9, 9:16 | |
| Resolution | Literal | 720p, 1080p | |
| Size | Literal | 4 dimensions | |
| Frames | Literal | auto / none / first_only | **No `first_last`** — upgrade to Wan 2.7 for that. |
| Negative prompt | str | free text | |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native synchronised audio with multi-speaker dialogue. |
| Seed | int | 0 / 32-bit int | |
| Provider options JSON | str | raw JSON | |
| Audio reference URL | str | URL | WAV/MP3, 3–30s, ≤15 MB; longer clips truncate. |
| Enable prompt expansion | Literal | model_default / on / off | LLM-based prompt rewriter for short prompts. |
| Shot type | str | e.g. `medium_to_closeup` | **Wan 2.6-only** — controls camera framing; removed in 2.7. |

### ByteDance: Seedance 2.0 Fast / Seedance 2.0 / Seedance 1.5 Pro

| Knob | Type | Values (varies by variant) | Notes |
|------|------|----------------------------|-------|
| Duration | Literal | 4–15 (Fast/2.0); 4–12 (1.5 Pro) | Token-priced. |
| Aspect ratio | Literal | 1:1, 3:4, 9:16, 4:3, 16:9, 21:9, 9:21 (+ 9:21 on 1.5 Pro) | Widest aspect coverage. |
| Resolution | Literal | 480p, 720p (Fast); 480p, 720p, 1080p (2.0, 1.5 Pro) | |
| Size | Literal | 13 (Fast); 19 (2.0); 21 (1.5 Pro) | The most flexible canvas options of any catalog model. |
| Frames | Literal | auto / none / first_only / first_last | |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native audio in same pass as video. |
| Seed | int | 0 / 32-bit int | |
| Provider options JSON | str | raw JSON | |
| Watermark | Literal | model_default / on / off | Visible ByteDance watermark on output MP4. |
| Req key | str | provider routing string | Volcengine ModelArk SKU/endpoint identifier. |

**No negative prompt** on any Seedance variant — not in `allowed_passthrough_parameters`.

### OpenAI: Sora 2 Pro

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 4, 8, 12, 16, 20 | **Up to 20s — longest in catalog.** |
| Aspect ratio | Literal | 16:9, 9:16 | No square / cinematic widescreen. |
| Resolution | Literal | 720p, 1080p | $0.30/s vs $0.50/s. |
| Size | Literal | 4 dimensions | |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native dialogue, SFX, ambience — Sora's signature. |
| Provider options JSON | str | raw JSON | |
| Quality | Literal | "" / standard / hd | Hint — resolution drives the actual tier. |
| Style | str | free text | Stylistic hint (e.g. `cinematic`, `documentary handheld`). |

**No frames knob** (catalog says `supported_frame_images: null` — text-to-video only). **No seed knob** (`seed: false`). **No negative prompt.**

---

## Filter UserValve identifiers (master reference)

The per-model parameter tables above use friendly UI labels ("Duration",
"Person generation"). Internally each maps to a Pydantic `UserValves`
field with a `VIDEO_*` identifier rendered into the filter source. Use
this table when grepping the source, writing tests, or programmatically
constructing filter inputs.

`Type` column reads as Pydantic field type. `Default` is the value
treated as "leave model default" (skipped from the request). `Gate` is
the catalog condition under which the valve renders.

### Core UserValves (pre-existing — every variant reuses these)

| Identifier | Type | Default | Maps to API field | Gate (catalog condition) | Exposed on |
|------------|------|---------|-------------------|---------------------------|------------|
| `VIDEO_PROVIDER_OPTIONS_JSON` | `str` | `""` | `provider.options` (raw JSON object keyed by slug) | always | all 11 |
| `VIDEO_DURATION` | `Literal[0, …]` | `0` | top-level `duration` | `supported_durations` non-empty | all 11 |
| `VIDEO_ASPECT_RATIO` | `Literal["", …]` | `""` | top-level `aspect_ratio` | `supported_aspect_ratios` non-empty | all 11 |
| `VIDEO_RESOLUTION` | `Literal["", …]` | `""` | top-level `resolution` | `supported_resolutions` non-empty | all 11 |
| `VIDEO_SIZE` | `Literal["", …]` | `""` | top-level `size` | `supported_sizes` non-empty | all 11 |
| `VIDEO_FRAME_MODE` | `Literal["auto", "none", "first_only"(, "first_last")]` | `"auto"` | controls `frame_images[]` shaping | `supported_frame_images` non-empty | 10 (all except Sora 2 Pro) |
| `VIDEO_NEGATIVE_PROMPT` | `str` | `""` | top-level `negative_prompt` (or `negativePrompt` on Veo) | `"negative_prompt"` or `"negativePrompt"` in `allowed_passthrough_parameters` | Veo trio, Kling, Wan 2.6, Wan 2.7 |
| `VIDEO_GENERATE_AUDIO` | `Literal["model_default", "on", "off"]` | `"model_default"` | top-level `generate_audio` (boolean) | top-level `generate_audio: true` in catalog | 10 (all except Hailuo) |
| `VIDEO_SEED` | `int` (`ge=0`) | `0` | top-level `seed` | top-level `seed: true` in catalog | 8 (Veo trio, Wan 2.6, Wan 2.7, Seedance trio) |
| `VIDEO_AUDIO_URL` | `str` | `""` | passthrough `audio` (URL) | `"audio"` in `allowed_passthrough_parameters` | Wan 2.6, Wan 2.7 |
| `VIDEO_REFERENCE_VIDEO_URL` | `str` | `""` | passthrough `video` | `"video"` in `allowed_passthrough_parameters` | Wan 2.7 |
| `VIDEO_REFERENCE_VIDEOS_JSON` | `str` (JSON array) | `""` | passthrough `videos` | `"videos"` in `allowed_passthrough_parameters` | Wan 2.7 |
| `VIDEO_REFERENCE_IMAGES_JSON` | `str` (JSON array) | `""` | passthrough `images` | `"images"` in `allowed_passthrough_parameters` | Wan 2.7 |
| `VIDEO_LAST_IMAGE_URL` | `str` | `""` | passthrough `last_image` | `"last_image"` in `allowed_passthrough_parameters` | Wan 2.7 |

### New typed UserValves added in this feature (passthrough-param wrappers)

These were added in the verification + upgrade pass so users no longer
need to hand-write JSON for every model-specific knob. Each renders only
when the corresponding string appears in the model's
`allowed_passthrough_parameters`.

| Identifier | Type | Default | Maps to API field | Gate | Exposed on |
|------------|------|---------|-------------------|------|------------|
| `VIDEO_PERSON_GENERATION` | `Literal["", "allow_all", "allow_adult", "dont_allow"]` | `""` | passthrough `personGeneration` | `"personGeneration"` allowed | Veo trio |
| `VIDEO_CONDITIONING_SCALE` | `float` (`ge=0.0`, `le=1.0`) | `0.0` | passthrough `conditioningScale` | `"conditioningScale"` allowed | Veo trio |
| `VIDEO_ENHANCE_PROMPT` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `enhancePrompt` (boolean) | `"enhancePrompt"` allowed | Veo trio |
| `VIDEO_PROMPT_OPTIMIZER` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `prompt_optimizer` (boolean) | `"prompt_optimizer"` allowed | Hailuo |
| `VIDEO_FAST_PRETREATMENT` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `fast_pretreatment` (boolean) | `"fast_pretreatment"` allowed | Hailuo |
| `VIDEO_PROMPT_EXTEND` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `prompt_extend` (boolean) | `"prompt_extend"` allowed | Wan 2.7 |
| `VIDEO_RATIO` | `str` | `""` | passthrough `ratio` | `"ratio"` allowed | Wan 2.7 |
| `VIDEO_ENABLE_PROMPT_EXPANSION` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `enable_prompt_expansion` (boolean) | `"enable_prompt_expansion"` allowed | Wan 2.6 |
| `VIDEO_SHOT_TYPE` | `str` | `""` | passthrough `shot_type` | `"shot_type"` allowed | Wan 2.6 |
| `VIDEO_WATERMARK` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `watermark` (boolean) | `"watermark"` allowed | Seedance trio |
| `VIDEO_REQ_KEY` | `str` | `""` | passthrough `req_key` | `"req_key"` allowed | Seedance trio |
| `VIDEO_QUALITY` | `Literal["", "standard", "hd"]` | `""` | passthrough `quality` | `"quality"` allowed | Sora 2 Pro |
| `VIDEO_STYLE` | `str` | `""` | passthrough `style` | `"style"` allowed | Sora 2 Pro |

### Conventions for "skip when default"

A valve set to its **default value** (`""`, `0`, `0.0`, or
`"model_default"`) is **NOT** included in the request body — the upstream
provider's own default applies. This matters because most providers
accept different defaults for the same parameter, and forcing a value
overrides them. Specifically:

- `int` / `float` valves → skipped when `0` / `0.0`.
- `str` valves → skipped when `""` (after `.strip()`).
- 3-state Literal valves (`"model_default"`, `"on"`, `"off"`) →
  `"model_default"` is the skip sentinel; `"on"` and `"off"` translate
  to `True` and `False` in the API.
- Top-level Literal valves with empty-string variant
  (`Literal["", "allow_all", …]`) → skipped when `""`.

### Routing (where each valve lands in the OpenRouter request body)

- **Top-level** request fields (`duration`, `aspect_ratio`,
  `resolution`, `size`, `seed`, `generate_audio`, `negative_prompt`,
  `frame_images`, `input_references`): set directly in the
  `/videos` POST body. The adapter's
  `_select_passthrough_key` ([video.py:732](../open_webui_openrouter_pipe/integrations/video.py#L732))
  decides this from the model's `allowed_passthrough_parameters` plus
  hardcoded core fields.
- **Provider passthrough** fields (everything else —
  `personGeneration`, `watermark`, etc.): land at top-level too, since
  OpenRouter "passes through to the provider". The pipe trusts
  OpenRouter's documented behaviour here — see
  [Provider passthrough](#provider-passthrough) for the alternative
  `provider.options.<slug>` path used for advanced overrides.
- `VIDEO_PROVIDER_OPTIONS_JSON` is the only valve that writes to
  `provider.options.<slug>.parameters.<field>` — it's the explicit
  escape hatch.

---

## The chat filter UI (UserValves)

Each video model gets its OWN filter function in Open WebUI's Functions
table. The display name in the Integrations menu is just the model name
(e.g. `Veo 3.1 Lite`) with a leading space — the leading space is an
invisible sort anchor that pins all video filters to the top of the
dropdown so they don't get lost between `OR Direct Uploads`, `OR Web
Tools`, etc.

When you open the filter's settings icon, you see the per-model knobs
listed in the [Per-model parameter reference](#per-model-parameter-reference)
above. Behaviour rules:

- A knob set to its **default value** (empty string `""`, `0`, or
  `model_default`) is **NOT sent** to OpenRouter. The model's own default
  applies.
- 3-state Literals (`model_default`, `on`, `off`) translate to the
  provider as: not-sent / `True` / `False` respectively.
- Numeric knobs (`Seed`, `Conditioning scale`) are skipped when set to 0.
- String knobs (`Style`, `Shot type`, `Reference video URL`, etc.) are
  skipped when blank.
- The `Provider options JSON` knob accepts a raw JSON object keyed by
  provider slug — see [Provider passthrough](#provider-passthrough).
- The Frames knob has 4 modes; meaning:
  - `auto`: if you attach images, the first becomes `first_frame` (and if
    the model supports `last_frame` AND you attached more, the last
    becomes `last_frame`).
  - `none`: ignore attached images for frame conditioning. They stay in
    chat as normal attachments.
  - `first_only`: even if multiple images are attached, only the first is
    used as `first_frame`.
  - `first_last`: explicitly attach two images as start and end keyframes.

The filter is **always-on by default** for its model
(`AUTO_DEFAULT_VIDEO_FILTERS`). Disabling it for a single chat usually
means the model can't function — these filters provide essential per-model
parameters, not optional ergonomics. If you do disable, the model receives
defaults for everything.

---

## The `help` command

In any video-model chat, send a single message containing only the word
`help` (case-insensitive). The pipe short-circuits before submitting a
generation job and returns the model's help blurb directly:

- **Best known for**: research-grounded paragraph (signature features,
  use cases, position vs siblings).
- **Output capabilities**: live durations, aspect ratios, resolutions,
  frame controls, audio, seed — read from the live catalog.
- **Knobs in this filter**: every UserValve the model exposes, with a
  one-sentence description tailored to this model.
- **Tips & pitfalls**: 3–4 practical bullets — what works, what fails,
  prompt patterns.
- **Cost** (live): every SKU rate from the model's `pricing_skus` dict,
  formatted as readable bullets (e.g. `per second (with audio, 4K): $0.30`).

Help blurbs are stored statically in
[`integrations/video_help.py`](../open_webui_openrouter_pipe/integrations/video_help.py)
and rendered through a template that pulls live data from the catalog at
every call. So if OpenRouter changes a SKU rate, the help text reflects
it the next time you type `help` — no code change needed.

---

## Frame images and image-to-video

Most models accept up to two images as **frame references** that anchor
the start and/or end of the generated clip. To use this:

1. Attach images via the paperclip icon in chat (or drag-drop).
2. Set the filter's `Frames` knob to `auto`, `first_only`, or
   `first_last` (depending on intent).
3. Send your prompt. The pipe encodes each image as a base64 data URL,
   wraps it in OpenRouter's `frame_images[]` schema, and submits.

Constraints (admin-tunable):

- **`VIDEO_FRAME_IMAGE_MAX_BYTES`** (default 12 MB): per-image decoded
  size cap. Oversized images fail before submission.
- **`VIDEO_FRAME_TOTAL_MAX_BYTES`** (default 50 MB): combined cap across
  all frames in one request.
- **`VIDEO_FRAME_IMAGE_MIME_ALLOWLIST`** (default
  `image/jpeg,image/png,image/webp`): wrong-MIME images fail before
  submission.

Per-model frame support:

| Model | first_frame | last_frame |
|-------|:-----------:|:----------:|
| Veo 3.1 / Fast / Lite | ✅ | ✅ |
| Kling Video O1 | ✅ | ✅ |
| Hailuo 2.3 | ✅ | ❌ |
| Wan 2.7 | ✅ | ✅ |
| Wan 2.6 | ✅ | ❌ |
| Seedance 2.0 / 2.0 Fast / 1.5 Pro | ✅ | ✅ |
| Sora 2 Pro | ❌ | ❌ |

If you attach an image to a Sora chat, it's not used as a frame — Sora's
catalog has no `supported_frame_images`. The image stays in chat as a
regular attachment.

---

## Multimodal references (Wan 2.7)

Wan 2.7 is the only catalog model that exposes a full multi-reference
workflow. It accepts:

- **Reference video URL** (`video` passthrough): single video for motion
  / camera language / vocal-identity transfer.
- **Reference videos JSON** (`videos` passthrough): JSON array of up to
  5 reference videos. Each entry may be a URL string or an object the
  upstream Alibaba API accepts.
- **Reference images JSON** (`images` passthrough): JSON array — Wan
  2.7's 9-image structured grid for identity/wardrobe/props/environment
  anchoring without describing them in text.
- **Audio reference URL** (`audio` passthrough): voice timbre + lip-sync
  conditioning. The basis for Wan 2.7's multi-language lip-sync.
- **Last image URL** (`last_image` passthrough): closing-frame anchor;
  used together with `first_frame` for controlled in-between motion.

JSON arrays must be valid JSON. Example for `Reference images JSON`:

```json
["https://example.com/character.png", "https://example.com/style-ref.png"]
```

If the array is malformed JSON the filter raises an error before
submission and the chat shows a clear failure message — no half-submitted
job.

---

## Provider passthrough

The OpenRouter `/videos` API accepts a `provider.options.<slug>` block
that is "spread into the upstream request body". This lets each provider
expose model-specific parameters that aren't part of the universal core
fields.

For each model, the typed valves exposed in the filter UI cover all the
parameters listed in OpenRouter's `allowed_passthrough_parameters` for
that model. So most users never need to write raw JSON.

For advanced users or future fields not yet typed, the
`Provider options JSON` valve accepts a raw object keyed by provider slug:

```json
{
  "google-vertex": {
    "parameters": {
      "experimentalFlag": "value"
    }
  }
}
```

The pipe deep-merges this into `provider.options` after typed valves are
written. The filter normalises the shape — you can write the inner object
either with or without an explicit `parameters` wrapper, and the filter
wraps it correctly per the
[Phase 0 probe](#phase-0-probe) decision.

Chat-routing fields (`order`, `sort`, `max_price`, `zdr`) are NEVER
forwarded to `/videos`. Only model-relevant `options` survive.

---

## Pricing and cost display

OpenRouter publishes per-SKU rates in each model's catalog `pricing_skus`
dict. Examples:

```json
"pricing_skus": {
  "duration_seconds_with_audio": "0.40",
  "duration_seconds_without_audio": "0.20",
  "duration_seconds_with_audio_4k": "0.60",
  "duration_seconds_without_audio_4k": "0.40"
}
```

The pipe surfaces these in two places:

1. **In-chat `help` command** — bullets each SKU as "per second (with
   audio, 4K): $0.60". Read live from the catalog every time `help` is
   invoked.
2. **Final status footer** after generation — shows the actual usage
   cost from OpenRouter's poll response (e.g. `Generated in 35.2s ·
   $0.40`). This is the authoritative cost for that specific
   generation.

SKU key conventions decoded:

| Suffix / prefix | Meaning |
|------------------|---------|
| `duration_seconds` | Per second of video |
| `video_tokens` | Per video token (Seedance pricing model) |
| `_with_audio` / `_without_audio` | With or without generated audio |
| `_4k` / `_1080p` / `_720p` / `_480p` | Resolution-tiered SKU |
| `text_to_video_` / `image_to_video_` | Generation mode (Wan 2.6) |

Prices are read live every call — never baked into static help text — so
OpenRouter rate updates surface without a new bundle.

---

## Output rendering and message format

When generation succeeds, the assistant message contains:

```markdown
[openrouter:v1:videojob:<job_id>]: #
[openrouter:v1:videomodel:<model_id>]: #

<video>
/api/v1/files/<owui_file_id>/content
</video>

*Generated in <elapsed>s · $<cost>*
```

The two `[label]: #` lines are CommonMark **reference-link definitions**.
They render as nothing — they are invisible markers used internally for
[resume](#resume-recovery-and-disconnect-resilience). The marked.js
parser treats them as label-only references with no body, so they don't
appear in the rendered chat.

The `<video>` tag with the URL on its own line is the only format that
marked.js tokenises as a single CommonMark "type 7 HTML block". Without
the blank lines and the URL on a separate line, marked either fragments
the block into 3 inline tokens (rendering as text) or merges adjacent
`<video>` blocks into one HTML token (HTMLToken's non-greedy regex then
matches only the first, hiding the rest).

The trailing `*Generated in ...*` line ends with `\n` — defensive against
any later concatenation that could smash markers from a follow-up message
into inline text.

The video file itself is stored in Open WebUI's file storage backend
(local, S3, GCS, or Azure depending on `STORAGE_PROVIDER`), inserted
into the `files` table, and linked to the chat message via
`Chats.insert_chat_files`. The file appears in the chat's Files panel
and can be downloaded directly.

---

## Resume, recovery, and disconnect resilience

The `[openrouter:v1:videojob:<job_id>]: #` marker is the recovery
mechanism. The marker is **persisted on submit**: immediately after
the adapter receives a job_id from OpenRouter (and before the bg poll
loop starts), the adapter emits an OWUI socket `'message'` event with
a pending content block. OWUI's socket handler routes that event to
`Chats.upsert_message_to_chat_by_id_and_message_id`, so the marker
appears in the chat DB even if the pipe process dies a second later.
At end-of-stream, OWUI's stream finalizer overwrites the message with
the final success/failure content (a full replacement, not an append),
so the pending marker is cleanly replaced — no flash, no duplication.

Every time `pipe()` is invoked for a video chat:

1. The adapter looks up the assistant message and scans for an existing
   marker.
2. If a marker is found AND a final `<video>` block also exists, the
   adapter returns the cached content (no re-poll, no double-submit).
3. If a marker exists but no `<video>` block, the adapter resumes
   polling that job_id — skipping submission.
4. If no marker, the adapter submits a new job AND emits the pending
   marker via the `'message'` event before spawning the bg poll loop.

This handles:

- **Browser refresh** during generation — reload picks up where it left
  off.
- **OWUI process restart** — the marker survives in the message body
  (DB-persisted via the on-submit `'message'` emit). When the user
  re-engages the chat and triggers a new pipe call, the resume path
  picks up the marker and re-polls. The pipe does NOT proactively scan
  chat history at startup for orphan jobs — recovery is user-driven on
  the next request. (Auto-startup-recovery is a v1.1 concern.)
- **Client disconnect mid-poll** — the in-process bg task continues. Its
  result populates the active-task registry. When the user returns and
  the chat re-fires `pipe()`, the resume path sees the in-flight or
  completed bg task and delivers the result.

What does NOT survive:

- **`local:` chats** (chat IDs starting with `local:`): Open WebUI does
  not persist these to chat storage, so markers can't be written. The
  on-submit `'message'` emit is skipped explicitly for `local:` chats.
  They complete in-process but aren't recoverable across process
  restarts.
- **OpenRouter job expiry**: OpenRouter videos expire after a
  provider-specific window (typically days). Resuming a too-old job
  returns an `expired` terminal status which the adapter renders as a
  visible failure block.

`Pipe.close()` cancels in-process video lifecycle tasks during pipe
restart or OWUI shutdown. OpenRouter does not expose a cancel endpoint
for these jobs — persisted markers are the recovery mechanism on the
next request.

---

## Concurrency limits

Two valves cap simultaneous generations:

- **`MAX_CONCURRENT_VIDEO_GENS`** (default 2): global cap per pipe
  process. Implemented as a class-level lazy `asyncio.Semaphore`. When
  exhausted, new requests wait in the semaphore queue (chat shows
  "Waiting for video slot...").
- **`MAX_CONCURRENT_VIDEO_GENS_PER_USER`** (default 2): per-user cap.
  Implemented as a counter + per-user lock. Exceeding the cap returns
  an immediate visible error in chat — the user must wait for one of
  their existing jobs to complete.

If two requests target the same `(chat_id, message_id)` (e.g. a user
hits send twice on the same message slot), the active-task registry
deduplicates: one is the **owner** (does the work), the other is a
**waiter** (awaits the owner's bg task and emits the result on its own
chat connection). This holds even across browser tabs.

Single-worker only: `_video_active_tasks` is process-local. Multi-worker
deployments would lose the dedupe guarantee — that's why this is a
single-worker constraint and documented as such. Multi-worker exact-once
would need a Redis lock and is a v1.1+ concern.

---

## Configuration valves (admin)

Nineteen valves control the video subsystem. All are visible in Admin →
Functions → OpenRouter pipe → Valves.

| Valve | Default | Range | Purpose |
|-------|---------|-------|---------|
| `ENABLE_VIDEO_GENERATION` | `True` | bool | Master kill switch. False removes all video models from `pipes()` output. |
| `AUTO_INSTALL_VIDEO_FILTERS` | `True` | bool | Install per-model filter rows in OWUI Functions table on `pipes()`. |
| `AUTO_ATTACH_VIDEO_FILTERS` | `True` | bool | Attach each filter to its corresponding video model row. |
| `AUTO_DEFAULT_VIDEO_FILTERS` | `True` | bool | Keep per-model filter enabled by default per chat (**re-asserted on every catalog metadata sync** — admins who manually disable a filter will see it re-defaulted on the next sync; set to `False` to opt out). |
| `VIDEO_INITIAL_POLL_DELAY_SECONDS` | `5.0` | 0.0–60.0 | Wait before the first poll on a freshly submitted job. |
| `VIDEO_POLL_INTERVAL_SECONDS` | `5.0` | 1.0–60.0 | Base polling interval. |
| `VIDEO_POLL_BACKOFF_FACTOR` | `1.2` | 1.0–4.0 | Multiplier applied to the interval after each non-terminal poll. |
| `VIDEO_POLL_INTERVAL_MAX_SECONDS` | `20.0` | 1.0–120.0 | Cap on the polling interval after backoff. |
| `VIDEO_MAX_POLL_TIME_SECONDS` | `600` | 30–7200 | Max wall-clock time before failing the lifecycle with a timeout error. |
| `VIDEO_STATUS_POLL_MAX_ERRORS` | `5` | 1–25 | Tolerable consecutive transient poll errors before failing. |
| `REMOTE_VIDEO_MAX_SIZE_MB` | `500` | 1–2048 | Max downloaded video size; oversized aborts streaming. |
| `VIDEO_DOWNLOAD_CHUNK_SIZE` | `1048576` | 65536–8388608 | Chunk size in bytes for streaming download. |
| `MAX_CONCURRENT_VIDEO_GENS` | `2` | 1–100 | Global concurrency cap per pipe process. |
| `MAX_CONCURRENT_VIDEO_GENS_PER_USER` | `2` | 1–25 | Per-user concurrency cap. |
| `VIDEO_FRAME_IMAGE_MAX_BYTES` | `12_582_912` (12 MB) | 65536–67108864 | Per-image decoded size cap. |
| `VIDEO_FRAME_TOTAL_MAX_BYTES` | `52_428_800` (50 MB) | 65536–134217728 | Combined frame-bytes cap across one request. |
| `VIDEO_FRAME_IMAGE_MIME_ALLOWLIST` | `image/jpeg,image/png,image/webp` | comma-list | Allowed MIMEs for frame images. |
| `VIDEO_OUTPUT_MIME_ALLOWLIST` | `video/mp4,video/webm` | comma-list | Allowed MIMEs for downloaded video (sniffed from prefix). |
| `VIDEO_FILTER_MARKER` | `openrouter_pipe:video_filter:v1` | string | Internal marker for filter identification. Don't change unless you know why. |

Tuning hints:

- **High-volume deployments** with many users: bump
  `MAX_CONCURRENT_VIDEO_GENS` (process-wide cap) but keep
  `MAX_CONCURRENT_VIDEO_GENS_PER_USER` low (per-user fairness). Watch
  memory pressure — each lifecycle pins a temp file ~50 MB to ~500 MB.
- **Long jobs** (Sora 20s clips): bump `VIDEO_MAX_POLL_TIME_SECONDS` to
  e.g. `1200` (20 min) so jobs don't time out before completion.
- **Slow networks** to OpenRouter: bump `VIDEO_POLL_INTERVAL_MAX_SECONDS`
  to reduce poll storm.
- **Smaller storage budgets**: lower `REMOTE_VIDEO_MAX_SIZE_MB` to
  reject oversized clips before they hit your file backend.

---

## Errors and troubleshooting

### "AUTO_ATTACH_VIDEO_FILTERS is enabled but no OpenRouter Video Generation filters are installed"

The catalog manager couldn't ensure per-model filter installs. Causes:

- `AUTO_INSTALL_VIDEO_FILTERS` is False — turn it on.
- The pipe's API key is invalid — `pipes()` exited early before installing.
- Open WebUI's `Functions` table is read-only or has a permission issue
  for the pipe's user context.

### Filter is in Filters list but not toggled on

The per-model filter row exists but isn't auto-attached to the model.
Either `AUTO_ATTACH_VIDEO_FILTERS` is `False`, or the catalog metadata
sync hasn't run since the last filter install. Toggle the auto-attach
valve off → save → on → save to force a resync, or restart the pipe.

### "Generated video is empty" / "Generated video temp file is missing"

The download step failed mid-stream or wrote zero bytes. Check:

- OpenRouter job status was actually `completed` (not `failed`/`expired`).
- Downstream storage (`STORAGE_PROVIDER`) is healthy and writable.
- The pipe process has filesystem write permission to its temp dir.

### "The selected video model does not accept frame images"

You attached an image to a chat with a model that has no
`supported_frame_images` (e.g. Sora 2 Pro) AND set Frames to anything
other than `none`. Either pick a frame-capable model or set Frames =
`none`.

### "Frame image MIME 'application/octet-stream' is not allowed"

The attached image's content_type wasn't in
`VIDEO_FRAME_IMAGE_MIME_ALLOWLIST`. The pipe sniffs MIME from the file
record — if OWUI stored it with a generic content type, re-attach via
the chat input rather than via URL ingestion.

### Generation status shows "expired"

OpenRouter timed out the upstream provider and discarded the job. The
adapter renders a visible failure block. Re-submit the prompt to start
a fresh job.

### "Multiple OpenRouter Video Generation filter candidates found"

Log warning. Indicates two or more filter rows match the marker for the
same model. The pipe uses the most recently updated one. Manually
delete duplicates from Admin → Functions if you want to clean up.

### Chat reload after disconnect shows duplicate video

Was a bug in earlier versions where the bg task and outer adapter both
emitted the same content. Fixed in the current bundle — the bg task no
longer emits, only the outer/waiter does. If you still see duplication,
ensure you're running the latest bundle.

### Two `<video>` players for one generation

Same root cause as above (duplicate emit). Fix is in the current bundle.
Symptom in older bundles: chat assistant message contains the entire
content block twice, with the second copy's marker `[openrouter:v1:videojob:...]`
visibly leaked because there's no newline before it.

---

## Architecture overview

Roughly, in order of who-calls-who:

```
pipe()
  └─ orchestrator dispatches to VideoGenerationAdapter.generate() if model.features has "video_generation"
        ├─ help short-circuit (prompt == "help" → render_video_help)
        ├─ resume check (read message → scan for [videojob:...] marker)
        ├─ acquire user slot + global semaphore
        ├─ submit job via VideoGenClient.submit() (returns job_id)
        ├─ emit pending content via OWUI socket 'message' event
        │      (routed to Chats.upsert_message_to_chat_by_id_and_message_id —
        │       persists the [videojob:<id>] marker BEFORE the bg task starts)
        ├─ spawn _run_lifecycle_after_submit() as bg asyncio.Task
        │     ├─ poll with backoff until terminal status
        │     ├─ download generated video (streaming, bounded)
        │     ├─ MIME-sniff against VIDEO_OUTPUT_MIME_ALLOWLIST
        │     ├─ stream-upload to OWUI storage (per-backend: Local/S3/GCS/Azure)
        │     ├─ insert Files row + link to chat
        │     ├─ build success content (markers + <video> + footer)
        │     └─ return VideoLifecycleResult — does NOT emit
        ├─ outer awaits bg task with asyncio.shield (survives client disconnect)
        ├─ outer emits status footer + chat:completion (the SOLE emit)
        └─ outer returns content string
              └─ functions.py wraps as SSE chunk, OWUI middleware accumulates,
                 stream finalizer upserts to message DB (one write).
```

Key invariant: **exactly one `_emit_completion` per `(chat_id,
message_id)`**. The bg task does the work and returns the result;
the outer (or waiter for de-duped re-entries) is the sole emitter. This
prevents the duplicate-content / leaked-marker bug class.

Key files:

- [`integrations/video.py`](../open_webui_openrouter_pipe/integrations/video.py)
  — `VideoGenerationAdapter` (entry point, lifecycle, emit).
- [`integrations/video_client.py`](../open_webui_openrouter_pipe/integrations/video_client.py)
  — HTTP client for `/videos/*` endpoints.
- [`integrations/video_catalog.py`](../open_webui_openrouter_pipe/integrations/video_catalog.py)
  — fetches `/videos/models` and registers them in
  `OpenRouterModelRegistry`.
- [`integrations/video_help.py`](../open_webui_openrouter_pipe/integrations/video_help.py)
  — per-model help blurbs + live pricing renderer.
- [`integrations/video_types.py`](../open_webui_openrouter_pipe/integrations/video_types.py)
  — `VideoLifecycleResult`, `DownloadedVideo` dataclasses.
- [`filters/video_filter_renderer.py`](../open_webui_openrouter_pipe/filters/video_filter_renderer.py)
  — generates the per-model OWUI filter source code.
- [`filters/filter_manager.py`](../open_webui_openrouter_pipe/filters/filter_manager.py)
  — installs filter rows in OWUI Functions table.
- [`storage/video_persistence.py`](../open_webui_openrouter_pipe/storage/video_persistence.py)
  — thin resume-path helper that reads the persisted chat message to
  detect prior `videojob` markers. The upload/storage/link path lives
  in [`storage/multimodal.py`](../open_webui_openrouter_pipe/storage/multimodal.py)
  via `_download_remote_url_streaming` + `_upload_to_owui_storage_from_path`
  + `_try_link_file_to_chat`, reused by both image-gen and video-gen.
- [`models/registry.py`](../open_webui_openrouter_pipe/models/registry.py)
  — `register_video_models()` merges video models into the chat catalog.
- [`models/catalog_manager.py`](../open_webui_openrouter_pipe/models/catalog_manager.py)
  — metadata sync that attaches and defaults filters.
- [`core/config.py`](../open_webui_openrouter_pipe/core/config.py)
  — Valve definitions.

---

