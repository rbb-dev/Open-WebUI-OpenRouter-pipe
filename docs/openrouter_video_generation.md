# OpenRouter Video Generation

This pipe exposes OpenRouter's twenty-two async video-generation models as
selectable chat models in Open WebUI. You pick a video model in the chat
header (just like any other LLM), type a prompt, and the pipe submits a
job, polls until completion, downloads the generated video into Open WebUI
file storage, and renders it inline with a `<video>` tag.

The feature is on by default (`ENABLE_VIDEO_GENERATION=True`). If you want
to disable it, set that valve to `False` in Admin → Functions → OpenRouter
pipe → Valves.

## Table of contents

- [Quickstart](#quickstart)
- [Video models](#video-models)
- [Per-model deep dive](#per-model-deep-dive)
- [Per-model parameter reference](#per-model-parameter-reference)
- [Filter UserValve identifiers (master reference)](#filter-uservalve-identifiers-master-reference)
- [The chat filter UI (UserValves)](#the-chat-filter-ui-uservalves)
- [The `help` command](#the-help-command)
- [Frame images and image-to-video](#frame-images-and-image-to-video)
- [Attachments that are not frames](#attachments-that-are-not-frames)
- [Multimodal references (Wan 2.7)](#multimodal-references-wan-27)
- [Provider passthrough](#provider-passthrough)
- [Pricing and cost display](#pricing-and-cost-display)
- [Output rendering and message format](#output-rendering-and-message-format)
- [Resume, recovery, and disconnect resilience](#resume-recovery-and-disconnect-resilience)
- [Concurrency limits](#concurrency-limits)
- [Configuration valves (admin)](#configuration-valves-admin)
- [Errors and troubleshooting](#errors-and-troubleshooting)
- [Architecture overview](#architecture-overview)

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
7. The chat shows a status line while the job runs — submitting, then
   whatever the provider reports while it works, then downloading — and
   a closing line when it finishes. Where a charge above zero is
   reported it lands on that final status line, as long as usage details
   are on: that is your own Show usage details setting once you have set
   it, and the site default your administrator chooses until then; with
   usage details off the line carries the elapsed time alone. Generation
   typically takes 30s–4min depending on the model and duration.
8. The final message renders an inline video player. Click play.

### Model-specific help in chat

Typing the literal word `help` (no other text) into a chat against any
video model returns a research-grounded model-specific help blurb covering:

- What the model is best known for
- Output capabilities (durations, aspect ratios, resolutions, frames, audio, seed)
- Every filter knob exposed for this model and what it does
- 3–4 tips and pitfalls

It quotes no rates: what a model charges is on OpenRouter's pricing page.

This is the fastest way to learn a model without leaving the chat. Try
it on each video model — the answers are different for every one.

### For administrators

Out-of-the-box defaults are sensible for most deployments:

```
ENABLE_VIDEO_GENERATION = True
AUTO_INSTALL_VIDEO_FILTERS = True       # creates per-model filter rows
AUTO_ATTACH_VIDEO_FILTERS = True        # attaches each filter to its model
AUTO_DEFAULT_VIDEO_FILTERS = True       # filter is on-by-default per chat
MAX_CONCURRENT_VIDEO_GENS = 2           # global cap per pipe process
MAX_CONCURRENT_VIDEO_GENS_PER_USER = 2  # per-user cap
DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS = True   # see below
```

#### Built-in tools on video models

A video model answers with a clip, not a tool call. Offering it Open WebUI's
built-in tools usually ends in a turn that fails or comes back empty, and the
cause is hard to spot because the model's `Built-in tools` box still looks
ticked.

With `DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS` on, which is the default, the pipe
unticks that box on each video model at the moment it first adds the model, so
the state is visible on the model's page rather than being applied invisibly at
request time. Tick it back on for a model if you want tools there — your choice
is kept, because the pipe fills this setting in only where a model has none yet.

The same setting covers image models. It needs `UPDATE_MODEL_CAPABILITIES` on,
since that is the switch that lets the pipe write to capability boxes at all.

#### File context on video models

`UPDATE_MODEL_CAPABILITIES` also unticks Open WebUI's `File context` box on
video and image models, whatever `DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS` is set
to. Left on — Open WebUI's own default — an attachment makes Open WebUI run an
extra billed round-trip that turns the conversation into search queries and
pastes the retrieved text into what was meant to be a video prompt. As with the
tools box, the pipe fills it in only where a model has no setting yet.

Turning it off has a second, wanted effect: attachments a video model was sent
then stay in the chat as normal attachments as well, instead of being taken out
of the request. See
[Attachments that are not frames](#attachments-that-are-not-frames).

If the per-model filters do not appear in the Integrations menu, check:
- `AUTO_INSTALL_VIDEO_FILTERS` and `AUTO_ATTACH_VIDEO_FILTERS` are both
  `True`. While `AUTO_INSTALL_VIDEO_FILTERS` is off, an already-installed
  filter is never rewritten, so a fix shipped in a newer release is not
  delivered; the pipe writes a warning to its log naming any filter whose
  stored version is out of date.
- The pipe has been called at least once with a logged-in user.
- Open WebUI's own Admin → Functions screen lists one entry per catalogued video model,
  named ` Veo 3.1 Lite`,
  ` Seedance 2.0`, etc. (note the leading space — that's intentional, see
  [The chat filter UI](#the-chat-filter-ui-uservalves)).

**Access control for non-admin users.** Video models are inserted PRIVATE
by default per the standard `NEW_MODEL_ACCESS_CONTROL` valve (default
`admins`). Non-admin users will not see video models in the picker until
an admin explicitly grants access via Admin → Models → [video model row]
→ Access. The pipe's auto-attach and auto-default behaviour fully
prepares the model row beforehand (filter wired, defaulted on, ready to
generate), so the per-model access grant is the only manual step
required. This is intentional policy — video generation is heavyweight
enough that operators usually want admin-curated access.

**Auto-default re-assert.** The per-model filter is re-defaulted to
enabled on every catalog metadata sync (typically every pipe `pipes()`
call). If you manually disable a video filter for a chat, the next sync
will re-default it. Set `AUTO_DEFAULT_VIDEO_FILTERS=False` to opt out
of the re-assert.

See [Configuration valves](#configuration-valves-admin) for the full list of video valves.

---

## Video models

| Model id | Display name | Best for | Audio | Seed | Frames |
|----------|--------------|----------|:----:|:----:|:------:|
| `google/veo-3.1` | Google: Veo 3.1 | Flagship hero shots; best prompt adherence; native synchronised audio with ~120ms lip-sync; up to 4K. | ✅ | ✅ | first + last |
| `google/veo-3.1-fast` | Google: Veo 3.1 Fast | Drafting/iteration at close to Veo 3.1 quality; A/B-testing concepts; image-to-video. | ✅ | ✅ | first + last |
| `google/veo-3.1-lite` | Google: Veo 3.1 Lite | Lightest Veo tier; high-volume / batch / consumer-app integrations; same speed as Fast. | ✅ | ✅ | first + last |
| `kwaivgi/kling-video-o1` | Kling: Video O1 | Cinematic film-grade clips, character/identity consistency, physics-aware human motion. No deterministic seed. | ✅ | ❌ | first + last |
| `kwaivgi/kling-v3.0-pro` | Kling: Video v3.0 Pro | Top tier of Kling v3.0 — higher visual quality and motion fidelity than Standard; granular 3–15s clips; first/last-frame anchoring. New `cfg_scale` knob. No deterministic seed. | ✅ | ❌ | first + last |
| `kwaivgi/kling-v3.0-std` | Kling: Video v3.0 Standard | Standard tier of Kling v3.0 — same capability matrix as Pro; granular 3–15s clips; first/last-frame anchoring. New `cfg_scale` knob. No deterministic seed. | ✅ | ❌ | first + last |
| `minimax/hailuo-2.3` | MiniMax: Hailuo 2.3 | State-of-the-art human physics and emotional micro-expressions; fluid + cloth + fire dynamics. **Silent — no audio.** | ❌ | — | first only |
| `minimax/hailuo-3` | MiniMax: H3 | Lightweight open-weights model for instruction-guided edits and controlled content; the one that renders legible text and brand marks. 2K only. | ✅ | ❌ | first + last |
| `alibaba/wan-2.7` | Alibaba: Wan 2.7 | Image-grid reference control, generated lip-sync across languages, FLF2V. Tuned for character-led narrative. Clip and voice references are published but not declared as input, so they are not offered. | ✅ | ✅ | first + last |
| `alibaba/wan-2.6` | Alibaba: Wan 2.6 | Feature-rich Wan tier with multi-shot storyboarding, 24fps, dialogue + lip-sync, shot_type cinematography. **First-frame only.** | ✅ | ✅ | first only |
| `bytedance/seedance-1-5-pro` | ByteDance: Seedance 1.5 Pro | First Dual-Branch DiT with native unified video+audio, multilingual lip-sync, 21 exact pixel sizes. | ✅ | ✅ | first + last |
| `bytedance/seedance-2.0` | ByteDance: Seedance 2.0 | Universal Reference (text + 9 images + 3 video/audio), best character consistency for branded/series content. | ✅ | ✅ | first + last |
| `bytedance/seedance-2.0-fast` | ByteDance: Seedance 2.0 Fast | Speed-optimised Seedance 2.0; 480p/720p only; ideal for drafts and bulk pipelines. | ✅ | ✅ | first + last |
| `bytedance/seedance-2.5` | ByteDance: Seedance 2.5 | Longest single take in the catalogue at 30s; long-form storytelling, reference-driven generation, editing and extending existing clips. 480p/720p. | ✅ | ✅ | first + last |
| `openai/sora-2-pro` | OpenAI: Sora 2 Pro | Physics-accurate motion + world-state persistence across multi-shot sequences. 20s clips. **Text-only — no frame images.** | ✅ | ❌ | none |
| `x-ai/grok-imagine-video` | SpaceXAI: Grok Imagine Video | Fast iteration with per-second duration control (any integer 1–15s, 24fps); 7 aspect ratios; image-to-video via first frame. | — | — | first only |
| `x-ai/grok-imagine-video-1.5` | SpaceXAI: Grok Imagine Video 1.5 | Same per-second granularity and seven framings, now up to 1080p, so drafting rough and finishing sharp is one control change. No provider parameters at all. | — | — | first only |
| `black-forest-labs/flux-3-video` | Black Forest Labs: FLUX.3 Video | Keyframe-driven shots with opening and closing stills, and continuation of an existing clip so long sequences can be built a segment at a time. Up to 20s at 1080p. | ✅ | ❌ | first + last |
| `runway/gen-4.5` | Runway: Gen-4.5 | Cinematic text- and image-to-video with strong motion and close prompt adherence; deliberately narrow — 720p, 16:9 or 9:16, 2–10s. | ❌ | ✅ | first only |
| `runway/aleph-2` | Runway: Aleph 2.0 | In-context **video editor**: applies an instruction across footage you attach while leaving the rest untouched. Length and size come from your clip, not from a control. | ❌ | ✅ | none |
| `alibaba/happyhorse-1.1` | Alibaba: HappyHorse 1.1 | Unusually wide framing range — the usual five plus ultrawide 21:9 and tall 9:21 — in 3–15s clips at 720p or 1080p; 1.1 refines 1.0 with the same controls. | — | ✅ | first only |
| `alibaba/happyhorse-1.0` | Alibaba: HappyHorse 1.0 | The first HappyHorse tier, same controls and framings as 1.1; reach for it only to pin 1.0's exact generation behaviour. | — | ✅ | first only |

A `—` in the Audio or Seed column means OpenRouter publishes nothing
either way for that model. The control is still offered, and whatever the
model does by default is what you get — so test one short clip rather than
assuming.

Pick model selection rules of thumb:

- **Speed matters most** → Veo 3.1 Lite, Seedance 2.0 Fast, or Grok Imagine Video, whose 1-second duration granularity lets a draft be as short as you ask for.
- **Hero shot for client work** → Veo 3.1 (full) or Seedance 2.0.
- **Multi-shot story with consistent characters** → Wan 2.7 or Seedance 2.0.
- **Dialogue / lip-sync** → Wan 2.7 or Seedance 1.5 Pro, both generating the voice from your prompt. No model in this catalog declares audio input, so conditioning on a voice you supply is not available through the pipe.
- **Physics realism / human motion** → Sora 2 Pro, Hailuo 2.3.
- **Longest clip** → Seedance 2.5 (30s), then FLUX.3 Video and Sora 2 Pro (20s each).
- **Long sequence in pieces** → FLUX.3 Video, which continues an existing clip so you can build and redirect a segment at a time.
- **Exact clip length (e.g. precisely 7s)** → most models here take any whole number of seconds inside their range; only Veo 3.1 (all three tiers), Kling O1, Hailuo 2.3, Wan 2.6 and Sora 2 Pro are locked to fixed steps. For anything shorter than 2s, the Grok Imagine Video tiers are the only ones that start at 1.
- **Editing footage you already have** → Aleph 2.0 is built for it: attach the clip and describe the change. H3 and Seedance 2.5 also take an instruction against footage you supply, alongside the scenes they generate from scratch. All three need an administrator to have turned on sending media to a file host before an attached clip can reach the model at all.
- **Legible text or a brand mark in shot** → H3, which is built for controlled rendering rather than free-running scenes.
- **Ultrawide framing (21:9)** → nine models offer it: both HappyHorse tiers, all four Seedance, FLUX.3 Video, H3 and Aleph 2.0.
- **Very tall framing (9:21)** → five: both HappyHorse tiers and Seedance 1.5 Pro, 2.0 and 2.0 Fast. Seedance 2.5 does not publish it.
- **No audio needed** → Hailuo 2.3, Gen-4.5 and Aleph 2.0 are silent, or set `Audio = off` on Veo Lite.

---

## Per-model deep dive

This section is a written-up companion to the in-chat `help` command, not a
copy of what it prints. It covers what each model is for and how to prompt
it. `help` covers that too, and then adds what the model can output, the
controls its panel draws — all read from OpenRouter when you ask, so
where the two disagree, `help` is the one that is current.
Skip to a model that matches your use case, or read them all to get a feel
for the catalog. The descriptions draw on what OpenRouter publishes about
each model together with public research on its reputation, papers, and
signature features.

### Google: Veo 3.1

> **id**: `google/veo-3.1`

Google DeepMind's flagship video model, positioned for production-quality
output where visual fidelity is the priority — commercial deliverables,
hero shots, and cinematic sequences. Its standout trait is jointly-diffused
native audio: dialogue, SFX, and ambience are generated alongside the
video in a single pass with lip-sync within roughly 120ms. Compared to
Fast and Lite, the full tier delivers sharper motion, stronger prompt
adherence, finer texture/lighting detail, and access to 4K output. It also leads
MovieGenBench evaluations on
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

The speed-optimised tier of Veo 3.1, generating 4-, 6-, or 8-second
clips up to 4K with native synchronised audio, rendering faster than
full Veo 3.1. Editor blind tests
put its quality close to the full tier's, making
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
  turning audio off avoids out-of-sync lip movement on talking heads.
- At 8s, faces and logos can drift partway through. Lock with reference
  text, reuse a seed when iterating, and use the negative prompt to
  exclude common failures ("no text overlays, no extra fingers, no
  warped logos").

### Google: Veo 3.1 Lite

> **id**: `google/veo-3.1-lite`

Google's lightest Veo 3.1 tier, positioned for high-volume video
applications and rapid iteration. It matches Veo 3.1 Fast's latency —
making it the go-to pick for batch pipelines, social automation, and
consumer-app integrations. Tradeoffs: a hard cap at
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

---

### Kling: Video v3.0 Pro

> **id**: `kwaivgi/kling-v3.0-pro`

Kuaishou's top tier of Kling v3.0 and the highest-quality Kling SKU
here — sharper detail, stronger character consistency and richer motion
than Standard. Clips run 3 to 15 seconds at 720p in 16:9, 9:16 or 1:1,
with both endpoints anchorable and native audio. Best for hero shots,
marketing deliverables and pre-vis where quality is the priority. Pro
and Standard publish an identical knob set; the difference is output
quality, so iterate on Standard and finish here.

**Tips & pitfalls**

- Kling responds to cinematic intent: describe the camera move (slow
  dolly-in, tracking), the motion physics and the end state, rather than
  listing what is in the frame.
- No seed is exposed (`seed: false`), so re-running the same prompt does
  not reproduce the same clip. Lock the look with `first_frame` /
  `last_frame` and the negative prompt instead.
- `cfg_scale` is new in v3.0 and absent from the older O1 SKU. Leave it at
  0 for the provider default, or raise it when the prompt must be followed
  strictly at the cost of creative variation.

---

### Kling: Video v3.0 Standard

> **id**: `kwaivgi/kling-v3.0-std`

The standard tier of Kling v3.0, with exactly the same capability
surface as Pro — 3-to-15-second clips at 720p in 16:9, 9:16 or 1:1, both
endpoints anchorable, native audio. Best for prompt iteration, drafts
and bulk runs where throughput matters more than the last few percent of
polish.

**Tips & pitfalls**

- Draft here and switch to Pro for finals. Because the knob set is
  identical, a prompt that works on Standard works unchanged on Pro.
- Same cinematic-intent prompting as Pro: camera move, motion physics,
  end state.
- No seed is exposed (`seed: false`) — lock the look with `first_frame` /
  `last_frame` and the negative prompt.
- `cfg_scale` behaves as it does on Pro: 0 for the provider default,
  higher for stricter prompt adherence.

---

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
  `fast_pretreatment` runs that same step more quickly with a small loss
  of quality — handy for batch runs, otherwise leave it alone.

### Alibaba: Wan 2.7

> **id**: `alibaba/wan-2.7`

Alibaba Tongyi Lab's flagship multimodal video model, unifying text,
image, audio, and video conditioning in a single 27B-parameter Diffusion
Transformer with Flow Matching. Its standout capability is
multi-reference control: lock subject identity, props and visual style
across new scenes by feeding a grid of reference images, plus last-frame
anchoring. Alibaba describes reference clips and voice conditioning for
this model, but OpenRouter reports it as taking only text and pictures,
so neither is offered here. A "Thinking
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
a 14B-parameter MoE architecture. Best known for multi-shot
1080p @ 24fps generation with synchronised native audio (multi-speaker
dialogue, lip-sync, voice/music conditioning) and intelligent multi-shot
narrative storyboarding that holds character and lighting consistency
across cuts. Pick 2.6 over 2.7 when you want the well-tuned
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
  or terse prompts — it adds camera and lighting detail for you; turn it
  OFF when you've already crafted a long, precise prompt.
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
production-validated audio-visual workflow; it offers 1080p output, the
wider 4–12s duration window, and reliable
multilingual lip-sync (Mandarin, English, Japanese, Korean, Spanish, plus
dialects).

**Tips & pitfalls**

- Toggle Audio off for silent B-roll, layout passes, or anything you'll
  dub later.
- Use 1.5 Pro for short, repeatable clips with simple camera work and
  known-good prompts; switch to 2.0 only when you need richer multimodal
  references, 4K output, or longer 15s shots — 1.5 Pro caps at 1080p
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

ByteDance's speed-optimised variant of the Seedance 2.0 family, built on
the same unified multimodal architecture but using distillation and
accelerated sampling to cut generation time relative to standard
Seedance 2.0. Best known for cinematic 480p/720p
output with native audio synchronised in a single pass, support for
text-to-video, image-to-video with first/last frame control, and
multimodal reference-to-video, plus very wide aspect-ratio coverage
including 21:9 cinematic and 9:21.

**Tips & pitfalls**

- Use Fast for drafting, prompt iteration, and bulk pipelines; switch
  to standard Seedance 2.0 for hero shots — Fast trades a small amount
  of motion refinement and detail for the shorter generation time.
- Iterate at 480p and a short duration first, then commit to the
  full-size take.
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
also runs to 20-second clips — only Seedance 2.5's 30s goes longer —
at full 1080p.

**Tips & pitfalls**

- **Text-to-video only here**: this catalog entry has no
  `frame_image` support, so you can't seed it with a start/end image —
  drive the result entirely from prompt language.
- Long takes at 1080p render slowly — community tests report 2–5
  minutes for a 20s clip and much longer at peak, so prefer 4–8s 720p
  for iteration and reserve 16–20s 1080p for finals.
- Plays to its strengths on physics, motion weight, lighting, and
  ambient/dialogue audio; struggles with on-screen text, brand logos,
  fine hand details, and highly choreographed multi-character action —
  don't ship as-is for client deliverables that depend on legible text.
- `Quality` and `Style` are passthrough hints OpenRouter forwards; the
  OpenAI Videos API itself doesn't expose a discrete quality enum
  (resolution drives the tier), so treat them as soft hints rather than
  guaranteed switches.

### SpaceXAI: Grok Imagine Video

> **id**: `x-ai/grok-imagine-video`

SpaceXAI's fast text-, image-, and reference-conditioned video generator,
producing short clips at 24fps in 480p or 720p across seven aspect
ratios. Its standout differentiator is how short it will go: any
integer from 1 to 15 seconds, starting at 1 — the lowest floor in the
catalog, where the next shortest models start at 2. That makes it well
suited to rapid iteration and high-volume production.

**Tips & pitfalls**

- Duration is any integer 1–15 seconds, so a draft can be as short as a
  single second.
- Two resolutions only: 480p for iteration and 720p for finishing. There
  is no 1080p/4K tier.
- **Silent — no audio generation.** Pair with external audio in post if
  needed.
- Single-frame conditioning only: `first_frame` is supported,
  `last_frame` is not. Use Veo 3.1 or Kling if you need both endpoints
  locked.
- No negative prompt — this model accepts no provider parameters at all.
- OpenRouter publishes nothing either way about audio or a seed here. Both
  controls are offered and the model's own behaviour decides, so try one
  short clip before planning around either.

---

### SpaceXAI: Grok Imagine Video 1.5

> **id**: `x-ai/grok-imagine-video-1.5`

The successor tier, built for the same fast iteration and adding a 1080p
finishing resolution the original stops short of. Duration is any whole
number of seconds from 1 to 15, so you can draft an idea as a two-second
clip and only commit to a full-length take once the prompt is right.
Seven framings — more
than most models — including the 3:2 and 2:3 photographic shapes its
neighbours skip. Works from a text prompt alone or from a supplied
opening still, and accepts no provider parameters at all, which makes it
one of the simplest models here to drive.

**Tips & pitfalls**

- Draft at 480p and one or two seconds; the composition reads clearly
  enough at the low tier to judge before committing to 1080p.
- 3:2 and 2:3 are worth remembering when a clip has to sit alongside
  photography.
- No provider parameters exist to fall back on — everything has to come
  from the prompt and the controls.
- Nothing is published either way about audio or a seed. Both controls
  are offered and the model's default applies.

---

### Black Forest Labs: FLUX.3 Video

> **id**: `black-forest-labs/flux-3-video`

Text- and image-to-video built around controlled, keyframe-driven shots.
You can hand it an opening still, a closing still, or both, and it fills
in the motion between them — and it will continue an existing clip rather
than only starting a new one, so a long sequence can be built a segment at
a time instead of being asked for in one go. Clips run 5 to 20 seconds at
720p or 1080p, across six framings from ultrawide 21:9 through to vertical
9:16, with audio generated alongside the picture.

**Tips & pitfalls**

- Anchor both ends when you know where the shot should finish; a closing
  frame is what separates this from a model you can only point at a
  starting still.
- Build long sequences as a chain of continuations rather than one very
  long request — each segment stays sharper and you can redirect between
  them.
- No deterministic seed, so an idea you like cannot be re-rolled exactly.
  Save the clip you want before iterating on the prompt.
- `safety_tolerance` and `version` are the two provider parameters, both
  free text.

---

### ByteDance: Seedance 2.5

> **id**: `bytedance/seedance-2.5`

The long-form member of the Seedance family: 30 seconds in a single clip,
twice what most video models will give you, aimed at storytelling that has
to hold together across that span. Reference-driven generation, editing an
existing clip, and extending one that already exists. Takes an opening
still, a closing still, or both, offers six framings including ultrawide
21:9, generates audio, and honours a seed. Output is 480p or 720p, with
twelve exact canvas sizes if you need to pin dimensions.

**Tips & pitfalls**

- The 30-second ceiling is the reason to pick this model; if your shot is
  under 10 seconds another model will usually cover it.
- Lock a seed before refining — over a half-minute clip an unseeded re-roll
  changes far more than the line you edited.
- Long clips reward one continuous action described plainly over a list of
  cuts. Ask for a scene, not a sequence of shots.
- `watermark`, `req_key` and `output_format` are the provider parameters.

---

### MiniMax: H3

> **id**: `minimax/hailuo-3`

A lightweight open-weights model aimed at precise, instruction-guided work
rather than free-running scenes. The one to reach for when the clip has to
carry legible text or a brand mark correctly, or when you want an edit
applied to supplied footage instead of a scene invented from scratch.
Everything it makes is 2K — there is no lower tier to trade down to —
across six framings from ultrawide 21:9 to vertical 9:16, in clips of 5 to
15 seconds, with audio.

**Tips & pitfalls**

- Write the instruction, not the scene. This model rewards precise wording
  over atmosphere.
- Put any text you need rendered in quotes exactly as it should appear,
  capitalisation included.
- Every clip is 2K, so there is no lower resolution to draft at. Keep
  drafts short instead and lengthen once the prompt is right.
- Trim the reference set to the images that are doing work; each extra one
  is another thing the model has to reconcile.
- No deterministic seed, so an exact re-run is not available.
- `aigc_watermark` is the single provider parameter.

---

### Runway: Aleph 2.0

> **id**: `runway/aleph-2`

An in-context **video editor**, not a text-to-video model. You attach
footage and give an instruction — change the weather, replace what is on
the wall, take the parked cars out of the street — and it applies that
across the clip while leaving everything you did not ask about alone.
Keyframes let you show it what a moment should look like instead of
describing it. Because the work is done on your footage, the length and
the dimensions of the result come from the clip you supply rather than
from a control here; the eight framings it publishes run from 21:9 down to
9:16. It honours a seed and adds no audio.

**Tips & pitfalls**

- Attach the clip you want edited. There is nothing to work from if no
  footage arrives with the instruction.
- Name the change and nothing else. Re-describing the whole scene invites
  the model to redo parts you wanted kept.
- One instruction per pass holds up far better than a list; run a second
  pass for the second change and you keep the ability to reject either.
- There is no duration, resolution or size control — trim the footage to
  what you want before sending it.
- Batch small corrections into a single pass where you can, rather than
  sending a string of one-second fixes. What a model charges is on
  OpenRouter's pricing page.
- `contentModeration` and `keyframes` are the provider parameters.

---

### Runway: Gen-4.5

> **id**: `runway/gen-4.5`

The Runway model that creates the shot in the first place, where Aleph 2.0
edits one you already have. Text- and image-to-video tuned for cinematic
work: strong motion, high visual fidelity, close adherence to what the
prompt asked for. Deliberately narrow — 720p, landscape 16:9 or portrait
9:16 only, clips of 2 to 10 seconds, animated from a single opening still
when you supply one — and that narrowness is the point. It honours a seed
and does not generate audio.

**Tips & pitfalls**

- Write it like a shot list: subject, action, camera move, lens feel,
  lighting. Prompt adherence is this model's strength.
- Two seconds is a real option; stringing several short beats together
  often beats asking for one ten-second take.
- Only a first frame is accepted. Describe where the shot should end up
  rather than expecting to pin it.
- Landscape and portrait are the only framings — no square, no ultrawide.
- `contentModeration` is the single provider parameter.

---

### Alibaba: HappyHorse 1.1

> **id**: `alibaba/happyhorse-1.1`

Text- and image-to-video in 3-to-15-second clips at 720p or 1080p, with an
unusually wide aspect-ratio range: the usual 16:9 / 9:16 / 1:1 / 4:3 / 3:4
plus ultrawide 21:9 and tall 9:21. That makes it a fit for cinematic
letterbox shots and full-bleed vertical formats the 8-second-capped models
cannot cover in one clip. First-frame conditioning and a seed for
reproducible runs. 1.1 refines 1.0 with the same controls.

**Tips & pitfalls**

- Front-load one clear shot in plain prose; one idea per clip holds
  together far better than crowded multi-subject scenes.
- Use a first-frame image to lock the opening composition and identity,
  then describe only the motion that follows.
- Longer durations (10–15s) tax motion and identity consistency harder —
  reuse a seed when iterating so the clip does not drift between runs.
- Nothing is published either way about audio. The control is offered and
  the model's default applies, so test one short clip before planning a
  soundtrack around it.
- No provider parameters at all.

---

### Alibaba: HappyHorse 1.0

> **id**: `alibaba/happyhorse-1.0`

The first HappyHorse tier: the same wide aspect-ratio range including
ultrawide 21:9 and tall 9:21, 3-to-15-second clips at 720p or 1080p,
first-frame conditioning and seed control. Largely superseded by 1.1,
which refines it with the same controls — reach for 1.0 only when you
need to pin the exact 1.0 generation behaviour.

**Tips & pitfalls**

- Prefer 1.1 for new work; it matches 1.0's controls and resolutions.
- Front-load a single clear shot and keep to one idea per clip;
  multi-subject action remains a weak spot.
- Anchor the opening with a first-frame image and reuse a seed across
  iterations to keep identity stable.
- Nothing is published either way about audio — test rather than assume.
- No provider parameters at all.

---

## Per-model parameter reference

This section enumerates exactly which filter knobs each model exposes,
based on the OpenRouter catalog at the time of writing.
The chat-filter UI auto-hides knobs the model does not support, so this table
is also the spec for the **model-specific** knobs you can change per-message.

Four further controls are on every video filter and are in none of these
tables, because they are pipe behaviour rather than anything a model
publishes:

- `Reuse previous videos`
- `Clarifying question limit`
- `Which frame to use from previous video`
- `Show what was reused`

`Reuse previous videos` is on by default, and is what makes a follow-up like
"make it black" edit the clip you just got instead of starting an unrelated
new one. Their defaults and ranges are in
[Video Intent Classifier](openrouter_video_intent_classifier.md); when an
admin turns `VIDEO_INTENT_ENABLED` off, all four disappear from the filter.

### Google: Veo 3.1 / Veo 3.1 Fast / Veo 3.1 Lite

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 4, 6, 8 | Three fixed lengths; no values in between. |
| Aspect ratio | Literal | 16:9, 9:16 | Native composition (no crop). |
| Resolution | Literal | 720p, 1080p, 4K (full + Fast); 720p, 1080p (Lite) | 4K is on the full and Fast tiers only; Lite caps at 1080p. |
| Size | Literal | from `supported_sizes` in catalog | Exact pixel dimensions; used when you need a specific canvas. |
| Frames | Literal | auto / none / first_only / first_last | first/last requires both images attached. |
| Negative prompt | str | free text | Routed via `negativePrompt` passthrough. |
| Audio (`generate_audio`) | Literal | model_default / on / off | Off loses the signature joint-diffusion soundtrack. |
| Seed | int | 0 = model default; otherwise 32-bit integer | Same prompt + seed yields a near-identical clip. |
| Person generation | Literal | "" / allow_all / allow_adult / dont_allow / disallow | Safety gate. `dont_allow` is the Gemini API spelling, `disallow` the Vertex AI one. EU/UK/CH/MENA only allow `allow_adult`. |
| Conditioning scale | float | 0.0 = default; 0.0–1.0 | Bounds and behaviour are unsourced — Google documents no `conditioningScale` on any Veo surface. |
| Enhance prompt | Literal | model_default / on / off | Auto-rewrite prompt (officially Veo 2 only on Vertex; provider may ignore). |
| Provider options JSON | str | raw JSON object keyed by provider slug | Escape hatch — see [Provider passthrough](#provider-passthrough). |

### Kling: Video O1

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 5, 10 | Two fixed lengths. |
| Aspect ratio | Literal | 16:9, 9:16, 1:1 | Square output uses 720×720. |
| Resolution | Literal | 720p only | Single tier. |
| Size | Literal | 1280×720, 720×1280, 720×720 | |
| Frames | Literal | auto / none / first_only / first_last | first/last for controlled transitions. |
| Negative prompt | str | free text | Hard guardrails — Kling honours these strongly. |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native ambient audio. |
| Provider options JSON | str | raw JSON | |

**No seed knob** — Kling's catalog says `seed: false`. Lock visual identity via reference frames + prompt language, not bit-exact replay.

### Kling: Video v3.0 Pro / v3.0 Standard

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 3–15 (any integer) | Any whole number of seconds in range. |
| Aspect ratio | Literal | 16:9, 9:16, 1:1 | |
| Resolution | Literal | 720p | The only tier published. |
| Size | Literal | 1280×720, 720×1280, 720×720 | |
| Frames | Literal | auto / none / first_only / first_last | Both endpoints can be locked. |
| Negative prompt | str | free text | Kling honours these strongly. |
| CFG scale | float | 0 = provider default | New in v3.0; higher follows the prompt more strictly. |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native audio generated in the same pass as the picture. |
| Provider options JSON | str | raw JSON | |

**No seed knob** (`seed: false`). Both tiers publish an identical knob set; the difference is output quality.

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
| Fast pretreatment | Literal | model_default / on / off | Quicker optimiser pass; small loss of quality. |
| Seed | int | ≥ 0 | The catalog says nothing either way (`seed: null`), so the control is offered; left at `0` nothing is sent and MiniMax's own behaviour applies. |

**No audio knob** (`generate_audio: false`). **No negative prompt.**

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
| Last image URL | str | URL | Anchor the closing frame. |
| Reference images JSON | str | JSON array | 9-image structured grid for identity, wardrobe and props. |
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
| Duration | Literal | 4–15 (Fast/2.0); 4–12 (1.5 Pro) | Any whole number of seconds in range. |
| Aspect ratio | Literal | 1:1, 3:4, 9:16, 4:3, 16:9, 21:9, 9:21 (+ 9:21 on 1.5 Pro) | Widest aspect coverage. |
| Resolution | Literal | 480p, 720p (Fast); 480p, 720p, 1080p, 4K (2.0); 480p, 720p, 1080p (1.5 Pro) | Only 2.0 reaches 4K. |
| Size | Literal | 13 (2.0 Fast); 25 (2.0); 21 (1.5 Pro); 12 (2.5) | 2.0 publishes the most exact pixel sizes of any catalog model. |
| Frames | Literal | auto / none / first_only / first_last | |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native audio in same pass as video. |
| Seed | int | 0 / 32-bit int | |
| Provider options JSON | str | raw JSON | |
| Watermark | Literal | model_default / on / off | Visible ByteDance watermark on output MP4. |
| Request key | str | provider routing string | Volcengine ModelArk SKU/endpoint identifier. |

**No negative prompt** on any Seedance variant — not in `allowed_passthrough_parameters`.

### OpenAI: Sora 2 Pro

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 4, 8, 12, 16, 20 | 20s clips; only Seedance 2.5 goes longer. |
| Aspect ratio | Literal | 16:9, 9:16 | No square / cinematic widescreen. |
| Resolution | Literal | 720p, 1080p | 1080p is the finishing tier. |
| Size | Literal | 4 dimensions | |
| Audio (`generate_audio`) | Literal | model_default / on / off | Native dialogue, SFX, ambience — Sora's signature. |
| Provider options JSON | str | raw JSON | |
| Quality | Literal | "" / standard / hd | Hint — resolution drives the actual tier. |
| Style | str | free text | Stylistic hint (e.g. `cinematic`, `documentary handheld`). |

**No frames knob** (catalog says `supported_frame_images: null` — text-to-video only). **No seed knob** (`seed: false`). **No negative prompt.**

### SpaceXAI: Grok Imagine Video

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 1–15 (any integer) | 1-second granularity, shared with the 1.5 tier. |
| Aspect ratio | Literal | 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3 | Seven framings; Aleph 2 publishes these plus 21:9. |
| Resolution | Literal | 480p, 720p | 480p for drafts, 720p for finishing. |
| Size | Literal | 14 dimensions | e.g. 854×480, 1280×720, 720×1280, 480×480. |
| Frames | Literal | first only | Image-to-video via first frame; no last frame. |
| Audio | Literal | model default / on / off | Offered because nothing is published either way; the model's own behaviour decides. |
| Seed | int | 0 = model default | Offered because nothing is published either way; a repeat is likely rather than guaranteed. |
| Provider options JSON | str | raw JSON | |

**No negative prompt** (`allowed_passthrough_parameters` is empty).

---

### SpaceXAI: Grok Imagine Video 1.5

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 1–15 (any integer) | 1-second granularity. |
| Aspect ratio | Literal | 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3 | Same seven framings as the original tier. |
| Resolution | Literal | 480p, 720p, 1080p | Adds the 1080p finishing tier. |
| Frames | Literal | first only | Image-to-video via first frame; no last frame. |
| Audio | Literal | model default / on / off | Offered because nothing is published either way. |
| Seed | int | 0 = model default | Offered because nothing is published either way. |
| Provider options JSON | str | raw JSON | |

**No size knob** (no fixed dimensions published). **No negative prompt** (`allowed_passthrough_parameters` is empty).

---

### Black Forest Labs: FLUX.3 Video

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 5–20 (any integer) | Any whole number of seconds in range. |
| Aspect ratio | Literal | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 | |
| Resolution | Literal | 720p, 1080p | 1080p is the finishing tier. |
| Frames | Literal | first only, first + last | Both endpoints can be locked. |
| Audio | Literal | model default / on / off | |
| `safety_tolerance` | str | free text | Provider parameter; no values published. |
| `version` | str | free text | Provider parameter; no values published. |
| Provider options JSON | str | raw JSON | |

**No seed knob** (`seed: false`). **No size knob** (no fixed dimensions published).

---

### ByteDance: Seedance 2.5

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 4–30 (any integer) | Longest single take in the catalog. |
| Aspect ratio | Literal | 16:9, 4:3, 1:1, 3:4, 9:16, 21:9 | |
| Resolution | Literal | 480p, 720p | |
| Size | Literal | 12 dimensions | e.g. 1280×720, 960×960, 720×1280, 1470×630. |
| Frames | Literal | first only, first + last | |
| Audio | Literal | model default / on / off | Generated alongside the picture in a single pass. |
| Seed | int | 0 = model default | |
| Watermark | Literal | model default / on / off | Provider branding overlay. |
| `req_key` | str | free text | Provider-side request identifier. |
| `output_format` | str | free text | Provider parameter; no values published. |
| Provider options JSON | str | raw JSON | |

---

### MiniMax: H3

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 5–15 (any integer) | Any whole number of seconds in range. |
| Aspect ratio | Literal | 21:9, 16:9, 4:3, 1:1, 3:4, 9:16 | |
| Resolution | Literal | 2K | The only tier published; there is no lower one to draft at. |
| Frames | Literal | first only, first + last | |
| Audio | Literal | model default / on / off | |
| `aigc_watermark` | str | free text | Provider parameter; no values published. |
| Provider options JSON | str | raw JSON | |

**No seed knob** (`seed: false`). **No size knob** (no fixed dimensions published).

---

### Runway: Gen-4.5

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 2–10 (any integer) | Any whole number of seconds in range. |
| Aspect ratio | Literal | 16:9, 9:16 | Landscape or portrait only. |
| Resolution | Literal | 720p | The only tier published. |
| Size | Literal | 1280×720, 720×1280 | |
| Frames | Literal | first only | No last frame. |
| Seed | int | 0 = model default | |
| `contentModeration` | str | free text | Provider parameter; no values published. |
| Provider options JSON | str | raw JSON | |

**No audio knob** (`generate_audio: false` — silent output).

---

### Runway: Aleph 2.0

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Aspect ratio | Literal | 16:9, 4:3, 3:2, 1:1, 2:3, 3:4, 9:16, 21:9 | |
| Seed | int | 0 = model default | |
| `contentModeration` | str | free text | Provider parameter; no values published. |
| `keyframes` | str | free text | Provider parameter; show a moment rather than describing it. |
| Provider options JSON | str | raw JSON | |

**No duration, resolution, size or frames knob.** This is an in-context editor: the clip you attach sets the length and the dimensions of the result, so none of those are published and none are drawn. **No audio knob** (`generate_audio: false`).

---

### Alibaba: HappyHorse 1.1 / HappyHorse 1.0

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Duration | Literal | 3–15 (any integer) | Any whole number of seconds in range. |
| Aspect ratio | Literal | 16:9, 9:16, 1:1, 4:3, 3:4, 21:9, 9:21 | The only models offering tall 9:21. |
| Resolution | Literal | 720p, 1080p | 1080p is the finishing tier. |
| Size | Literal | 14 dimensions | e.g. 1920×1080, 1080×1920, 2520×1080, 1080×2520. |
| Frames | Literal | first only | No last frame. |
| Audio | Literal | model default / on / off | Offered because nothing is published either way. |
| Seed | int | 0 = model default | |
| Provider options JSON | str | raw JSON | |

**No negative prompt** (`allowed_passthrough_parameters` is empty on both tiers).

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

A gate written as "publishes X and it is not `false`" is deliberate: a
model that publishes nothing at all about a capability is a different
case from one that publishes it does not have it. The first gets the
control, and leaving the control alone sends nothing so the model's own
default applies; the second gets no control.

### Core UserValves (pre-existing — every variant reuses these)

| Identifier | Type | Default | Maps to API field | Gate (catalog condition) | Exposed on |
|------------|------|---------|-------------------|---------------------------|------------|
| `VIDEO_PROVIDER_OPTIONS_JSON` | `str` | `""` | `provider.options` (raw JSON object keyed by slug) | always | all 22 |
| `VIDEO_DURATION` | `Literal[0, …]` | `0` | top-level `duration` | `supported_durations` non-empty | 21 (all except Aleph 2.0) |
| `VIDEO_ASPECT_RATIO` | `Literal["", …]` | `""` | top-level `aspect_ratio` | `supported_aspect_ratios` non-empty | all 22 |
| `VIDEO_RESOLUTION` | `Literal["", …]` | `""` | top-level `resolution` | `supported_resolutions` non-empty | 21 (all except Aleph 2.0) |
| `VIDEO_SIZE` | `Literal["", …]` | `""` | top-level `size` | `supported_sizes` non-empty | 18 (all except FLUX.3 Video, H3, Aleph 2.0, Grok Imagine Video 1.5) |
| `VIDEO_FRAME_MODE` | `Literal["auto", "none", "first_only"(, "first_last")]` | `"auto"` | controls `frame_images[]` shaping | `supported_frame_images` non-empty | 20 (all except Sora 2 Pro and Aleph 2.0) |
| `VIDEO_NEGATIVE_PROMPT` | `str` | `""` | passthrough `negative_prompt` (or `negativePrompt` on Veo) | `"negative_prompt"` or `"negativePrompt"` in `allowed_passthrough_parameters` | 8 (Veo trio, Kling trio, Wan 2.6, Wan 2.7) |
| `VIDEO_GENERATE_AUDIO` | `Literal["model_default", "on", "off"]` | `"model_default"` | top-level `generate_audio` (boolean) | `generate_audio` present and not published as `false` | 19 (all except Hailuo 2.3, Gen-4.5, Aleph 2.0) |
| `VIDEO_SEED` | `int` (`ge=0`) | `0` | top-level `seed` | `seed` present and not published as `false` | 16 (all except FLUX.3 Video, H3, the Kling trio, Sora 2 Pro) |
| `VIDEO_AUDIO_URL` | `str` | `""` | passthrough `audio` (URL) | `"audio"` allowed **and** `audio` in the model's declared input modalities | none — Wan 2.6 and 2.7 publish the parameter but declare only text and pictures |
| `VIDEO_REFERENCE_VIDEO_URL` | `str` | `""` | passthrough `video` | `"video"` allowed **and** `video` in the model's declared input modalities | none — Wan 2.7 publishes the parameter but declares only text and pictures |
| `VIDEO_REFERENCE_VIDEOS_JSON` | `str` (JSON array) | `""` | passthrough `videos` | `"videos"` allowed **and** `video` in the model's declared input modalities | none — Wan 2.7 publishes the parameter but declares only text and pictures |
| `VIDEO_REFERENCE_IMAGES_JSON` | `str` (JSON array) | `""` | passthrough `images` | `"images"` in `allowed_passthrough_parameters` | Wan 2.7 |
| `VIDEO_LAST_IMAGE_URL` | `str` | `""` | passthrough `last_image` | `"last_image"` in `allowed_passthrough_parameters` | Wan 2.7 |

### New typed UserValves added in this feature (passthrough-param wrappers)

These were added in the verification + upgrade pass so users no longer
need to hand-write JSON for every model-specific knob. Each renders only
when the corresponding string appears in the model's
`allowed_passthrough_parameters`.

| Identifier | Type | Default | Maps to API field | Gate | Exposed on |
|------------|------|---------|-------------------|------|------------|
| `VIDEO_PERSON_GENERATION` | `Literal["", "allow_all", "allow_adult", "dont_allow", "disallow"]` | `""` | passthrough `personGeneration` | `"personGeneration"` allowed | Veo trio |
| `VIDEO_CONDITIONING_SCALE` | `float` (`ge=0.0`, `le=1.0`) | `0.0` | passthrough `conditioningScale` | `"conditioningScale"` allowed | Veo trio |
| `VIDEO_CFG_SCALE` | `float` (`ge=0.0`, `le=1.0`) | `0.0` | passthrough `cfg_scale` | `"cfg_scale"` allowed | Kling v3.0 (Pro, Standard) |
| `VIDEO_ENHANCE_PROMPT` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `enhancePrompt` (boolean) | `"enhancePrompt"` allowed | Veo trio |
| `VIDEO_PROMPT_OPTIMIZER` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `prompt_optimizer` (boolean) | `"prompt_optimizer"` allowed | Hailuo |
| `VIDEO_FAST_PRETREATMENT` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `fast_pretreatment` (boolean) | `"fast_pretreatment"` allowed | Hailuo |
| `VIDEO_PROMPT_EXTEND` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `prompt_extend` (boolean) | `"prompt_extend"` allowed | Wan 2.7 |
| `VIDEO_RATIO` | `str` | `""` | passthrough `ratio` | `"ratio"` allowed | Wan 2.7 |
| `VIDEO_ENABLE_PROMPT_EXPANSION` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `enable_prompt_expansion` (boolean) | `"enable_prompt_expansion"` allowed | Wan 2.6 |
| `VIDEO_SHOT_TYPE` | `str` | `""` | passthrough `shot_type` | `"shot_type"` allowed | Wan 2.6 |
| `VIDEO_WATERMARK` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `watermark` (boolean) | `"watermark"` allowed | 4 (Seedance 1.5 Pro, 2.0, 2.0 Fast, 2.5) |
| `VIDEO_REQ_KEY` | `str` | `""` | passthrough `req_key` | `"req_key"` allowed | 4 (Seedance 1.5 Pro, 2.0, 2.0 Fast, 2.5) |
| `VIDEO_QUALITY` | `str` | `""` | passthrough `quality` | `"quality"` allowed | Sora 2 Pro |
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
  `resolution`, `size`, `seed`, `generate_audio`, `frame_images`,
  `input_references`): set directly in the `/videos` POST body.
  `negative_prompt` is not one of them — it travels as a provider
  setting, as the table above shows.
- **Provider passthrough** fields (everything else —
  `personGeneration`, `watermark`, etc.): go under
  `provider.options.<slug>`, keyed by the provider slug the catalog
  publishes for the model. Written at the request root they are accepted
  and ignored, which is indistinguishable from working.
- `VIDEO_PROVIDER_OPTIONS_JSON` writes to the same place. It is the
  escape hatch for fields the pipe has no typed valve for, not a
  different destination.

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
above, followed by the four reuse-of-previous-video controls documented in
[Video Intent Classifier](openrouter_video_intent_classifier.md). Behaviour
rules:

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
  - `none`: no attached image anchors the clip. They are still sent, as
    references the model may draw on — see
    [Attachments that are not frames](#attachments-that-are-not-frames).
  - `first_only`: even if multiple images are attached, only the first is
    used as `first_frame`.
  - `first_last`: explicitly attach two images as start and end keyframes.
- Setting `Size` to exact pixel dimensions settles any argument with the
  other two shape knobs: a `Resolution` tier that disagrees with those
  pixels, or an `Aspect ratio` that is not the shape of those pixels, is
  left out and named in a warning notice in the chat. That is this pipe's
  rule rather than OpenRouter's — their video schema says only that
  `size` is interchangeable with `resolution` + `aspect_ratio`, and
  publishes nothing about what happens when the two disagree, so the pipe
  sends the one the user was most specific about instead of a request
  that contradicts itself. (The image API does publish a rejection rule;
  video does not.) Where the `Size` value is itself a tier, only one of
  the two is sent, and `Size` is the one that is kept.
- A free-text knob's value is sent as the text you typed. A value that
  starts with `[` or `{` is read as JSON and must be valid JSON, and the
  error names the knob. A bare number is sent as a number, since several
  provider options take one; `NaN`, `Infinity` and numbers too large to
  write down are refused rather than sent.

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
- **Controls**: every control this model's filter draws — the model's own
  settings, each with a one-sentence description tailored to it, then the
  four reuse-of-previous-video controls, which read the same on every
  model because they are pipe behaviour.
- **Tips & pitfalls**: 3–4 practical bullets — what works, what fails,
  prompt patterns.

The blurb quotes no rates. What a model charges is on OpenRouter's pricing
page, which is the only copy of it that cannot go stale. What a particular
generation was billed is reported on the status line when it finishes, as
long as usage details are on: that is your own Show usage details setting
once you have set it, and the site default your administrator chooses
until then.

The written-up part of each blurb ships with the pipe; everything about
capabilities is read from the catalog at the moment you ask. So a model
that gains a resolution shows it on the next `help` — nothing has to be
updated or redeployed for that.

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

These three are strict for **frames**, because the clip was meant to be
anchored on them: one that breaks a limit fails the whole request. The
same three limits are applied again to anything sent only as a reference
(below), and there a file that breaks one is left out with a warning
notice in the chat naming it and the reason, while the video still
renders. References count against their own combined budget, separate
from the frames'.

Per-model frame support:

| Model | first_frame | last_frame |
|-------|:-----------:|:----------:|
| Veo 3.1 / Fast / Lite | ✅ | ✅ |
| Kling Video O1 | ✅ | ✅ |
| Kling v3.0 Pro / Standard | ✅ | ✅ |
| Hailuo 2.3 | ✅ | ❌ |
| Wan 2.7 | ✅ | ✅ |
| Wan 2.6 | ✅ | ❌ |
| Seedance 2.0 / 2.0 Fast / 1.5 Pro | ✅ | ✅ |
| Sora 2 Pro | ❌ | ❌ |

If you attach an image to a Sora chat, it's not used as a frame — Sora's
catalog has no `supported_frame_images`. It is sent as a reference
instead, as described next.

---

## Attachments that are not frames

Anything you attach that the Frames control does not claim — extra
images, a second image on a first-frame-only model, a clip, a sound file,
or any image at all when Frames is set to `none` or the model has no
frame support — is sent to the model as a **reference**: material for it
to draw on rather than a fixed start or end point. Nothing you attach is
silently discarded any more.

A left-over image goes as an image reference. Clips and sound files go
as video and audio references, on the models that declare they read
them: OpenRouter publishes the kinds each model takes, and one that does
not name a kind is not sent that kind — the file is left out with a
notice saying the model does not take it, rather than sent somewhere it
would be discarded without a word. Each reference is checked against the
frame limits above; a clip or sound file, which travels by way of a public
file host, is bounded by `MEDIA_FILE_HOST_MAX_SIZE_MB` instead.
`REMOTE_VIDEO_MAX_SIZE_MB` bounds only the finished video coming back, not
anything you attach. A reference that fails on kind, format, pixel size,
count or the combined budget is left out with a warning notice naming it
and why, and the render still goes ahead; one that is simply too large
stops the request instead, so that nothing is generated and billed from a
prompt the attachment was meant to anchor.

Because a reference is enough to generate from, a turn with attachments
and **no typed words** is now submitted rather than refused.

Whether an attachment also stays visible in the chat as a normal
attachment depends on Open WebUI's **File context** capability for that
model. While it is on — Open WebUI's own default — Open WebUI answers an
attachment-bearing turn with an extra retrieval round-trip whose text
would be pasted into the video prompt, so the pipe hands the files to the
model and takes them out of the request. Where an admin has
`UPDATE_MODEL_CAPABILITIES` on, the pipe unticks File context on video
and image models, and from then on the attachments stay in the chat as
well as being sent. Files that are not images, clips or sound — a PDF,
say — are left alone either way.

---

## Multimodal references (Wan 2.7)

Wan 2.7 is the only catalog model that exposes more than one kind of
reference. Two of its controls are drawn:

- **Reference images JSON** (`images` passthrough): JSON array — Wan
  2.7's 9-image structured grid for identity/wardrobe/props/environment
  anchoring without describing them in text.
- **Last image URL** (`last_image` passthrough): closing-frame anchor;
  used together with `first_frame` for controlled in-between motion.

Wan 2.7 also publishes `video`, `videos` and `audio` in its
`allowed_passthrough_parameters`, and Alibaba documents clip and voice
conditioning for the model. **Neither is offered here.** OpenRouter
declares Wan 2.6 and Wan 2.7 as accepting only text and image input, and
a reference of a kind a model does not declare is refused at submission —
so drawing those controls would have offered a setting whose only outcome
is a failed job. The gate that withholds them reads the model's own
declared input modalities, so if OpenRouter later declares video or audio
input for these models, the controls appear again with no code change.

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
    "experimentalFlag": "value"
  }
}
```

The pipe deep-merges this into `provider.options` after typed valves are
written, and forwards whatever nesting was written. It does not move a value
between a `parameters` wrapper and the slug in either direction, because both
placements are in use and neither is right everywhere: OpenRouter's own
provider-specific video options cookbook posts
`options.<slug>.parameters` for `google-vertex`, while a recorded probe against
the `seed` provider showed a knob under `parameters` accepted with a job id and
never applied, and the same knob written directly under the slug taking effect.
Which placement a provider reads is the operator's call, so the pipe does not
choose one for them.

OpenRouter's video request schema defines exactly one provider property,
`options`. Chat-routing and privacy fields — `only`, `order`, `sort`,
`max_price`, `zdr`, `data_collection` and the rest — are not part of it, and
the schema does not reject unknown keys, so sending them would be accepted
and ignored. The pipe withholds them and logs which ones it withheld, so a
preference that cannot be honoured on this transport is visible rather than
silently absent.

The same reasoning decides which provider carries an attachment. Because the
video request carries no `only`, routing never sees an operator's pin, so the
pipe keys attachments to a provider drawn from the catalog rather than to the
pin. On the image endpoint `only` *is* accepted, so there the pin decides.

---

## Pricing and cost display

No rate is quoted anywhere in this pipe. Nothing in the `help` reply, in a
model's filter, or on this page states what a model charges. OpenRouter's
pricing page is the one place to read them, and it is the only copy that
cannot go stale: a figure written down anywhere else stops being true on
the day OpenRouter changes it, and nothing reports that day.

What is reported is what a generation actually came to. Where OpenRouter's
poll response carries a charge above zero, the **final status line** shows
it, as long as usage details are on: that is your own Show usage details
setting once you have set it, and the site default your administrator
chooses until then. The figure is what that specific generation was
billed, not a rate.

---

## Output rendering and message format

When generation succeeds, the assistant message contains:

```markdown
[openrouter:v1:videojob:<job_id>]: #
[openrouter:v1:videomodel:<model_id>]: #

<video>
/api/v1/files/<owui_file_id>/content
</video>
```

A model that returns more than one clip for a single job gets one
`<video>` block per clip, in the order OpenRouter returned them, each
stored as its own file. If some clips download and others do not, the
ones that arrived are still delivered rather than the whole job being
thrown away.

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

The message always ends with a newline — defensive against any later
concatenation that could smash markers from a follow-up message into
inline text.

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

- **Chats with no stored row** (chat IDs starting with `temporary:`, `local:` or `channel:`): Open WebUI does
  not persist these to chat storage, so markers can't be written. The
  on-submit `'message'` emit is skipped for all three. `local:` is Open WebUI's legacy
  spelling of `temporary:`; `channel:` is an ordinary channel invocation, not an edge case.
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
| `REMOTE_VIDEO_MAX_SIZE_MB` | `500` | 1–2048 | Max downloaded video size; oversized aborts streaming. Bounds the generated video only, never an attachment. |
| `VIDEO_DOWNLOAD_CHUNK_SIZE` | `1048576` | 65536–8388608 | Chunk size in bytes for streaming download. |
| `MAX_CONCURRENT_VIDEO_GENS` | `2` | 1–100 | Global concurrency cap per pipe process. |
| `MAX_CONCURRENT_VIDEO_GENS_PER_USER` | `2` | 1–25 | Per-user concurrency cap. |
| `VIDEO_FRAME_IMAGE_MAX_BYTES` | `12_582_912` (12 MB) | 65536–67108864 | Per-image decoded size cap. |
| `VIDEO_FRAME_TOTAL_MAX_BYTES` | `52_428_800` (50 MB) | 65536–134217728 | Combined frame-bytes cap across one request. |
| `VIDEO_FRAME_IMAGE_MIME_ALLOWLIST` | `image/jpeg,image/png,image/webp` | comma-list | Allowed MIMEs for frame images. |
| `VIDEO_OUTPUT_MIME_ALLOWLIST` | `video/mp4,video/webm` | comma-list | Allowed MIMEs for downloaded video (sniffed from prefix). |
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
  reject an oversized generated video before it hits your file backend.
  To bound what users send *out*, lower `MEDIA_FILE_HOST_MAX_SIZE_MB`.

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
  └─ orchestrator dispatches to VideoGenerationAdapter.generate() if
     model.features has "video_generation"
        ├─ help short-circuit (prompt == "help" → render_video_help)
        ├─ resume check (read message → scan for [videojob:...] marker)
        ├─ intent classification, when it is switched on and the turn
        │  qualifies — may answer with a clarification instead of a job,
        │  and degrades open if it fails
        ├─ acquire user slot
        ├─ prepare attachments
        │     ├─ frame images inlined as data URLs
        │     └─ clip and sound references need a public https link, so
        │        with the file-host valves on the pipe uploads them and
        │        sends the link (media_relay); without those valves such
        │        a reference is left out with a notice. A reference is
        │        only uploaded when the requester owns the file and the
        │        stored record names its media type, at most sixteen per
        │        request and at most MEDIA_FILE_HOST_MAX_SIZE_MB of them
        │        put together, inside one wall-clock budget for the whole
        │        request; the chat is told which host and for how long
        │        BEFORE the first byte is sent, and nothing is uploaded
        │        when that could not be delivered. A file that did go out
        │        is written into the message Open WebUI stores, so the
        │        record survives a reload and a resumed job
        ├─ build the request body (top-level fields the model publishes,
        │  the rest under provider.options.<slug>), resolving and checking
        │  every address it carries — done before the global slot is taken,
        │  since the hosts come from the request and decide how long the
        │  lookups block
        ├─ acquire global semaphore
        ├─ submit job via OpenRouterVideoClient.submit(); read the job id
        │  out of the accepted payload
        ├─ emit pending content via OWUI socket 'message' event
        │      (routed to Chats.upsert_message_to_chat_by_id_and_message_id —
        │       persists the [videojob:<id>] marker BEFORE the bg task starts)
        ├─ spawn _run_lifecycle_after_submit() as bg asyncio.Task
        │     ├─ poll with backoff until terminal status
        │     ├─ download each generated clip (streaming, bounded, capped
        │     │  number of outputs)
        │     ├─ MIME-sniff against VIDEO_OUTPUT_MIME_ALLOWLIST
        │     ├─ stream-upload to OWUI storage (per-backend: Local/S3/GCS/Azure)
        │     ├─ insert Files row + link to chat
        │     ├─ build success content (markers + <video> blocks)
        │     └─ return VideoLifecycleResult — emits status lines, but never
        │        the message content
        ├─ outer awaits bg task with asyncio.shield (survives client disconnect)
        ├─ outer emits status line + chat:completion (the SOLE emit)
        └─ outer returns content string
              └─ functions.py wraps as SSE chunk, OWUI middleware accumulates,
                 stream finalizer upserts to message DB (one write).
```

Key invariant: **exactly one `_emit_completion` per `(chat_id,
message_id)`**. The bg task does the work and returns the result;
the outer (or waiter for de-duped re-entries) is the sole emitter. This
prevents the duplicate-content / leaked-marker bug class.

Second invariant, on the way in: **a clip or a sound file is only ever
sent as a link**. OpenRouter takes those references as https URLs, so
there is no inline-data path for them to fall back to — a file that
cannot be given a link is left out and the user is told why, rather than
being encoded into the request and silently dropped at the other end.
Pictures are the exception: a frame image is always inlined, and an image
reference is inlined too unless it has been sent to the file host as well.

Third invariant, on the same way in: **nothing is published without the
user being told first, and only the owner of a file may publish it**. The
upload is anonymous — that is what lets it work with no account — so it
carries no credential anybody here could later use to delete it, and the
notice says so rather than implying a takedown that nobody can perform.
Read access inside Open WebUI is granted by sharing a chat, a channel, a
knowledge base or a workspace model; publication is not, so a reference
the requester can open but does not own is withheld with a note.

Key files:

- [`integrations/video.py`](../open_webui_openrouter_pipe/integrations/video.py)
  — `VideoGenerationAdapter` (entry point, lifecycle, emit).
- [`integrations/video_client.py`](../open_webui_openrouter_pipe/integrations/video_client.py)
  — `OpenRouterVideoClient`, the HTTP client for `/videos/*` endpoints.
- [`integrations/video_catalog.py`](../open_webui_openrouter_pipe/integrations/video_catalog.py)
  — fetches `/videos/models` and registers them in
  `OpenRouterModelRegistry`.
- [`integrations/media_relay.py`](../open_webui_openrouter_pipe/integrations/media_relay.py)
  — puts an attached clip or sound file behind a public link so it can be
  sent as a reference: which hosts are known, which origins each may
  answer with, how long each keeps a file, the size ceiling, and the
  retries. A 200 carrying a link the chosen host does not serve is not an
  answer, and whatever the host did say is quoted inertly rather than
  rendered into the chat as markdown.
- [`integrations/video_intent.py`](../open_webui_openrouter_pipe/integrations/video_intent.py)
  and [`integrations/video_intent_prompts.py`](../open_webui_openrouter_pipe/integrations/video_intent_prompts.py)
  — work out what the turn is asking for before a job is submitted; see
  [the intent classifier document](openrouter_video_intent_classifier.md).
- [`integrations/video_help.py`](../open_webui_openrouter_pipe/integrations/video_help.py)
  — per-model help blurbs, with the capability lines and the control list
  read from the live catalog row.
- [`integrations/video_types.py`](../open_webui_openrouter_pipe/integrations/video_types.py)
  — `VideoLifecycleResult`, `DownloadedVideo` dataclasses.
- [`integrations/provider_options.py`](../open_webui_openrouter_pipe/integrations/provider_options.py)
  — the per-transport set of provider keys OpenRouter documents. The video
  schema defines `options` and nothing else, so a routing preference that
  belongs to chat completions is dropped here rather than sent where
  nothing would enforce it.
- [`filters/video_filter_renderer.py`](../open_webui_openrouter_pipe/filters/video_filter_renderer.py)
  — generates the per-model OWUI filter source code.
- [`filters/filter_manager.py`](../open_webui_openrouter_pipe/filters/filter_manager.py)
  — installs filter rows in OWUI Functions table.
- [`storage/video_persistence.py`](../open_webui_openrouter_pipe/storage/video_persistence.py)
  — thin resume-path helper that reads the persisted chat message to
  detect prior `videojob` markers.
- [`storage/multimodal.py`](../open_webui_openrouter_pipe/storage/multimodal.py)
  — `_download_remote_url_streaming`: the size- and type-bounded download
  that fetches the finished clip. Video generation is its only caller; the
  non-streaming sibling next to it is what the rest of the pipe uses.
- [`storage/owui_files.py`](../open_webui_openrouter_pipe/storage/owui_files.py)
  — `OwuiFileGateway.upload_to_owui_storage_from_path` and
  `try_link_file_to_chat`, which put the downloaded clip into OWUI storage
  and attach it to the chat.
- [`models/registry.py`](../open_webui_openrouter_pipe/models/registry.py)
  — `register_video_models()` merges video models into the chat catalog.
- [`models/catalog_manager.py`](../open_webui_openrouter_pipe/models/catalog_manager.py)
  — metadata sync that attaches and defaults filters.
- [`core/config.py`](../open_webui_openrouter_pipe/core/config.py)
  — Valve definitions.

---

For the rest of the pipe — repository layout, the normal chat request
lifecycle, background workers — see
[the developer guide](developer_guide_and_architecture.md).
