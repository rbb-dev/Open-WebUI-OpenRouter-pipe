# OpenRouter Image Generation

This pipe exposes OpenRouter's forty native image-output models as
selectable chat models in Open WebUI. You pick an image model in the chat
header (just like any other LLM), type a prompt, and the pipe routes the
request on what that model declares it produces: a model that returns only
images goes to OpenRouter's dedicated image endpoint, and one that returns
both text and images goes to chat completions with
`modalities: ["image", "text"]`. Either way the image comes back inline as
base64, is persisted to OWUI file storage, and is rendered with
`![alt](url)` markdown. There is no polling; both transports answer
synchronously.

OpenRouter is migrating image models onto a dedicated image endpoint one at
a time, and a migrated model refuses the chat transport outright. The pipe
therefore routes on the model's declared output modalities: a model that
emits only images goes to the dedicated endpoint, while a multimodal model
that also emits text stays on chat completions. No catalog field predicts
which models have migrated, so the routing follows the modality rather than
a per-model list.

The feature is on by default (`ENABLE_OPENROUTER_IMAGE_GENERATION=True`).
If you want to disable it, set that valve to `False` in Admin → Functions
→ OpenRouter pipe → Valves; previously-registered pure-image-only models
will be removed from the dropdown immediately.

> **Note:** This document covers the **native image-output models**
> integration (Sourceful Riverflow, Black Forest Labs FLUX, ByteDance
> Seedream, Google Gemini Image, OpenAI GPT-5 Image, etc.). It is
> distinct from the **legacy `openrouter_image_gen` filter** which wires
> the OpenAI Responses-API `image_generation_call` server tool — that
> remains controlled by `ENABLE_IMAGE_GENERATION` /
> `AUTO_INSTALL_IMAGE_GEN_FILTER` / `AUTO_ATTACH_IMAGE_GEN_FILTER`
> valves and is unchanged by this feature.

## Table of contents

- [Quickstart](#quickstart)
- [Image models](#image-models)
- [Pure-image-only vs multimodal](#pure-image-only-vs-multimodal)
- [Per-model deep dive](#per-model-deep-dive)
- [What settings a model offers](#what-settings-a-model-offers)
- [What the settings panel looks like](#what-the-settings-panel-looks-like)
- [The `help` command](#the-help-command)
- [Output rendering and message format](#output-rendering-and-message-format)
- [Pricing and cost display](#pricing-and-cost-display)
- [Configuration valves (admin)](#configuration-valves-admin)
- [Errors and troubleshooting](#errors-and-troubleshooting)
- [Architecture overview](#architecture-overview)
- [Limitations and non-goals](#limitations-and-non-goals)

---

## Quickstart

### For end users

1. Open a chat in Open WebUI.
2. In the model picker, choose any image-output model (e.g.
   `Sourceful: Riverflow V2 Pro`, `Black Forest Labs: FLUX.2 Pro`,
   `Google: Nano Banana 2 (Gemini 3.1 Flash Image Preview)`,
   `OpenAI: GPT-5 Image`).
   Image models look like normal chat models — they are not in a
   separate menu.
3. (Optional) Open the Integrations menu (puzzle-piece icon below the
   prompt input). Each image model has one settings row of its own,
   auto-attached and default-on.
4. (Optional) Click its settings icon to adjust the model's options
   before you send. Most of what you see there is what that model
   accepts — two models rarely offer the same set, and a model that does
   not accept an aspect ratio will not list it. **Output size** is on
   every panel whatever the model publishes. Models that answer only with
   a picture also carry **Provider options**, **Reference images** and
   **Reference image links**; a model that answers with text as well takes
   its references from the message itself, so those three are left off.
5. (Optional) Attach images. They are sent as references for the model to
   work from, newest last, unless you change **Reference images**.
6. Type your prompt and press send.
7. The chat shows the generated image inline (typically 5–30 seconds).
   The image renders as a normal image attachment that you can right-
   click to download, copy, or open full-size. On models whose providers
   render in passes, the status line reports each preview as it arrives.

### Model-specific help in chat

Typing the literal word `help` (no other text) into a chat against any
image model returns a curated model-specific help blurb covering:

- What the model is best known for
- Tips and pitfalls (how to prompt, when to use vs alternatives)
- What OpenRouter charges for it right now
- The controls this model publishes and what they do, listed together
  with the always-present ones above that this model carries

This is the fastest way to learn a model without leaving the chat. Try
it on each image model — the answers are different for every one (the auto-router `openrouter/auto` is a routing layer rather than a generator).

### For administrators

Out-of-the-box defaults are sensible for most deployments:

```
ENABLE_OPENROUTER_IMAGE_GENERATION = True
AUTO_INSTALL_IMAGE_FILTERS = True       # one settings row per image model
AUTO_ATTACH_IMAGE_FILTERS  = True       # attaches each row to its own model
AUTO_DEFAULT_IMAGE_FILTERS = True       # those settings are on-by-default per chat
DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS = True   # see below
```

#### Built-in tools on image and video models

An image or video model answers with a picture or a clip, not a tool call.
Offering it Open WebUI's built-in tools — web search, code execution and the
rest — usually ends in a turn that fails or comes back empty, and the cause is
hard to spot because the model's `Built-in tools` box still looks ticked.

With `DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS` on, which is the default, the pipe
unticks that box on each image and video model at the moment it first adds the
model. The box is unticked rather than the tools quietly withheld, so the state
is visible on the model's page.

If you want tools on one of these models, tick the box back on for it. Your
choice is kept: the pipe fills this setting in only where a model has none yet,
and never overwrites one you made. Turning the valve off stops the pipe setting
it on any model.

This needs `UPDATE_MODEL_CAPABILITIES` on, since that is the switch that lets
the pipe write to a model's capability boxes at all.

#### File context on image and video models

`UPDATE_MODEL_CAPABILITIES` also unticks Open WebUI's `File context` box on
image and video models, and it does so whatever
`DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS` is set to. Left on — which is Open
WebUI's own default — an attachment makes Open WebUI run an extra billed
round-trip that turns the conversation into search queries and pastes the
retrieved text into what was meant to be a picture or clip prompt. As with the
tools box, the pipe fills it in only where a model has no setting yet, so a box
you tick yourself is left alone.

There is one visible side effect on video models, and it is the desired one:
once `File context` is off, attachments the model was sent stay in the chat as
normal attachments instead of being taken out of the request.

If the per-model filters do not appear in the Integrations menu, check:

- `AUTO_INSTALL_IMAGE_FILTERS` and `AUTO_ATTACH_IMAGE_FILTERS` are both
  `True`.
- The pipe has been called at least once with a logged-in user.
- Open WebUI Admin → Functions lists one entry per image model, each named
  after the model it belongs to. A model whose options have never been read
  from OpenRouter has no entry; the next catalogue refresh retries.

If a panel is installed but out of date, check `AUTO_INSTALL_IMAGE_FILTERS`
again: while it is off the pipe leaves an installed panel exactly as it is, and
writes a warning to its log naming any panel whose stored version no longer
matches what this release would install.

**Access control for non-admin users.** Pure-image-only models (FLUX,
Sourceful Riverflow non-multimodal, Seedream) are inserted PRIVATE by
default per the standard `NEW_MODEL_ACCESS_CONTROL` valve (default
`admins`). Non-admin users will not see image-only models in the picker
until an admin explicitly grants access via Admin → Models →
[image model row] → Access. Multimodal models (gpt-5-image, gemini-
image variants) follow the standard chat-catalog access policy.

**Auto-default re-assert.** All applicable image filters are
re-defaulted to enabled on every catalog metadata sync (typically
every pipe `pipes()` call). If you manually disable an image filter
for a chat, the next sync will re-default it. Set
`AUTO_DEFAULT_IMAGE_FILTERS=False` to opt out of the re-assert.

**Web Tools / Web Search guard.** The `OR Web Tools` filter (web
search + web fetch + datetime) and the `OR Web Search` overlay are
**capability-gated to skip image-output models** — these models do
not support tool use and would fail with an HTTP 404 "No endpoints
found that support tool use" if web search were attached. The same
guard applies to video-generation models.

See [Configuration valves](#configuration-valves-admin) for the
image-specific valves; the master `MODEL_CATALOG_REFRESH_SECONDS`
TTL is shared with the video and chat catalogs.

---

## Image models

| Model id | Display name | Output | Cost rate |
|----------|--------------|:----:|-----------|
| `black-forest-labs/flux.2-flex` | Black Forest Labs: FLUX.2 Flex | image only | Mid-tier FLUX.2 |
| `black-forest-labs/flux.2-klein-4b` | Black Forest Labs: FLUX.2 Klein 4B | image only | Smallest, cheapest FLUX |
| `black-forest-labs/flux.2-max` | Black Forest Labs: FLUX.2 Max | image only | Highest FLUX.2 tier |
| `black-forest-labs/flux.2-pro` | Black Forest Labs: FLUX.2 Pro | image only | Premium FLUX.2 |
| `bytedance-seed/seedream-4.5` | ByteDance Seed: Seedream 4.5 | image only | Image-only with sampling controls |
| `google/gemini-2.5-flash-image` | Google: Nano Banana (Gemini 2.5 Flash Image) | text + image | Standard Gemini multimodal |
| `google/gemini-3-pro-image` | Google: Nano Banana Pro (Gemini 3 Pro Image) | text + image | Most capable Gemini image model |
| `google/gemini-3-pro-image-preview` | Google: Nano Banana Pro (Gemini 3 Pro Image Preview) | text + image | Premium Gemini 3 with image |
| `google/gemini-3.1-flash-image` | Google: Nano Banana 2 (Gemini 3.1 Flash Image) | text + image | Pro-level quality at Flash speed |
| `google/gemini-3.1-flash-image-preview` | Google: Nano Banana 2 (Gemini 3.1 Flash Image Preview) | text + image | Cost-optimized; 512 tier for cheap iteration |
| `google/gemini-3.1-flash-lite-image` | Google: Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image) | text + image | Fastest, cheapest Gemini image model |
| `krea/krea-2-large` | Krea: Krea 2 Large | image only | Rawer, less house-styled output |
| `krea/krea-2-medium` | Krea: Krea 2 Medium | image only | Krea's balanced default |
| `krea/krea-2-medium-turbo` | Krea: Krea 2 Medium Turbo | image only | Distilled Krea 2 Medium for fast iteration |
| `microsoft/mai-image-2.5` | Microsoft: MAI-Image-2.5 | image only | Token-priced via Azure AI Foundry |
| `microsoft/mai-image-2.5-pro` | Microsoft: MAI-Image-2.5 Pro | image only | Larger MAI-Image-2.5; token-priced via Azure |
| `openai/gpt-5-image` | OpenAI: GPT-5 Image | text + image | GPT-5 chat token economics |
| `openai/gpt-5-image-mini` | OpenAI: GPT-5 Image Mini | text + image | Cheaper GPT-5 Image tier |
| `openai/gpt-5.4-image-2` | OpenAI: GPT-5.4 Image 2 | text + image | Updated GPT-5.4 generation |
| `openai/gpt-image-1` | OpenAI: GPT Image 1 | image only | Up to 16 reference images for edits |
| `openai/gpt-image-1-mini` | OpenAI: GPT Image 1 Mini | image only | Cheaper, faster GPT Image 1 |
| `openai/gpt-image-2` | OpenAI: GPT Image 2 | image only | OpenAI's newest image model |
| `qwen/qwen-image-3` | Qwen: Qwen Image 3 | image only | Text and detail down to ~10px |
| `qwen/qwen-image-3-pro` | Qwen: Qwen Image 3 Pro | image only | Larger Qwen 3 with more world knowledge |
| `recraft/recraft-v3` | Recraft: Recraft V3 | image only | Typography champion; tuned for long-form text |
| `recraft/recraft-v4` | Recraft: Recraft V4 | image only | Design-taste rebuild; 1024x1024; ~10s/image |
| `recraft/recraft-v4-pro` | Recraft: Recraft V4 Pro | image only | Print-ready 2048x2048 (~30s/image); flat per-image fee |
| `recraft/recraft-v4-pro-vector` | Recraft: Recraft V4 Pro Vector | image only | High-fidelity SVG finals |
| `recraft/recraft-v4-vector` | Recraft: Recraft V4 Vector | image only | True SVG output; scales without quality loss |
| `recraft/recraft-v4.1` | Recraft: Recraft V4.1 | image only | Aesthetic refresh of V4; 1024x1024; ~10s/image |
| `recraft/recraft-v4.1-pro` | Recraft: Recraft V4.1 Pro | image only | Print-ready 2048x2048 with V4.1 aesthetics |
| `recraft/recraft-v4.1-pro-vector` | Recraft: Recraft V4.1 Pro Vector | image only | Highest-fidelity SVG finals |
| `recraft/recraft-v4.1-utility` | Recraft: Recraft V4.1 Utility | image only | General-purpose (non-aesthetic) tier; 1024x1024 |
| `recraft/recraft-v4.1-utility-pro` | Recraft: Recraft V4.1 Utility Pro | image only | General-purpose at 2048x2048 |
| `recraft/recraft-v4.1-vector` | Recraft: Recraft V4.1 Vector | image only | V4.1 aesthetics, SVG output |
| `sourceful/riverflow-v2-fast` | Sourceful: Riverflow V2 Fast | image only | Faster, cheaper Sourceful |
| `sourceful/riverflow-v2-pro` | Sourceful: Riverflow V2 Pro | image only | Premium Sourceful tier |
| `sourceful/riverflow-v2.5-fast` | Sourceful: Riverflow V2.5 Fast | image only | Cheapest Riverflow tier; settled per job |
| `sourceful/riverflow-v2.5-pro` | Sourceful: Riverflow V2.5 Pro | image only | Premium Riverflow tier; settled per job |
| `x-ai/grok-imagine-image-quality` | SpaceXAI: Grok Imagine Image Quality | image only | Per generated image; 2K dearer than 1K |

Pick model selection rules of thumb:

- **Long-form text or precise text-at-position in images** → Recraft V3
  (every Recraft variant takes `text_layout` for explicit placement, and
  V3 is the one tuned to render full sentences/paragraphs cleanly).
- **Print-ready high-resolution finals** → Recraft V4/V4.1 Pro
  (2048x2048 with design-taste output), or a model that publishes a 4K
  tier: Riverflow V2.5 Pro, Riverflow V2 Pro/Fast, Seedream 4.5, Nano
  Banana Pro or Nano Banana 2.
- **Transparent or opaque backgrounds** → Riverflow 2.5 Pro/Fast, GPT
  Image 1/1 Mini or GPT-5 Image/Image Mini (**Background**: auto,
  transparent or opaque). GPT Image 2 and GPT-5.4 Image 2 offer auto and
  opaque only.
- **Vector (SVG) output for logos/icons** → Recraft V4/V4.1 Vector
  variants (true `<svg>`, scales infinitely).
- **Ultrawide / ultratall layouts (4:1, 1:4, 8:1, 1:8)** → the Gemini
  3.1 Flash Image line: GA, preview and Lite each publish all four. Qwen
  Image 3 and 3 Pro take 4:1 and 1:4.
- **Tall phone-screen ratios (9:19.5, 9:20)** → xAI Grok Imagine Image
  Quality (14 ratios) or Seedream 4.5 (18 ratios); those two are the only
  recorded models publishing them. Seedream 4.5 also reaches 4K and up to
  10 images a request, Grok 1K/2K and exactly one.
- **An `auto` ratio, letting the model choose the frame shape** → widely
  published: the four FLUX.2 variants, both MAI-Image-2.5 models, GPT
  Image 1/1 Mini/2, every Recraft variant, every Riverflow variant,
  Seedream 4.5 and Grok Imagine Image Quality all offer it.
- **Multiple variations per request** → Seedream 4.5 or any of the GPT
  Image models (up to 10 per call), or Qwen Image 3/3 Pro and the Recraft
  variants (up to 6); cost scales linearly. Grok Imagine Image Quality
  makes one image per request, so it draws no such control.
- **Cheap iteration** → Gemini 3.1 Flash Image Preview at 512 (far
  fewer pixels than 1K on a token-billed model), FLUX.2 Klein 4B,
  Riverflow V2.5 Fast, or Recraft V4.1 Utility.
- **Photorealism / hero shots** → FLUX.2 Pro/Max, Riverflow V2.5 Pro,
  Gemini 3 Pro Image, Recraft V4.1 Pro, or Microsoft MAI-Image-2.5.
- **Color-palette-driven design (corporate brand colors)** → any
  Recraft variant, through its `controls` setting; Recraft's own API
  reference documents what that setting accepts.
- **Want commentary alongside the image (chat-style)** → multimodal
  text+image models (GPT-5 Image, Gemini Image variants).
- **Deterministic regeneration with same prompt** → any model that
  publishes a seed: the four FLUX.2 variants, Seedream 4.5, the three
  Krea 2 variants, and Qwen Image 3 and 3 Pro.
- **Don't know which to pick** → `openrouter/auto` routes for you.

---

## Pure-image-only vs multimodal

OpenRouter image-output models split into two categories that this pipe
handles differently:

### Pure-image-only

These models answer with a picture and no text. Examples: all 4 Sourceful
Riverflow variants, all 4 FLUX.2 variants, ByteDance Seedream 4.5.

- **Where they come from**: OpenRouter's own list of models that output
  pictures. The list is re-read on every catalog refresh, so a model
  OpenRouter withdraws disappears from the picker on the next sync, and
  one it adds appears without anything being configured.
- **Multimodal dedupe**: if a model has `text` in `output_modalities`,
  `register_image_models` skips it (those stay in the chat catalog).
- **Master-disable**: setting `ENABLE_OPENROUTER_IMAGE_GENERATION=False`
  stops the pipe reading the image list at all, and drops the models
  registered while it was on. Both happen on the next model-list build —
  the next time Open WebUI asks the pipe for its models — and the drop runs
  ahead of the catalogue refresh window, so it does not wait on
  `MODEL_CATALOG_REFRESH_SECONDS`. The models are already gone from that
  same model list.

### Multimodal (text + image)

Models with both `text` AND `image` in `output_modalities` — GPT-5
Image variants, Gemini Image variants. These already appear in the
chat catalog via the standard `/api/v1/models` endpoint and are NOT
re-registered as image-only. They answer with both a picture and text.

- **Settings**: these models get their own settings row too, built from
  what they publish, exactly like the image-only ones.
- **`openrouter/auto`**: this auto-router is treated as multimodal
  (universal input modalities). Lives in the chat catalog.

## Per-model deep dive

This section is written-up prose about most of the models, plus
`openrouter/auto`, which picks a model rather than being one. It does not
cover all of them. Qwen Image 3 and Qwen Image 3 Pro, the three Krea 2
tiers, GPT Image 1, GPT Image 1 Mini and GPT Image 2, Nano Banana Pro
(Gemini 3 Pro Image), Nano Banana 2 (Gemini 3.1 Flash Image), Nano Banana 2
Lite (Gemini 3.1 Flash Lite Image) and MAI-Image-2.5 Pro have no entry
below yet.

Every model answers `help`, written up here or not. Send `help` in a chat
with the model and the reply describes it, prints what it charges, and
lists the controls its own panel draws. Skip to a model that matches your
use case, or read them all to get a feel for the catalog.

### OpenAI: GPT-5 Image

> **id**: `openai/gpt-5-image` · **multimodal**

OpenAI's flagship multimodal text+image model — generates both text
response AND inline images per turn. Best for chat-style image
generation where you want commentary alongside the visual.

- **Multimodal output:** model decides when to emit images based on
  prompt — be explicit ("Generate an image of...") for reliability.
- **Already in chat catalog** — its own settings panel attaches like any
  other image model's.
- **Pricing follows GPT-5 chat token economics**; image output is
  included in completion tokens.

### OpenAI: GPT-5 Image Mini

> **id**: `openai/gpt-5-image-mini` · **multimodal**

Cost-efficient variant of GPT-5 Image with the same multimodal
text+image output. Best for high-volume image generation, drafts, and
iteration where premium-tier quality isn't required.

- Same prompting style as GPT-5 Image — be explicit about wanting
  images in the prompt.
- Lower cost-per-token than GPT-5 Image; ideal for prototyping and
  bulk runs.
- Its settings panel carries whatever this model publishes; type `help` to
  it to see the list.

### OpenAI: GPT-5.4 Image 2

> **id**: `openai/gpt-5.4-image-2` · **multimodal**

Updated GPT-5.4 generation of multimodal text+image output. Improved
prompt adherence and visual fidelity over GPT-5 Image.

- Successor to GPT-5 Image — it still answers with both text and
  pictures in one turn, at better quality.
- Use for production deliverables that need the latest OpenAI image
  model.

### Google: Nano Banana (Gemini 2.5 Flash Image)

> **id**: `google/gemini-2.5-flash-image` · **multimodal**

Google's standard Gemini multimodal text+image model. Best for
prompt-following tasks with cinematic composition and natural-looking
output. Outputs both text and image.

- Multimodal: model decides emission based on prompt; be explicit.
- Strong at photoreal scenes and prompt-faithful composition.

### Google: Nano Banana Pro (Gemini 3 Pro Image Preview)

> **id**: `google/gemini-3-pro-image-preview` · **multimodal**

Premium tier of Gemini 3 with native image output. Highest fidelity
Gemini image model OpenRouter exposes; best for hero shots and
high-detail outputs.

- Premium variant — higher cost than Flash; reserve for finals.
- Multimodal text+image output.

### Google: Nano Banana 2 (Gemini 3.1 Flash Image Preview)

> **id**: `google/gemini-3.1-flash-image-preview` · **multimodal**

Cost-optimized Gemini 3.1 with native image output, four extra aspect
ratios (1:4, 4:1, 1:8, 8:1) for ultrawide and tall layouts, and a 512
low-res tier for cheap iteration. The Gemini 3.1 Flash Image line is the
only one offering all four ratios — Qwen Image 3 and 3 Pro publish 1:4
and 4:1 — and the 512 tier is on this model and the GA release, not on
the Lite.

- Set aspect from this model's own aspect-ratio control; the values it
  offers are the ones this model published.
- 512 renders far fewer pixels than 1K — good for prompt iteration.

### OpenRouter: Auto (Image Routing)

> **id**: `openrouter/auto` · **router**

OpenRouter's automatic routing for image generation. Routes to the
best available image model based on prompt. Useful when you want
OpenRouter to pick rather than committing to a specific provider.

- Auto-routing — exact model used varies; check the response metadata
  for routed model id.
- Takes text, images, audio, files and video alongside the prompt, so
  almost anything you attach can go with it.

### Microsoft: MAI-Image-2.5

> **id**: `microsoft/mai-image-2.5` · **pure-image-only**

Microsoft's high-quality image generation model served via Azure AI
Foundry — photorealistic and artistic output from text prompts with
optional reference-image input. Best for general-purpose photoreal
work on Azure-backed infrastructure, billed by token rather than by
picture.

- Token-priced rather than per-image, so a long prompt costs more than
  a short one for the same picture.
- Multimodal input: accepts reference images alongside the text prompt
  for editing/guidance.

### Sourceful: Riverflow V2 Pro

> **id**: `sourceful/riverflow-v2-pro` · **pure-image-only**

Sourceful's premium tier — pure image-only output with custom font
rendering (`font_inputs`) and up to ten reference images for
image-to-image work. Strongest for marketing assets requiring exact text
rendering at scale.

- **PURE-image-only** — does NOT output text.
- **4.5MB request size limit** — pass image URLs instead of base64 to
  avoid bloat.

### Sourceful: Riverflow V2 Fast

> **id**: `sourceful/riverflow-v2-fast` · **pure-image-only**

Faster, cheaper variant of Riverflow V2 — same Sourceful extension
(`font_inputs`) and the same per-reference charge, at lower quality and
reduced cost. Best for iteration before committing to a Pro render.

- Same caveats as Riverflow V2 Pro: pure-image-only, 4.5MB request
  limit, image URLs preferred.
- Use Fast for prompt iteration and font/reference tuning; switch to
  Pro for finals.

### Sourceful: Riverflow V2.5 Pro

> **id**: `sourceful/riverflow-v2.5-pro` · **pure-image-only**

The most powerful variant of Sourceful's Riverflow 2.5 lineup — a
unified text-to-image and image-to-image family. Best for top-tier
control and quality-sensitive outputs: brand assets, marketing finals,
and work that benefits from the new 2.5 self-scoring and background
controls. Priced per image, rising with the output size you ask for.

- Publishes a **Background** choice (auto, transparent or opaque) and
  an output format, alongside 1K, 2K and 4K sizes.
- Pricing is dynamic: the published per-image rate is a starting point,
  and the final charge is settled per job from the processing it
  actually took.

### Sourceful: Riverflow V2.5 Fast

> **id**: `sourceful/riverflow-v2.5-fast` · **pure-image-only**

The speed-optimized variant of Sourceful's Riverflow 2.5 lineup — best
for production deployments and latency-critical workflows. Same unified
text-to-image and image-to-image family as Pro, at a fraction of the
cost, with the charge settled per job at completion. It is the narrower
of the two listings.

- PURE-image-only — does NOT output text.
- Use Fast for iteration and high-volume production; switch to V2.5
  Pro for quality-sensitive finals.
- Publishes the same **Background** choice (auto, transparent or opaque)
  as V2.5 Pro, but only 1K and 2K sizes and only JPEG output, and it
  takes fewer reference pictures per request. Switch to Pro for PNG or
  WebP, for 4K, or for more references than Fast will take.

### Black Forest Labs: FLUX.2 Pro

> **id**: `black-forest-labs/flux.2-pro` · **pure-image-only**

Black Forest Labs' premium FLUX.2 model — pure-image-only with strong
photorealism and prompt adherence. Best for high-quality deliverables.
**Supports seed for deterministic generation.**

- PURE-image-only — does NOT output text.
- Seed support enables deterministic regeneration with same prompt +
  seed.
- No Sourceful-only or Gemini-only extensions.

### Black Forest Labs: FLUX.2 Max

> **id**: `black-forest-labs/flux.2-max` · **pure-image-only**

Highest-tier FLUX.2 — best fidelity in the Black Forest Labs lineup.
Pure-image-only with seed support. Reserve for hero shots and finals
where Pro isn't enough.

- PURE-image-only — does NOT output text.
- Seed enables deterministic regeneration.
- Most expensive FLUX tier — use for finals only.

### Black Forest Labs: FLUX.2 Flex

> **id**: `black-forest-labs/flux.2-flex` · **pure-image-only**

Mid-tier FLUX.2 balancing quality and cost. Pure-image-only with seed
support. Best for general production work.

- PURE-image-only.
- Seed support; balanced cost-quality vs Pro/Max.

### Black Forest Labs: FLUX.2 Klein 4B

> **id**: `black-forest-labs/flux.2-klein-4b` · **pure-image-only**

Smallest FLUX.2 variant (4B parameters) — lowest cost in the FLUX
lineup. Pure-image-only with seed support. Best for high-volume / draft
work.

- PURE-image-only — does NOT output text.
- Seed support; cheapest FLUX tier.
- Quality trades against cost — use for iteration, not finals.

### ByteDance Seed: Seedream 4.5

> **id**: `bytedance-seed/seedream-4.5` · **pure-image-only**

ByteDance Seed's image-only model. Pure-image-only output; a seed and a
batch of up to 10 images per request give you several varied takes on one
prompt to choose between.

- PURE-image-only — does NOT output text.
- **Publishes a seed and up to 10 images per request** — ask for several
  takes at once and pick the one you want. OpenRouter publishes no range
  for the seed and does not promise the same seed repeats an image.

### Recraft: Recraft V3

> **id**: `recraft/recraft-v3` · **pure-image-only**

Recraft's typography champion — released October 2024, 20B parameters,
held #1 on the Artificial Analysis benchmark for 5+ consecutive months
at launch. The only AI image model that can render long-form text
(full sentences/paragraphs) reliably AND place text at exact positions
inside the image. Used in production by Shopify and Salesforce.
~1K resolution output, pure-image-only.

- PURE-image-only — does NOT output text in chat.
- **Publishes the full Recraft set**: `style`, `controls` and
  `text_layout` — the same three every Recraft variant takes.
- For text rendering: put exact wording in quotes in the prompt AND
  use `text_layout` to place each line exactly where you want it.
- `text_layout` uses normalized 0-1 coordinates; bbox is 4 corner
  [x,y] points (TL, TR, BR, BL).
- Image-to-image: only one input image supported.
- Style names: see [Recraft style list](https://www.recraft.ai/docs/api-reference/styles).
  This model draws pixels; for SVG, pick one of the Recraft Vector
  models.

### Recraft: Recraft V4

> **id**: `recraft/recraft-v4` · **pure-image-only**

Recraft's February 2026 ground-up rebuild — "design taste meets image
generation." 1024x1024 raster output, ~10s/image. Topped the
Hugging Face Text-to-Image Arena (blind human preference) over
Midjourney V8, DALL-E 3, FLUX, and Stable Diffusion. Strengths:
balanced composition, cohesive color, clean readable embedded text
(short / mid-length), and outputs that feel deliberate rather than
stock-like. Best for infographics, signage, packaging, and rapid
iteration on branded assets.

- PURE-image-only.
- Takes `style`, `controls` and `text_layout`, like every Recraft
  variant. For long-form text, V3 is the one tuned for it.
- Image-to-image: only one input image supported.
- Limitations: photorealistic human faces and hands can be unreliable;
  not the right tool for editorial portraiture.
- Use V4 for fast iteration / social / web; switch to V4 Pro for
  print-ready finals at 2K.

### Recraft: Recraft V4 Pro

> **id**: `recraft/recraft-v4-pro` · **pure-image-only**

Premium V4 — same design taste, 2x resolution. Outputs at 2048x2048
(~4 megapixels), ~30s/image. Built for print-ready work where fine
detail matters: magazine layouts, posters, billboards, packaging,
editorial illustration. Same prompt accuracy and creative judgment as
V4 but with sharper geometry, finer textures, and better
anatomy/realism in complex compositions.

- PURE-image-only.
- Same three settings as V4: `style`, `controls` and `text_layout`.
- ~3x slower than V4 due to higher resolution — reserve for finals,
  not iteration.
- **Flat per-image fee** rather than per-token, so prompt length does
  not change what a render costs.
- Image-to-image: only one input image supported.
- Same human-subject limitations as V4.

### Recraft: Recraft V4 Vector

> **id**: `recraft/recraft-v4-vector` · **pure-image-only (SVG)**

Vector (SVG) variant of V4 — true `<svg>` output destined for logos,
icon sets, and flat illustrations that need to scale and edit
downstream. The SVG arrives complete and Open WebUI draws it in the
chat at full sharpness, whatever size you view it at.

- Output is SVG, not PNG/JPEG — scales infinitely without quality loss.
- Prefer simple, graphic prompts (logos, icons, flat illustrations)
  over photoreal subjects.
- `style`, `controls` and `text_layout` are sent through, but how the
  vector model honors them is undocumented — verify visually.
- Image-to-image input is rasterised internally; output is SVG either
  way.

### Recraft: Recraft V4 Pro Vector

> **id**: `recraft/recraft-v4-pro-vector` · **pure-image-only (SVG)**

High-fidelity SVG counterpart to V4 Pro — ~2K-equivalent detail in
true vector output. Use V4 Vector for iteration; V4 Pro Vector for
final logo/brand deliverables.

- Same SVG caveats as V4 Vector (graphic prompts, undocumented color
  steering, rasterised i2i input).
- Higher fidelity, slower, costlier than V4 Vector — reserve for finals.

### Recraft: Recraft V4.1

> **id**: `recraft/recraft-v4.1` · **pure-image-only**

V4.1 is Recraft's May 2026 aesthetic refresh of V4 — same 1024x1024
raster output, but tuned for stronger
composition, color cohesion, and visual polish. Best for marketing
assets, social posts, and hero imagery where V4 felt
almost-but-not-quite-right aesthetically. Same speed envelope as V4
(~10s/image).

- PURE-image-only.
- Same three settings as V4: `style`, `controls` and `text_layout`.
- Drop-in successor to V4 — try V4.1 first; fall back to V4 if its
  aesthetic doesn't suit a specific brand.
- Image-to-image: only one input image supported.
- For general-purpose / cost-sensitive work without aesthetic emphasis,
  prefer the V4.1 Utility variants.

### Recraft: Recraft V4.1 Pro

> **id**: `recraft/recraft-v4.1-pro` · **pure-image-only**

High-resolution counterpart to V4.1 — same aesthetic tuning at
2048x2048 (~4 MP), ~30s/image. Built for print-ready aesthetic work:
magazine layouts, posters, billboards, editorial illustration. Use
V4.1 for iteration, V4.1 Pro for finals.

- PURE-image-only; same knob set as V4.1.
- ~3x slower than V4.1 due to higher resolution — reserve for finals.
- Same human-subject limitations as the V4 family.

### Recraft: Recraft V4.1 Vector

> **id**: `recraft/recraft-v4.1-vector` · **pure-image-only (SVG)**

Vector (SVG) variant of V4.1 — V4.1's aesthetic tuning with ~1K
equivalent detail and true `<svg>` output. Best for aesthetic-driven
logos, icon sets, and flat illustrations destined for vector editing.
Faster/cheaper than V4.1 Pro Vector for iteration.

- Same SVG caveats as the V4 vector variants.
- Use V4.1 Vector for iteration; V4.1 Pro Vector for finals.

### Recraft: Recraft V4.1 Pro Vector

> **id**: `recraft/recraft-v4.1-pro-vector` · **pure-image-only (SVG)**

The highest-fidelity vector variant — V4.1 aesthetics, ~2K equivalent
detail, true SVG. Best for high-polish logos, editorial icon sets, and
brand assets that must scale and edit downstream.

- Same SVG caveats as the other vector variants.
- Try V4.1 Pro Vector first for final vector work; it carries the same
  aesthetic advantage over V4 Pro Vector that V4.1 has over V4.

### Recraft: Recraft V4.1 Utility

> **id**: `recraft/recraft-v4.1-utility` · **pure-image-only**

Recraft's general-purpose V4.1 variant — drops the aesthetic-tuning
bias in exchange for broader subject coverage. Best for spot
illustrations, diagrams, placeholder/stock imagery, and any work where
"on-brand aesthetics" is not the goal. 1024x1024.

- Pick Utility over regular V4.1 when you need versatility, not polish.
- Same three settings as V4.1: `style`, `controls` and `text_layout`.
- Utility and regular V4.1 are priced the same per image, so switch on
  the look you want — regular V4.1 for its aesthetic tuning, or V4.1 Pro
  when you need print resolution.

### Recraft: Recraft V4.1 Utility Pro

> **id**: `recraft/recraft-v4.1-utility-pro` · **pure-image-only**

High-resolution counterpart to V4.1 Utility — 2048x2048 (~4 MP)
general-purpose raster output. Use for general-purpose finals where
aesthetic polish is not the goal; otherwise prefer V4.1 Pro.

- ~3x slower than V4.1 Utility due to higher resolution.
- Same knob set and limitations as the rest of the V4.1 family.

### SpaceXAI: Grok Imagine Image Quality

> **id**: `x-ai/grok-imagine-image-quality` · **pure-image-only**

xAI's fast, high-fidelity image generation and editing model. Accepts
text prompts and optional reference images; produces photorealistic
outputs at 1K or 2K. Best for photoreal scenes, compositional control,
and workflows that need tall phone-screen aspect ratios (9:19.5, 9:20 —
Seedream 4.5 is the only other model offering them) or an `auto` ratio
that lets the model pick frame shape from the prompt.

- One image per request. Its published contract fixes the number of
  images at 1, so no **Number of images** control is drawn — asking for
  several means sending several requests. For variations in a single
  request use Seedream 4.5 or the GPT Image models (up to 10), or Qwen
  Image 3/3 Pro and the Recraft variants (up to 6).
- Multimodal input: pair the prompt with reference images for
  editing/style transfer.
- Charged per generated image, at a higher rate for 2K than for 1K,
  and reference images you supply are charged on top.

---

## What settings a model offers

Every image model on OpenRouter publishes its own list of the settings it
accepts — which aspect ratios, which output sizes, how many images at once, and
any options specific to the company that runs it. The pipe reads that list and
builds the model's settings row from it.

So most of a model's controls are not a fixed list, and are not listed in this
document. Where a ratio, size or quality tier is offered, that model accepts it;
where a setting appears for one model and not another, only the first supports
it.

The inline server tool a chat model calls mid-answer (see
[`ENABLE_IMAGE_GENERATION`](valves_and_configuration_atlas.md)) is built the same
way, from the same published list — it just reads the list of the one model an
admin pointed it at. It shows six controls. **Quality**, **Aspect ratio**,
**Background**, **Output format** and **Output compression** are there for every
drawing model; the sixth is either **Resolution**, for a model that publishes a
list of size tiers, or **Output size**, for one that does not. No model gets
both. Where the model publishes the values it takes, the control becomes a list
of exactly those; where it publishes nothing, the control offers what
OpenRouter's image API accepts in general and the company running the model
decides what to do with the value.

One control does appear on every per-model panel, because a request carries it
for any model and no model's published list mentions it:

- **Output size** — either a size tier (`512`, `1K`, `2K`, `4K`) or exact pixels
  such as `1024x1024`. A tier sets the same thing as **Resolution** and still
  takes its shape from **Aspect ratio**; what it is measured against depends on
  the model. Sixteen of the forty publish a tier list of their own, and on those
  a tier outside the list is withheld rather than sent, and named. The other
  twenty-four publish no list, so a tier is measured only against those four
  names and then goes out for the company running the model to interpret.
  Anything that is neither one of the four names nor pixels is withheld and named
  on every model. Exact pixels settle the picture on their own: no model
  publishes a list of pixel sizes, so those go out as typed and the company
  running the model decides — and because they already fix the dimensions,
  **Resolution** is not sent alongside them, nor is **Aspect ratio** unless it is
  the shape you typed. Anything dropped that way is named in a toast at the time;
  Open WebUI does not keep toasts with the message, so it is gone once the page
  reloads.

Models that answer only with a picture carry three more, for the same reason. A
model that answers with text as well takes its references from the message
itself, and the chat request it travels on takes no provider options, so these
three are not drawn for it:

- **Provider options** — extra settings for the company running the model, as a
  JSON object keyed by its OpenRouter name. Use it for anything the panel does
  not already offer.
- **Reference images** — which of the pictures attached to the turn are sent as
  references: every one of them (oldest first), only the most recent, or none.
- **Reference image links** — a JSON list of `https` links or `data:` URLs to
  use as well as, or instead of, the attached pictures. These go first, so they
  survive on models that take only one reference.

A request carries at most 16 references. Where a model publishes a lower limit
the lower one applies, and anything over the limit is dropped with a note saying
how many and why.

Every reference OpenRouter would have to fetch is checked against the same
address policy the pipe applies to any other outbound fetch, whether it was typed
into the links box or arrived as a picture in the conversation. A `data:` URL
carries the picture itself, so there is nothing to fetch and nothing to check. A
typed link the deployment will not fetch fails the request outright rather than
generating a picture that quietly ignored it; a picture already in the chat is
dropped instead, with a note saying how many and why, because a single unreachable
address in an old turn would otherwise fail every later request in that chat.

Other consequences worth knowing:

- **The controls change when the model does.** If OpenRouter adds a size to a
  model, it appears after the next catalogue refresh without an update to the
  pipe.
- **A settings list is kept once read.** If a later refresh cannot read it — the
  request timed out, say — the model keeps the settings from the last successful
  read rather than losing them. A model whose list has never been read gets no
  settings panel at all rather than a guessed one; it still generates images,
  using its own defaults, and the next refresh retries.
- **A model served by more than one company offers what they agree on, plus what
  only some of them take.** Which company serves a given request is normally
  decided when you send it. Values they all accept are offered plainly; a value
  only some of them accept is offered too and says so on the control, and
  choosing it pins the request to the companies that accept it, so it is sent
  and honoured rather than quietly turning into something else. Where OpenRouter
  names none of those companies for routing, the value still goes out and a
  warning says so before it does. Nothing checks afterwards which company served
  the request, so no message names one.
- **A provider option with published choices becomes a dropdown.** Where
  OpenRouter documents what a provider option accepts, the control lists those
  values instead of taking free text, so a misspelling cannot reach the wire.
  Today that is `moderation` on the OpenAI image models.

To see what a specific model accepts, type `help` to it in a chat. The reply
lists the settings read from that model's published list, together with the
always-present controls that model carries. A value only some of the companies
serving the model accept is shown among the choices with a note saying so.

## What the settings panel looks like

Each model's settings are visible to end users as form fields under that
model's settings icon in the Integrations menu. A parameter the model
publishes gets a plain-English label — `n` appears as **Number of images**.
An option specific to the provider keeps the name OpenRouter publishes for
it, because only that provider's own documentation defines what it means.

### Installing the panels (admin)

Panels are installed and refreshed on their own while
`AUTO_INSTALL_IMAGE_FILTERS` is on, one per image model, and only for models
actually offered in this workspace. A model whose panel fails to install does
not hold up the others.

While that valve is **off**, an already-installed panel is left exactly as it
is. If a newer release changes what that panel should offer, the change is not
delivered and a warning is written to the pipe's log naming the panel; turn the
valve back on to let it update.

### Attaching the panels (admin)

While `AUTO_ATTACH_IMAGE_FILTERS` is on, each model's panel is attached to that
model, and a panel that no longer applies is detached again — which is what
happens if a model stops producing images, or if a panel is renamed.
`AUTO_DEFAULT_IMAGE_FILTERS` additionally starts each new chat with the panel
already switched on.

---

## The `help` command

Typing the literal word `help` (no other text — case does not matter,
exactly four characters) in a chat against any image model returns a
curated help blurb for that specific model.

Help is the model's curated description, followed by what it charges and a
control list read from that model's own published settings.

The reply below is for `recraft/recraft-v3`, reproducible from the contract recorded in
this project's own test data for that model. **The money in it is an
illustration, not a quote**: the live reply reads the rate from OpenRouter at
the moment you ask, and the figure below was captured from one snapshot. For
what a model costs today, run `help` against it or look it up on OpenRouter's
pricing page.

```
# Recraft: Recraft V3

Recraft's typography champion — the only AI image model that can render long-form text (full sentences and paragraphs) reliably AND place text at exact positions inside the image. 20B parameters, released Oct 2024, held #1 on the Artificial Analysis benchmark for 5+ consecutive months at launch (beating Midjourney/DALL-E/FLUX). Used in production by Shopify and Salesforce. Pure-image-only at ~1K resolution. Best for posters, signage, packaging, marketing assets with embedded copy.

## Tips & pitfalls
- PURE-image-only — does NOT output text in chat.
- V3 is the Recraft tuned for long-form text, so it holds full sentences and paragraphs where the others hold short lines. Like every Recraft variant it takes `style`, `controls` and `text_layout`.
- For text rendering: put exact wording in quotes in your prompt AND use `text_layout` to place each line exactly where you want it.
- Style names: see https://www.recraft.ai/docs/api-reference/styles. This model draws pixels; for SVG, pick one of the Recraft Vector models.
- text_layout: array of {text, bbox} where bbox is 4 [x,y] corners in 0-1 coords (order: TL, TR, BR, BL).
- If you need newer composition or cleaner geometry, V4 and V4.1 offer the same settings with a different look.

## Cost

- Each image it makes: $0.04 per image

Where the company running the model reports a charge above zero, it is shown on the status line when it finishes, as long as usage details are on: that is your own Show usage details setting once you have set it, and the site default your administrator chooses until then.

## Controls
- **Provider options** — Extra settings for the company that runs this model, as a JSON object keyed by its OpenRouter name. Use it for anything this panel does not already offer. Empty sends nothing.
- **Reference images** — Which attached images go to the model as references. auto sends every picture in this chat, and where the model takes fewer than you attached the most recent ones are kept; latest-only sends just the most recent; none sends none of them.
- **Reference image links** — Reference images to use as well as, or instead of, the attached ones: a JSON list of https links or data URLs. These are placed first, so they survive when the model takes fewer references than are on offer.
- **Aspect ratio** — Frame shape. Choices: 1:1, 4:3, 3:4, 16:9, 9:16, auto.
- **Output size** — Either a size tier (512, 1K, 2K or 4K) or exact pixels written like 1024x1024. This model publishes no tiers of its own, so a tier is checked only against those four names and then goes out for the company running the model to interpret. It still takes its shape from Aspect ratio. Exact pixels settle the picture on their own, so Aspect ratio is not sent alongside them unless it is the shape you typed. A toast says so at the time, which Open WebUI does not keep with the message: it is gone once the page reloads. No model publishes a list of pixel sizes, so exact pixels go out as typed and the company running this one decides what to do with them. Empty leaves it unset.
- **Number of images** — How many images this request asks for. Accepts 1 to 6.
- **style** — a setting this model's provider accepts.
- **controls** — a setting this model's provider accepts.
- **text_layout** — a setting this model's provider accepts.
```

The `## Controls` section covers the settings that model publishes
together with the ones every panel carries whatever it publishes. A model
that publishes none of its own says so rather than showing an empty
section. On a model that answers only with a picture, Provider options,
Reference images and Reference image links head the list, ahead of
anything the model publishes. Output size comes after the published lists
of choices and before the rest of what the model publishes — its number
ranges and the settings named after what the company running it accepts.
Where a value is accepted by only some of the companies serving the model,
both the panel and this list offer it and say so.

The `## Cost` section comes from the same record and is read fresh every
time you ask, so it follows OpenRouter's rates without a new release.
Each published charge is one line naming what is charged for — the
images it makes, the images you supply, references, fonts, your prompt
text — and the rate, in the unit OpenRouter states: per image, per
megapixel, or per million tokens. A tier such as 1K, 2K or 4K gets its
own line, because choosing a tier chooses a price. Where several
companies serve the model and publish different figures, each line names
the company. A token-billed model carries a note that the token count of
a picture is not published, so the price of one image cannot be worked
out from the rate. A charge whose unit is not one of the three gets a
line saying so and pointing at OpenRouter, rather than a made-up
conversion. A model that publishes no price says so rather than showing
an empty section; three of the forty do (`krea/krea-2-large`,
`krea/krea-2-medium`, `krea/krea-2-medium-turbo`).

If a model isn't in the curated dataset (newly added by OpenRouter
between catalog refreshes, for example), `help` falls back to the
catalog metadata — display name, description, output/input modalities.

---

## Output rendering and message format

A picture that arrives on the chat route is saved to Open WebUI's file
store and shown inline, exactly as one from a dedicated image model is.
Both end at the same stored file and the same message. Anything larger
than `BASE64_MAX_SIZE_MB` is rejected rather than stored.

The rendered message looks like:

```markdown
![Generated image](/api/v1/files/01HX2K3D5N4P9F8GZQ2WV3R5BC/content)
```

OWUI displays the image inline with a download/copy/view-fullsize
context menu. The file is registered in OWUI's `Files` table linked to
the chat, surviving page reload.

---

## Pricing and cost display

Rates come from each model's own published contract, refreshed on the
shared catalog TTL (`MODEL_CATALOG_REFRESH_SECONDS`). OpenRouter states
them per image, per megapixel or per token depending on the model, so
there is no one formula behind every image charge: a per-image or
per-megapixel model is not billed from token counts at all.

Where the company running the model reports a charge above zero for the
generation, it is shown on the status line when it finishes, as long as
usage details are on: that is your own Show usage details setting once
you have set it, and the site default your administrator chooses until
then; with usage details off, that line carries the elapsed time alone.
The amount is the one OpenRouter returns with the generation rather than
one this pipe works out, and it arrives as a single total, not a
per-item breakdown.

On Riverflow V2 Pro and V2 Fast, the published contract prices each
reference image you supply at $0.20 and each font file at $0.03, and the
`help` reply lists both on their own lines under `## Cost`. That section
is in the reply only when the model's contract could be read for it —
either it was already held from building that model's panel, or it was
fetched there and then, which happens only while
`ENABLE_OPENROUTER_IMAGE_GENERATION` is on. When neither holds, the reply
stops after the description and tips: no `## Cost` and no `## Controls` at
all. So a reply naming no reference charge means the contract was not
read, not that references are free.

---

## Configuration valves (admin)

Four valves control the native image-generation subsystem. All are
visible in Admin → Functions → OpenRouter pipe → Valves. Catalog TTL
is shared with chat/video catalogs (`MODEL_CATALOG_REFRESH_SECONDS`).

| Valve | Default | Range | Purpose |
|-------|---------|-------|---------|
| `ENABLE_OPENROUTER_IMAGE_GENERATION` | `True` | bool | Master kill switch. False drops pure-image-only models from the model list AND clears them from OWUI's catalog on the next model-list build, ahead of the catalogue refresh window, so it does not wait on `MODEL_CATALOG_REFRESH_SECONDS`. Multimodal models stay since they're in the chat catalog. |
| `AUTO_INSTALL_IMAGE_FILTERS` | `True` | bool | Install and keep current one settings panel per image model, built from what that model publishes. Every panel also carries `Output size`, where a tier is checked against the tiers that model publishes -- or against `512`, `1K`, `2K` and `4K` where it publishes none -- while exact pixels such as `1024x1024` travel as typed; and a model that answers with a picture and no text carries `Provider options`, `Reference images` and `Reference image links` on top of that. A model whose settings list has never been read gets no panel; one read before keeps its last successful set. |
| `AUTO_ATTACH_IMAGE_FILTERS` | `True` | bool | Attach each model's own settings panel to it, so its settings appear in the chat controls when that model is selected. A single model can opt out with the `disable_image_filter_auto_attach` advanced parameter. |
| `AUTO_DEFAULT_IMAGE_FILTERS` | `True` | bool | Keep attached image filters enabled by default per chat. Re-asserted on every catalog metadata sync. |

Related (existing) valves:

| Valve | Default | Purpose |
|-------|---------|---------|
| `MODEL_CATALOG_REFRESH_SECONDS` | `3600` | TTL governing how often the image catalog is re-fetched from `/api/v1/models?output_modalities=image`. |
| `BASE64_MAX_SIZE_MB` | (multimodal section) | Cap on decoded image size before file persistence. |

Tuning hints:

- **Disabling image generation completely**:
  `ENABLE_OPENROUTER_IMAGE_GENERATION=False`. Pure-image-only models
  vanish from the dropdown on next sync; multimodal models remain
  (they're in the chat catalog).
- **Want filters created but not auto-attached**: set
  `AUTO_INSTALL_IMAGE_FILTERS=True`, `AUTO_ATTACH_IMAGE_FILTERS=False`.
  Useful for testing — admins can attach manually via Admin → Models
  → [model row] → Filters.
- **Want auto-attach but not auto-default**:
  `AUTO_DEFAULT_IMAGE_FILTERS=False`. The filter shows in Integrations
  menu as off-by-default; users opt in per chat.

---

## Errors and troubleshooting

### `OR Web Tools` filter showing on image models / 404 "No endpoints found that support tool use"

If the user sees the OR Web Tools filter toggle on an image-output
model (e.g. Sourceful Riverflow), and enabling it causes a 404, the
capability gate may not be working. The pipe explicitly excludes
image-output and video-generation models from Web Tools attach: a model
that answers with a picture or a clip is never given the Web Tools
filter, whatever the attach valves are set to.

If a model is mis-detected, check its `architecture.output_modalities`
in the OpenRouter catalog — only models with `image` (and not `text`,
or without `text` for pure-image-only) trigger the gate. Toggle
`ENABLE_OPENROUTER_IMAGE_GENERATION` off → save → on → save to force a
catalog sync; the gate is re-evaluated each sync.

### Pure-image-only model not appearing in dropdown

The catalog hasn't been fetched yet, or the master switch is off.
Check:

- `ENABLE_OPENROUTER_IMAGE_GENERATION=True` is set.
- The pipe has been called at least once with a logged-in user.
- The pipe's API key is valid — `pipes()` returns early without
  registering the image catalog if auth fails.
- Check the pipe logs for `Registered N OpenRouter image-output
  model(s) into the catalog.` — if missing, the fetch failed
  silently (the call is wrapped in a try/except that logs a warning).

### Multimodal image model (gpt-5-image, gemini-image) appears but no image filter is attached

The filter installer didn't run, or the catalog metadata sync hasn't
completed since install. Check:

- `AUTO_INSTALL_IMAGE_FILTERS=True` and `AUTO_ATTACH_IMAGE_FILTERS=True`.
- Admin → Functions has an entry named after that model.
- Restart the pipe to force a fresh `pipes()` cycle, or toggle
  `AUTO_ATTACH_IMAGE_FILTERS` off → save → on → save.

### A setting I expected is not in the model's panel

That model does not publish it. The panel lists what the model told
OpenRouter it accepts, so a missing setting means the model would not
have honoured it. Type `help` to the model to see its published list.

**Output size** is there on every panel whatever the model publishes.
Models that answer only with a picture also carry Provider options,
Reference images and Reference image links — so on those, if what you
want is a provider-specific setting the panel does not name, put it in
**Provider options** as a JSON object keyed by the company's OpenRouter
name.

If a model has **no panel at all**, its settings list has never been read
successfully — the next refresh retries. It still generates images
meanwhile, using its own defaults.

### A choice is marked as accepted by only some providers

Some models are served by several companies that do not all accept the
same values. Rather than hide a value one of them does take, the panel
offers it and says so on the control. Choosing it pins the request to the
companies that do accept it, so the value is sent and used — you are not
told anything, because there is nothing to report. Where OpenRouter names
none of those companies for routing, the value is still sent and a warning
says so beforehand. If you have pinned a company yourself, your pin wins
and the value goes to it as typed.

### A value I typed was rejected as invalid JSON

Settings that take a list or an object — a provider's own options,
usually — are typed as JSON. If what you typed starts with `[` or `{`
it has to be valid JSON, and the error names the setting it came from.
A bare number is read as a number, since several provider options take
one. Everything else is sent as the text you typed and does not need
quoting. `NaN`, `Infinity` and numbers too large to write down are
refused rather than sent.

### The picture appeared in stages, or the status line kept moving

Some companies render an image in passes and publish that they can send
it as it goes. Where every company that could serve the request does,
the request asks for that form and each preview is reported on the
status line. A model that draws in text rather than pixels — SVG —
streams that text instead of preview pictures, and that is reported once
as `Drawing the image…`. No model published today does both: the only
endpoints offering the streamed form are OpenAI's, and they send preview
pictures, so nothing currently reaches that second line. Either way the
finished image is what lands in the chat, and follow-up edits behave
exactly as they do otherwise.

If a streamed generation stops before the finished image arrives, it is
a failed generation and there is nothing to salvage. It costs nothing:
OpenRouter bills image generation all or nothing, so previews already
delivered are not charged. Re-submit to retry.

### Aspect ratio not honored on `openrouter/auto`

Auto-routing means OpenRouter picks the underlying model. Some
providers may not honor all aspect ratios. The router maps to the
closest equivalent. To get exact aspect ratio, pick a specific model.

### Image generation succeeds but no image renders inline

Check:

- The OpenRouter response has `message.images` populated (not an
  empty list).
- `BASE64_MAX_SIZE_MB` is large enough — if the decoded image
  exceeds it, persistence fails silently and the markdown contains a
  broken file reference.
- The pipe has filesystem write access to its temp dir and OWUI
  storage (Local/S3/GCS/Azure) is healthy.
- Check the pipe logs for storage errors.

### Body validation error mentioning `image_config`

Multimodal models carry their settings to OpenRouter in one object called
`image_config`, and the pipe checks the request against its own schema
before sending it. If the pipe log shows a validation error naming
`image_config`, that object was rejected as the wrong shape — it has to be
a set of named settings, and something upstream supplied a single value
instead.

Nothing an administrator sets in Valves can cause this, and nothing a user
types in a settings panel can either: both write named settings. It means
the running code has been modified or a hand-edited bundle is installed.
Reinstall the released bundle. Image generation on models that answer only
with a picture is unaffected — those requests do not carry this object.

---

## Architecture overview

Roughly, in order of who-calls-who:

```
pipes()
  ├─ ensure chat catalog loaded
  ├─ ensure video catalog loaded
  └─ ensure_image_catalog_loaded()   <- called on every build; the master
        valve is checked INSIDE it, not at this call site
          ├─ if ENABLE_OPENROUTER_IMAGE_GENERATION is off: drop any models
          │  registered while it was on, then return -- ahead of the TTL
          │  check, which is why the picker empties on this build rather
          │  than a TTL later
          ├─ TTL-gated fetch (cache_seconds = MODEL_CATALOG_REFRESH_SECONDS)
          ├─ /api/v1/models?output_modalities=image via OpenRouterImageClient
          ├─ if a filter valve is on, read each model's published contract
          │  from /api/v1/images/models/<id>/endpoints — 8 reads at a time,
          │  whole sweep capped at 45s by the pipe; a model that could not
          │  be read this pass keeps its last good record
          └─ register_image_models()
                ├─ skip multimodal (text in output_modalities)
                ├─ stale-norm cleanup (drop models removed from catalog)
                ├─ publish as one run of plain assignments with no await
                │  between them, so no request sees a half-updated catalog
                └─ features = {"image_output", "image_gen_tool"}, plus
                   {"vision", "file_input"} when the model takes images in

  └─ if AUTO_INSTALL_IMAGE_FILTERS:
        ensure_openrouter_image_filter_function_ids(available_models)
          ├─ one settings row per image model, built from its own contract
          ├─ a model with no readable contract gets none
          ├─ each install in own try/except — partial failures isolated
          └─ retire rows left over from the fixed-variant design

  └─ catalog_manager._update_or_insert_model_with_metadata()
        ├─ pipe_capabilities.image_output gate
        ├─ web_tools_supported = ... and not image_output
        ├─ _apply_list_filter_ids(meta_dict)       — writes filterIds
        └─ _apply_list_default_filter_ids(meta_dict) — writes defaultFilterIds

settings-row inlet (Open WebUI runs this before the pipe sees the body)
  ├─ model gate: every id form OWUI produces, and no other model
  ├─ merge the chosen values into body.image_config, per key
  ├─ typed values parsed as JSON only when they open a container
  └─ provider options and reference choices go to the pipe's metadata key

pipe(body, ...)
  └─ orchestrator._inject_image_modalities(body)
        ├─ no-op if model not in registry or no image in output_modalities
        ├─ pure-image: body["modalities"] = ["image"]
        └─ multimodal: body["modalities"] = ["image", "text"]

  └─ orchestrator: uses_dedicated_image_api(spec) chooses the transport
        ├─ image-only model → POST /api/v1/images
        │     ├─ image_config split against the model's published record
        │     ├─ what the record names goes top-level; a provider setting
        │     │  it names goes under provider.options.<slug>
        │     ├─ whatever was withheld is reported to the user
        │     ├─ each returned image persisted → file URL
        │     └─ renders "![alt](file_url)"
        └─ multimodal model → chat completions
              ├─ image_config fitted to the same kind of record, with any
              │  provider setting kept inside image_config
              ├─ response carries message.images
              ├─ chat_completions_adapter parses message.images
              ├─ streaming_core materialises the entry → persists → file URL
              └─ streaming_core renders "![alt](file_url)"
  └─ OWUI renders inline image
```

**One place decides the transport.** `uses_dedicated_image_api()` in
[`models/registry.py`](../open_webui_openrouter_pipe/models/registry.py) is
the only answer to "does this model go to the image endpoint". It says yes
when the model's published `output_modalities` contain `image` and do not
contain `text`. Everything that needs to know asks it: the request
dispatch, the data-retention gate, the `help` reply, and the settings-row
installer — which is handed the answer as an argument it cannot omit,
rather than working it out a second time. Two readings of that question
would let a request open one transport while being prepared for the other.

Note what does *not* decide it: the `modalities` written into the body a
few lines earlier. That value is prepared for every image-output model,
but a model that goes to the image endpoint has its request built fresh
from prompt and settings, so its `modalities` is never sent. It is the
multimodal models, staying on chat completions, that actually carry it.

**Turning the feature off.** With `ENABLE_OPENROUTER_IMAGE_GENERATION` set
to `False`, the pipe stops reading the image catalog and drops the models
registered while it was on. The drop runs on the next model-list build,
ahead of the catalogue refresh window, so the models are gone from that
same model list rather than lingering for up to
`MODEL_CATALOG_REFRESH_SECONDS`.

Key invariant: **both branches render the same markdown**. Multimodal
models keep the streaming path that has always handled them, and both
produce `![alt](file_url)`. That is what keeps iterative editing working:
the next request re-parses that markdown back into an input image.

Both branches also read the same contract. A multimodal model never
reaches the image endpoint, but it still gets a settings row built from
its published record, and its `image_config` is put through that record on
the way out: a value outside the published domain is withheld and reported
rather than sent, and a key no record names is withheld too. Where the two
differ is where a provider-specific setting may sit. OpenRouter's image
schema defines a `provider.options` block keyed by provider; its chat
schema does not — the chat provider block is closed (`additionalProperties:
false`) and lists no `options` — and OpenRouter documents chat's
`image_config` as "provider-specific image configuration options" in its
own right. So on chat completions a provider setting the record does name
stays inside `image_config`. A model whose contract is not in hand is left
exactly as it arrived; a read that failed is not a contract that shrank.

Key files:

- [`integrations/image_catalog.py`](../open_webui_openrouter_pipe/integrations/image_catalog.py)
  — TTL-gated catalog fetch and the bounded sweep that reads each model's
  published contract.
- [`integrations/image_client.py`](../open_webui_openrouter_pipe/integrations/image_client.py)
  — HTTP client for the image model catalog, the per-model endpoint record
  that publishes which knobs a model accepts, and image generation itself.
- [`integrations/image.py`](../open_webui_openrouter_pipe/integrations/image.py)
  — `ImageGenerationAdapter`: runs the image-endpoint request end to end,
  and on the chat side fits a multimodal model's `image_config` to its
  record. Either way it gates each requested knob against what the model
  published, reports the ones it withheld, persists the returned images
  and renders the markdown.
- [`integrations/provider_options.py`](../open_webui_openrouter_pipe/integrations/provider_options.py)
  — the single reader of a request's provider block, the per-transport set
  of provider keys OpenRouter documents, and the choice of which provider
  slug carries a value that cannot be duplicated across providers.
- [`integrations/image_help.py`](../open_webui_openrouter_pipe/integrations/image_help.py)
  — `_IMAGE_PER_MODEL_HELP_DATA` (per-model prose), `render_image_help()`
  (control list read from the model's endpoint record).
- [`filters/image_filter_renderer.py`](../open_webui_openrouter_pipe/filters/image_filter_renderer.py)
  — `build_image_model_filter_spec()` turns a model's endpoint record into
  its knob set, taking `dedicated_image_api` as a required keyword because
  the three always-on controls belong only to models on the image endpoint;
  `render_image_model_filter_source()` renders one settings row from that.
- [`filters/filter_manager.py::ensure_openrouter_image_filter_function_ids`](../open_webui_openrouter_pipe/filters/filter_manager.py)
  — installs rows in OWUI Functions table; returns per-model id mapping.
- [`models/catalog_manager.py`](../open_webui_openrouter_pipe/models/catalog_manager.py)
  — `_apply_list_filter_ids`, `_apply_list_default_filter_ids`,
  `pipe_capabilities.image_output` gate, capability-gated
  `web_tools_supported` exclusion.
- [`models/registry.py`](../open_webui_openrouter_pipe/models/registry.py)
  — `uses_dedicated_image_api` (the transport decision) and
  `register_image_models` (atomic registry merge with stale-norm cleanup;
  multimodal dedupe).
- [`requests/orchestrator.py::_inject_image_modalities`](../open_webui_openrouter_pipe/requests/orchestrator.py)
  — body modalities injection, and the dispatch that follows it.
- [`api/transforms.py`](../open_webui_openrouter_pipe/api/transforms.py)
  — `CompletionsBody`, whose `image_config` field is a mapping of named
  settings (`dict[str, Any] | None`).
- [`core/config.py`](../open_webui_openrouter_pipe/core/config.py)
  — the image valves and `_OPENROUTER_IMAGE_FILTER_MARKER`, the marker that
  identifies a settings row this pipe installed.

Also involved, shared with other features:

- [`api/gateway/chat_completions_adapter.py`](../open_webui_openrouter_pipe/api/gateway/chat_completions_adapter.py)
  — reads `message.images` off a chat response.
- [`streaming/streaming_core.py`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
  — materialises those images, persists them and renders the markdown, via
  `_persist_generated_image`.
- [`storage/owui_files.py`](../open_webui_openrouter_pipe/storage/owui_files.py)
  — `OwuiFileGateway.upload_to_owui_storage`, the single write into OWUI
  file storage used by both branches.
- The legacy `openrouter_image_gen` filter (OpenAI Responses-tool wiring),
  which is a separate feature on its own valves.

### `_inject_image_modalities()` (orchestrator)

The body modification happens at [`requests/orchestrator.py`](../open_webui_openrouter_pipe/requests/orchestrator.py)
in `_inject_image_modalities()`. Its decision logic, with the debug
logging that follows it left out:

```python
def _inject_image_modalities(body, *, logger=None):
    if not isinstance(body, dict):
        return
    raw_model = body.get("model")
    if not isinstance(raw_model, str) or not raw_model:
        return
    if "modalities" in body:
        return
    spec = OpenRouterModelRegistry.spec(raw_model)
    if not isinstance(spec, dict):
        return
    arch = spec.get("architecture") or {}
    out_mods = arch.get("output_modalities") or []
    if "image" not in out_mods:
        return
    if "text" in out_mods:
        body["modalities"] = ["image", "text"]
    else:
        body["modalities"] = ["image"]
```

Key behavior:

- **No-op on non-image models.** No injection if `output_modalities`
  doesn't contain `image`.
- **Respects user override.** If `body.modalities` is already set
  (manual config or an older settings row), the orchestrator leaves it
  alone.
- **Pure-image gets `["image"]`**, multimodal gets `["image", "text"]`.

It runs on every request, before the transport is chosen, so a model bound
for the image endpoint is written too — and, as above, that value is not
what routes it and is not sent. On chat completions the value does go out,
and asking for both modalities is what gets a multimodal model to answer
with a picture as well as text.

---

## Limitations and non-goals

- **Synchronous only.** Image generation is a single request, whichever
  transport it takes — no polling lifecycle, no resume, no disconnect
  recovery.
  If the request fails or the user disconnects, the generation is lost.
  Re-submit to retry.
- **Previews, not partial results.** A few providers — currently only
  OpenAI's — render an image in passes and publish
  `supports_streaming: true` on their endpoint record. Where every
  endpoint that could serve the request publishes it, the pipe asks for
  the streamed form and reports each preview as a status line, so the
  chat shows movement instead of a spinner. A model that streams a
  text-based format instead of preview pictures — SVG — sends text
  chunks, which OpenRouter's images API documents as its own event; the
  pipe reports that once as "Drawing the image…". No recorded contract
  combines the two, because every Recraft vector endpoint publishes
  `supports_streaming: false`, so nothing reaches that line today; it
  becomes reachable the day a vendor enables streaming on a vector model.
  Either way the answer is the
  same `![alt](file_url)` markdown built from the finished image, so
  nothing downstream — including iterative editing — sees a difference.
  A stream that ends before the finished image is a failed generation:
  OpenRouter bills image generation all-or-nothing, so previews already
  delivered cost nothing and there is nothing to salvage.
- **No batch generation.** One request, one image (or set of images
  the model emits per turn). For batch use, send multiple chats.
- **Multimodal models may emit text without an image.** GPT-5 Image
  and Gemini Image variants decide based on prompt. Be explicit in
  the prompt ("generate an image of...") if you want guaranteed
  image output.
- **No video output from these models.** Image-output models do not
  generate video. For video, use the
  [video-generation feature](openrouter_video_generation.md).
