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
- [The chat filter UI (UserValves)](#the-chat-filter-ui-uservalves)
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
   `Google: Gemini 3.1 Flash Image (Preview)`, `OpenAI: GPT-5 Image`).
   Image models look like normal chat models — they are not in a
   separate menu.
3. (Optional) Open the Integrations menu (puzzle-piece icon below the
   prompt input). Each image model has one settings row of its own,
   auto-attached and default-on.
4. (Optional) Click its settings icon to adjust the model's options
   before you send. What you see there is what that model accepts —
   nothing more. Two models rarely offer the same set, and a model that
   does not accept an aspect ratio will not list it.
5. Type your prompt and press send.
6. The chat shows the generated image inline (typically 5–30 seconds).
   The image renders as a normal image attachment that you can right-
   click to download, copy, or open full-size.

### Model-specific help in chat

Typing the literal word `help` (no other text) into a chat against any
image model returns a curated model-specific help blurb covering:

- What the model is best known for
- Tips and pitfalls (how to prompt, when to use vs alternatives)
- Every filter knob exposed for this model and what it does

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

If the per-model filters do not appear in the Integrations menu, check:

- `AUTO_INSTALL_IMAGE_FILTERS` and `AUTO_ATTACH_IMAGE_FILTERS` are both
  `True`.
- The pipe has been called at least once with a logged-in user (the
  filters install during `pipes()` warmup).
- Open WebUI Admin → Functions lists one entry per image model, each named
  after the model it belongs to. A model whose options have never been read
  from OpenRouter has no entry; the next catalogue refresh retries.

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
guard applies to video-generation models. See
[`models/catalog_manager.py`](../open_webui_openrouter_pipe/models/catalog_manager.py)
where `web_tools_supported` checks for `image_output` and
`video_generation` capabilities.

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
| `google/gemini-2.5-flash-image` | Google: Gemini 2.5 Flash Image | text + image | Standard Gemini multimodal |
| `google/gemini-3-pro-image` | Google: Nano Banana Pro (Gemini 3 Pro Image) | text + image | Most capable Gemini image model |
| `google/gemini-3-pro-image-preview` | Google: Gemini 3 Pro Image (Preview) | text + image | Premium Gemini 3 with image |
| `google/gemini-3.1-flash-image` | Google: Nano Banana 2 (Gemini 3.1 Flash Image) | text + image | Pro-level quality at Flash speed |
| `google/gemini-3.1-flash-image-preview` | Google: Gemini 3.1 Flash Image (Preview) | text + image | Cost-optimized; 0.5K is ~50% cheaper than 1K |
| `google/gemini-3.1-flash-lite-image` | Google: Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image) | text + image | Fastest, cheapest Gemini image model |
| `krea/krea-2-large` | Krea: Krea 2 Large | image only | Rawer, less house-styled output |
| `krea/krea-2-medium` | Krea: Krea 2 Medium | image only | Krea's balanced default |
| `krea/krea-2-medium-turbo` | Krea: Krea 2 Medium Turbo | image only | Distilled Krea 2 Medium for fast iteration |
| `microsoft/mai-image-2.5` | Microsoft: MAI-Image-2.5 | image only | $5/M tokens via Azure AI Foundry |
| `microsoft/mai-image-2.5-pro` | Microsoft: MAI-Image-2.5 Pro | image only | Larger MAI-Image-2.5; token-priced via Azure |
| `openai/gpt-5-image` | OpenAI: GPT-5 Image | text + image | GPT-5 chat token economics |
| `openai/gpt-5-image-mini` | OpenAI: GPT-5 Image Mini | text + image | Cheaper GPT-5 Image tier |
| `openai/gpt-5.4-image-2` | OpenAI: GPT-5.4 Image 2 | text + image | Updated GPT-5.4 generation |
| `openai/gpt-image-1` | OpenAI: GPT Image 1 | image only | Up to 16 reference images for edits |
| `openai/gpt-image-1-mini` | OpenAI: GPT Image 1 Mini | image only | Cheaper, faster GPT Image 1 |
| `openai/gpt-image-2` | OpenAI: GPT Image 2 | image only | OpenAI's newest image model |
| `qwen/qwen-image-3` | Qwen: Qwen Image 3 | image only | Text and detail down to ~10px |
| `qwen/qwen-image-3-pro` | Qwen: Qwen Image 3 Pro | image only | Larger Qwen 3 with more world knowledge |
| `recraft/recraft-v3` | Recraft: Recraft V3 | image only | Typography champion; only model with text-at-position |
| `recraft/recraft-v4` | Recraft: Recraft V4 | image only | Design-taste rebuild; 1024x1024; ~10s/image |
| `recraft/recraft-v4-pro` | Recraft: Recraft V4 Pro | image only | Print-ready 2048x2048 (~30s/image); $0.25/image |
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
| `sourceful/riverflow-v2.5-fast` | Sourceful: Riverflow V2.5 Fast | image only | From $0.019/image (finalized per job) |
| `sourceful/riverflow-v2.5-pro` | Sourceful: Riverflow V2.5 Pro | image only | From $0.13/image (finalized per job) |
| `x-ai/grok-imagine-image-quality` | xAI: Grok Imagine Image Quality | image only | $0.01/image |

Pick model selection rules of thumb:

- **Long-form text or precise text-at-position in images** → Recraft V3
  (the only model with `text_layout` for explicit placement; renders
  full sentences/paragraphs cleanly).
- **Print-ready high-resolution finals** → Recraft V4/V4.1 Pro
  (2048x2048 with design-taste output) or FLUX.2 Max (4K).
- **Transparent or solid-color backgrounds** → Riverflow 2.5 Pro/Fast
  (`background_mode` original/transparent/solid + `background_hex_color`).
- **Self-scored candidate selection (model picks its best attempt)** →
  Riverflow 2.5 Pro/Fast (`scoring_prompt` + `scoring_rubric`).
- **Vector (SVG) output for logos/icons** → Recraft V4/V4.1 Vector
  variants (true `<svg>`, scales infinitely).
- **Ultrawide / ultratall layouts (4:1, 1:4, 8:1, 1:8)** → Gemini 3.1
  Flash Image — GA or preview (only line with extended aspect ratios).
- **Tall phone-screen ratios (9:19.5, 9:20) or auto-ratio** → xAI Grok
  Imagine Image Quality (14-value Grok ratio set).
- **Multiple variations per request** → Grok Imagine Image Quality
  (`n` up to 10 images per call; cost scales linearly).
- **Cheap iteration** → Gemini 3.1 Flash Image Preview at 0.5K (~50%
  cheaper than 1K), FLUX.2 Klein 4B, Riverflow V2.5 Fast (from
  $0.019/image), or Recraft V4.1 Utility.
- **Photorealism / hero shots** → FLUX.2 Pro/Max, Riverflow V2.5 Pro,
  Gemini 3 Pro Image, Recraft V4.1 Pro, or Microsoft MAI-Image-2.5.
- **Color-palette-driven design (corporate brand colors)** → any
  Recraft variant (`rgb_colors` + `background_rgb_color`).
- **Want commentary alongside the image (chat-style)** → multimodal
  text+image models (GPT-5 Image, Gemini Image variants).
- **Deterministic regeneration with same prompt** → FLUX.2 family
  (only models with seed support).
- **Don't know which to pick** → `openrouter/auto` routes for you.

---

## Pure-image-only vs multimodal

OpenRouter image-output models split into two categories that this pipe
handles differently:

### Pure-image-only

These models output ONLY images, no text. The orchestrator injects
`modalities: ["image"]` into the request body. Examples: all 7 Sourceful
Riverflow variants, all 4 FLUX.2 variants, ByteDance Seedream 4.5.

- **Catalog source**: discovered via `/api/v1/models?output_modalities=image`
  in [`integrations/image_catalog.py`](../open_webui_openrouter_pipe/integrations/image_catalog.py).
- **Registration**: registered into the shared model registry via
  `OpenRouterModelRegistry.register_image_models()` ([`models/registry.py`](../open_webui_openrouter_pipe/models/registry.py))
  with `features = {"image_output", "image_gen_tool"}`. Stale-norm
  cleanup runs on every refresh — if a model is dropped from the
  catalog it disappears from the dropdown on next sync.
- **Multimodal dedupe**: if a model has `text` in `output_modalities`,
  `register_image_models` skips it (those stay in the chat catalog).
- **Master-disable cleanup**: setting
  `ENABLE_OPENROUTER_IMAGE_GENERATION=False` calls
  `register_image_models([])` and `reset_image_fetch_timestamp()` so
  models vanish from OWUI's dropdown immediately.

### Multimodal (text + image)

Models with both `text` AND `image` in `output_modalities` — GPT-5
Image variants, Gemini Image variants. These already appear in the
chat catalog via the standard `/api/v1/models` endpoint and are NOT
re-registered as image-only. The orchestrator injects
`modalities: ["image", "text"]` to ensure both modalities are emitted.

- **Settings**: these models get their own settings row too, built from
  what they publish, exactly like the image-only ones.
- **`openrouter/auto`**: this auto-router is treated as multimodal
  (universal input modalities). Lives in the chat catalog.

### `_inject_image_modalities()` (orchestrator)

The body modification happens at [`requests/orchestrator.py`](../open_webui_openrouter_pipe/requests/orchestrator.py)
in `_inject_image_modalities()`:

```python
def _inject_image_modalities(body, *, logger=None):
    if not isinstance(body, dict):
        return
    raw_model = body.get("model")
    if not isinstance(raw_model, str) or not raw_model:
        return
    if "modalities" in body:  # respect explicit user setting
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
  (manual config or older filter), the orchestrator leaves it alone.
- **Pure-image gets `["image"]`** to suppress text output.
- **Multimodal gets `["image", "text"]`** to allow both.

---

## Per-model deep dive

This section is written-up prose about each model. The in-chat `help`
command carries a shorter version of it, followed by that model's live
control list. Skip to a model that matches your use case, or read them
all to get a feel for the catalog. All curated entries live in
[`integrations/image_help.py`](../open_webui_openrouter_pipe/integrations/image_help.py)
in `_IMAGE_PER_MODEL_HELP_DATA`.

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

- Successor to GPT-5 Image — same modalities + image_config schema,
  improved quality.
- Use for production deliverables that need the latest OpenAI image
  model.

### Google: Gemini 2.5 Flash Image

> **id**: `google/gemini-2.5-flash-image` · **multimodal**

Google's standard Gemini multimodal text+image model. Best for
prompt-following tasks with cinematic composition and natural-looking
output. Outputs both text and image.

- Multimodal: model decides emission based on prompt; be explicit.
- Strong at photoreal scenes and prompt-faithful composition.

### Google: Gemini 3 Pro Image (Preview)

> **id**: `google/gemini-3-pro-image-preview` · **multimodal**

Premium tier of Gemini 3 with native image output. Highest fidelity
Gemini image model OpenRouter exposes; best for hero shots and
high-detail outputs.

- Premium variant — higher cost than Flash; reserve for finals.
- Multimodal text+image output.

### Google: Gemini 3.1 Flash Image (Preview)

> **id**: `google/gemini-3.1-flash-image-preview` · **multimodal**

Cost-optimized Gemini 3.1 with native image output AND unique extended
knobs: 4 extra aspect ratios (1:4, 4:1, 1:8, 8:1) for ultrawide/tall
layouts AND a 0.5K low-res tier for cheap iteration. **Only Gemini
variant with these extensions.**

- Set aspect from this model's own aspect-ratio control; the values it
  offers are the ones this model published.
- 0.5K is ~50% cheaper than 1K — good for prompt iteration.

### OpenRouter: Auto (Image Routing)

> **id**: `openrouter/auto` · **router**

OpenRouter's automatic routing for image generation. Routes to the
best available image model based on prompt. Useful when you want
OpenRouter to pick rather than committing to a specific provider.

- Auto-routing — exact model used varies; check the response metadata
  for routed model id.
- Universal input modalities (text + image + audio + file + video) —
  flexible request shape.

### Microsoft: MAI-Image-2.5

> **id**: `microsoft/mai-image-2.5` · **pure-image-only**

Microsoft's high-quality image generation model served via Azure AI
Foundry — photorealistic and artistic output from text prompts with
optional reference-image input. Best for general-purpose photoreal
work on Azure-backed infrastructure with token-based pricing ($5/M
tokens) instead of per-image billing.

- Token-priced ($5/M) rather than per-image — long prompts cost
  proportionally more.
- Multimodal input: accepts reference images alongside the text prompt
  for editing/guidance.

### Sourceful: Riverflow V2 Pro

> **id**: `sourceful/riverflow-v2-pro` · **pure-image-only**

Sourceful's premium tier — pure image-only output with custom font
rendering and image-to-image super-resolution. Strongest for marketing
assets requiring exact text rendering at scale.

- **PURE-image-only** — does NOT output text. Filter writes
  `modalities=["image"]` for this model.
- **4.5MB request size limit** — pass image URLs instead of base64 to
  avoid bloat.

### Sourceful: Riverflow V2 Fast

> **id**: `sourceful/riverflow-v2-fast` · **pure-image-only**

Faster, cheaper variant of Riverflow V2 — same Sourceful extensions
(`font_inputs`, `super_resolution_references`) at lower quality and
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
controls. From $0.13/image (finalized per job at completion).

- Supports reasoning effort up to xhigh (low/medium/high/xhigh).
- Pricing is dynamic — the from-$0.13/image floor is finalized per job
  based on billable processing.

### Sourceful: Riverflow V2.5 Fast

> **id**: `sourceful/riverflow-v2.5-fast` · **pure-image-only**

The speed-optimized variant of Sourceful's Riverflow 2.5 lineup — best
for production deployments and latency-critical workflows. Same
unified text-to-image and image-to-image family and the same 2.5
extras as Pro at a fraction of the cost. From $0.019/image (finalized
per job at completion).

- PURE-image-only — does NOT output text.
- Use Fast for iteration and high-volume production; switch to V2.5
  Pro for quality-sensitive finals.
  plus the 2.5 extras (scoring_prompt, scoring_rubric,
  background_mode, background_hex_color).
- Supports reasoning effort low/medium/high (xhigh is Pro-only).

### Sourceful: Riverflow V2 Max (Preview)

> **id**: `sourceful/riverflow-v2-max-preview` · **pure-image-only**

Preview release of the highest-tier Riverflow variant. Higher fidelity
than Pro but preview status means specs may shift. Pure-image-only
output.

- Preview — quality and pricing may change without notice.

### Sourceful: Riverflow V2 Standard (Preview)

> **id**: `sourceful/riverflow-v2-standard-preview` · **pure-image-only**

Standard preview release of Riverflow V2 — entry-tier quality and
pricing. Pure-image-only.

- Preview status — specs may change.

### Sourceful: Riverflow V2 Fast (Preview)

> **id**: `sourceful/riverflow-v2-fast-preview` · **pure-image-only**

Preview release of the fastest Riverflow tier. Pure-image-only with
reduced quality versus Pro/Standard at lower cost.

- Preview — pricing/quality may shift.

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

ByteDance Seed's image-only model. Pure-image-only output; supports
temperature and top_p for controlled generation.

- PURE-image-only — does NOT output text.
- **Supports `temperature`/`top_p` (unusual for image models)** —
  useful for varied outputs from same prompt.

### Recraft: Recraft V3

> **id**: `recraft/recraft-v3` · **pure-image-only**

Recraft's typography champion — released October 2024, 20B parameters,
held #1 on the Artificial Analysis benchmark for 5+ consecutive months
at launch. The only AI image model that can render long-form text
(full sentences/paragraphs) reliably AND place text at exact positions
inside the image. Used in production by Shopify and Salesforce.
~1K resolution output, pure-image-only.

- PURE-image-only — does NOT output text in chat.
- **Has the FULL Recraft knob set** (5 image_config params): strength,
  rgb_colors, background_rgb_color, plus V3-only `style` and
  `text_layout`. V4/V4 Pro lack the last two.
- For text rendering: put exact wording in quotes in the prompt AND
  use `text_layout` for precise placement (V3-exclusive feature).
- `text_layout` uses normalized 0-1 coordinates; bbox is 4 corner
  [x,y] points (TL, TR, BR, BL).
- Image-to-image: only one input image supported. Use `strength`
  (0.0-1.0) to control deviation; default 0.5.
- Style names: see [Recraft style list](https://www.recraft.ai/docs/api-reference/styles).
  Vector styles NOT supported via OpenRouter.

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
- Does NOT support `style` or `text_layout` — those are V3 ONLY.
  For long-form text or precise placement use V3.
- Has `strength` + `rgb_colors` + `background_rgb_color` (3 Recraft
  image_config params).
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
- Same image_config knobs as V4 (strength + rgb_colors +
  background_rgb_color); NO style or text_layout (V3 only).
- ~3x slower than V4 due to higher resolution — reserve for finals,
  not iteration.
- **$0.25 per image** — flat per-image fee, not per-token.
- Image-to-image: only one input image supported.
- Same human-subject limitations as V4.

### Recraft: Recraft V4 Vector

> **id**: `recraft/recraft-v4-vector` · **pure-image-only (SVG)**

Vector (SVG) variant of V4 — true `<svg>` output destined for logos,
icon sets, and flat illustrations that need to scale and edit
downstream. OpenRouter returns the SVG inline as a
`data:image/svg+xml;base64,...` URL; OWUI renders it natively.

- Output is SVG, not PNG/JPEG — scales infinitely without quality loss.
- Prefer simple, graphic prompts (logos, icons, flat illustrations)
  over photoreal subjects.
- `rgb_colors`/`background_rgb_color` are sent through, but how the
  vector model honors them is undocumented — verify visually.
- `strength` image-to-image works but input is rasterised internally;
  output is SVG either way.

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
raster output, same image_config surface, but tuned for stronger
composition, color cohesion, and visual polish. Best for marketing
assets, social posts, and hero imagery where V4 felt
almost-but-not-quite-right aesthetically. Same speed envelope as V4
(~10s/image).

- PURE-image-only.
- Same knobs as V4 (strength + rgb_colors + background_rgb_color);
  NO style or text_layout (V3 ONLY).
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
bias in exchange for broader subject coverage and cheaper generation.
Best for spot illustrations, diagrams, placeholder/stock imagery, and
any work where "on-brand aesthetics" is not the goal. 1024x1024.

- Pick Utility over regular V4.1 when you need versatility, not polish.
- Same knob set as V4.1; NO style or text_layout (V3 ONLY).
- Switch to regular V4.1 (aesthetic) or V4.1 Pro (print) when output
  quality matters.

### Recraft: Recraft V4.1 Utility Pro

> **id**: `recraft/recraft-v4.1-utility-pro` · **pure-image-only**

High-resolution counterpart to V4.1 Utility — 2048x2048 (~4 MP)
general-purpose raster output. Use for general-purpose finals where
aesthetic polish is not the goal; otherwise prefer V4.1 Pro.

- ~3x slower than V4.1 Utility due to higher resolution.
- Same knob set and limitations as the rest of the V4.1 family.

### xAI: Grok Imagine Image Quality

> **id**: `x-ai/grok-imagine-image-quality` · **pure-image-only**

xAI's fast, high-fidelity image generation and editing model. Accepts
text prompts and optional reference images; produces photorealistic
outputs at 1K or 2K. Best for photoreal scenes, compositional control,
and workflows that need Grok-only tall phone-screen aspect ratios
(9:19.5, 9:20, 1:2, 2:1) or an `auto` ratio that lets the model pick
frame shape from the prompt.

- `n` fans out 1-10 variations per request — cost scales linearly.
  Pick n=1 (default) for iteration; bump to 3-5 for exploration.
- Multimodal input: pair the prompt with reference images for
  editing/style transfer.
- Charged per image output ($0.01/image at OpenRouter's listed rate).

---

## What settings a model offers

Every image model on OpenRouter publishes its own list of the settings it
accepts — which aspect ratios, which output sizes, how many images at once, and
any options specific to the company that runs it. The pipe reads that list and
builds the model's settings row from it.

So there is no fixed set of controls, and no list of them in this document. What
you see in a model's settings **is** what that model accepts. If a ratio is not
offered, that model does not take it. If a setting appears for one model and not
another, only the first one supports it.

This matters because the alternative — a fixed set offered to everything — is
what the pipe used to do, and most models rejected part of it. A user could pick
an aspect ratio the model would not honour and get something else back with no
explanation.

Three consequences worth knowing:

- **The controls change when the model does.** If OpenRouter adds a size to a
  model, it appears after the next catalogue refresh without an update to the
  pipe.
- **A settings list is kept once read.** If a later refresh cannot read it — the
  request timed out, say — the model keeps the settings from the last successful
  read rather than losing them. A model whose list has never been read gets no
  settings row rather than a guessed one; it still generates images, using its
  own defaults, and the next refresh retries.
- **A model served by more than one company offers what they agree on.** Which
  one serves a given request is decided when you send it, so offering a setting
  only one of them takes would mean a control that sometimes silently does
  nothing.

To see what a specific model accepts, type `help` to it in a chat. The reply
lists its settings, read from the same source the settings row is built from.

## The chat filter UI (UserValves)

Each model's settings are visible to end users as form fields under that
model's settings icon in the Integrations menu. A parameter the model
publishes gets a plain-English label — `n` appears as **Number of images**.
An option specific to the provider keeps the name OpenRouter publishes for
it, because only that provider's own documentation defines what it means.

### Filter installation (admin)

Filter rows are auto-installed during `pipes()` warmup via
[`filters/filter_manager.py::ensure_openrouter_image_filter_function_ids`](../open_webui_openrouter_pipe/filters/filter_manager.py).
Each filter:

- Is installed lazily — only on first model that needs it (e.g. the
  Sourceful filter is only installed if a Sourceful Pro/Fast model is
  in the available list).
- Is wrapped in its own `try/except` so one filter's install failure
  doesn't block the others.
- Returns `dict[model_id, list[function_id]]` mapping each model to its
  applicable filter ids. Both `model_id` and `original_id` keys point
  to **separate list instances** (no aliasing — modifying one list
  doesn't affect the other).

### Filter attachment (admin)

The catalog metadata sync at [`models/catalog_manager.py::_apply_list_filter_ids`](../open_webui_openrouter_pipe/models/catalog_manager.py)
writes the per-model `filterIds` list into each model's metadata, with
removal-set logic that drops previously-attached ids no longer in the
current set. This handles renamed filter functions and capability
flips (e.g. if a model loses its `image_output` capability, its image
filters get cleaned up automatically).

`_apply_list_default_filter_ids` mirrors this for the
`defaultFilterIds` list (the "default-on" semantics).

---

## The `help` command

Typing the literal word `help` (no other text — case does not matter,
exactly four characters) in a chat against any image model returns a
curated help blurb for that specific model. The renderer is
[`integrations/image_help.py::render_image_help()`](../open_webui_openrouter_pipe/integrations/image_help.py).

Help is the model's curated description, followed by a control list read from that
model's own published settings. This is the real reply for `recraft/recraft-v3`,
reproducible from the contract recorded in
`tests/fixtures/openrouter_image_endpoints_recraft_recraft-v3.json`:

```
# Recraft: Recraft V3

Recraft's typography champion — the only AI image model that can render long-form text (full sentences and paragraphs) reliably AND place text at exact positions inside the image. 20B parameters, released Oct 2024, held #1 on the Artificial Analysis benchmark for 5+ consecutive months at launch (beating Midjourney/DALL-E/FLUX). Used in production by Shopify and Salesforce. Pure-image-only at ~1K resolution. Best for posters, signage, packaging, marketing assets with embedded copy.

## Tips & pitfalls
- PURE-image-only — does NOT output text in chat.
- ONLY Recraft variant with `style` and `text_layout`. V4 / V4 Pro lack both.
- For text rendering: put exact wording in quotes in your prompt AND use `text_layout` for precise placement (V3-exclusive feature).
- Style names: see https://www.recraft.ai/docs/api-reference/styles. Vector styles NOT supported via OpenRouter.
- text_layout: array of {text, bbox} where bbox is 4 [x,y] corners in 0-1 coords (order: TL, TR, BR, BL).
- If you need newer composition / cleaner geometry → V4 / V4 Pro (but lose text_layout + style).

## Controls
- **Aspect ratio** — Frame shape. Choices: 1:1, 4:3, 3:4, 16:9, 9:16, auto.
- **Number of images** — How many images this request asks for. Accepts 1 to 6.
- **style** — a setting this model's provider accepts.
- **controls** — a setting this model's provider accepts.
- **text_layout** — a setting this model's provider accepts.
```

The `## Controls` section is read from the model's own published
settings, so it lists that model's choices and no others. A model that
publishes none says so rather than showing an empty section.

If a model isn't in the curated dataset (newly added by OpenRouter
between catalog refreshes, for example), `help` falls back to the
catalog metadata — display name, description, output/input modalities.

---

## Output rendering and message format

Multimodal image responses follow the chat-completion image rendering
pipeline that has always handled `gpt-5-image` and similar models. Models
that emit only images take the dedicated image adapter instead, described
above; both end at the same persisted file URL and the same markdown. The
chat pipeline:

1. **OpenRouter response** comes back with `message.images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]`.
2. [`api/gateway/chat_completions_adapter.py`](../open_webui_openrouter_pipe/api/gateway/chat_completions_adapter.py)
   parses `message.images` and emits an `image_generation_call` item.
3. [`streaming/streaming_core.py`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
   `_materialize_image_entry()` recursively resolves dicts (`url`,
   `image_url`, `imageUrl`, `content_url`), decodes base64 fields
   (`b64_json`, `b64`, `base64`, `data`, `image_base64`, `imageB64`),
   validates size against `BASE64_MAX_SIZE_MB`, persists via
   `_persist_generated_image`, returns `/api/v1/files/{stored}/content`.
4. [`streaming/streaming_core.py`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
   `_collect_image_output_urls()` resolves entries to a list of file
   URLs.
5. [`streaming/streaming_core.py`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
   `_render_image_markdown()` produces `![alt](url)` markdown that
   OWUI renders inline.
6. The renderer at [`streaming/streaming_core.py`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
   emits status, dedupes, and handles the final write.

The rendered message looks like:

```markdown
![Generated image](/api/v1/files/01HX2K3D5N4P9F8GZQ2WV3R5BC/content)
```

OWUI displays the image inline with a download/copy/view-fullsize
context menu. The file is registered in OWUI's `Files` table linked to
the chat, surviving page reload.

---

## Pricing and cost display

Pricing is pulled live from the OpenRouter catalog via the standard
chat catalog refresh path (`MODEL_CATALOG_REFRESH_SECONDS` TTL). The
status footer rendered on the assistant message includes the cost of
the generation in dollars, derived from `prompt_tokens` /
`completion_tokens` × the model's per-token rates.

For multimodal models (GPT-5 Image, Gemini Image), image output counts
as completion tokens — the cost is bundled. For pure-image-only models
(FLUX, Sourceful, Seedream), token-based pricing applies via OpenRouter's
standard usage accounting.

`super_resolution_references` adds +$0.20 per reference. These appear
in the cost breakdown if you use those features.

---

## Configuration valves (admin)

Four valves control the native image-generation subsystem. All are
visible in Admin → Functions → OpenRouter pipe → Valves. Catalog TTL
is shared with chat/video catalogs (`MODEL_CATALOG_REFRESH_SECONDS`).

| Valve | Default | Range | Purpose |
|-------|---------|-------|---------|
| `ENABLE_OPENROUTER_IMAGE_GENERATION` | `True` | bool | Master kill switch. False removes pure-image-only models from `pipes()` output AND clears them from OWUI's catalog (`register_image_models([])` runs once on the next cycle). Multimodal models stay since they're in the chat catalog. |
| `AUTO_INSTALL_IMAGE_FILTERS` | `True` | bool | Install and keep current one settings panel per image model, offering exactly what that model publishes. A model whose settings list has never been read gets none; one read before keeps its last successful set. |
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
image-output and video-generation models from Web Tools attach via
the `web_tools_supported` check in
[`models/catalog_manager.py`](../open_webui_openrouter_pipe/models/catalog_manager.py):

```python
web_tools_supported = bool(
    web_tools_filter_function_id
    and (valves.AUTO_ATTACH_WEB_TOOLS_FILTER or valves.AUTO_DEFAULT_WEB_TOOLS_FILTER)
    and not pipe_capabilities.get("image_output")
    and not pipe_capabilities.get("video_generation")
)
```

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
have honoured it. Type `help` to the model to see its full list.

If a model shows **no** settings at all, its list has never been read
successfully — the next refresh retries. It still generates images
meanwhile, using its own defaults.

### A value I typed was rejected as invalid JSON

Settings that take a list or an object — a provider's own options,
usually — are typed as JSON. If what you typed starts with `[` or `{`
it has to be valid JSON, and the error names the setting it came from.
Plain words are sent as they are and do not need quoting.

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
- Check the pipe logs for `_persist_generated_image` errors.

### Body validation error: `image_config` rejected at `CompletionsBody.model_validate`

If you see a Pydantic validation error mentioning `image_config`, the
type may have regressed. The field is `Optional[Dict[str, Any]]` per
this feature — see [`api/transforms.py`](../open_webui_openrouter_pipe/api/transforms.py).
If anyone changes it back to a scalar type, dict writes from the
filters will fail validation.

---

## Architecture overview

Roughly, in order of who-calls-who:

```
pipes()
  ├─ ensure chat catalog loaded (existing)
  ├─ ensure video catalog loaded (existing)
  └─ if ENABLE_OPENROUTER_IMAGE_GENERATION:
        ensure_image_catalog_loaded()
          ├─ TTL-gated fetch (cache_seconds = MODEL_CATALOG_REFRESH_SECONDS)
          ├─ /api/v1/models?output_modalities=image via OpenRouterImageClient
          ├─ register_image_models()
          │     ├─ skip multimodal (text in output_modalities)
          │     ├─ stale-norm cleanup (drop models removed from catalog)
          │     ├─ atomic publish (4 dict assignments, no await)
          │     └─ features = {"image_output", "image_gen_tool"}
          └─ if disabled: register_image_models([]) + reset_image_fetch_timestamp()

  └─ if AUTO_INSTALL_IMAGE_FILTERS:
        ensure_openrouter_image_filter_function_ids(available_models)
          ├─ one filter per image model, from its published contract
          ├─ a model with no readable contract gets none
          ├─ each install in own try/except — partial failures isolated
          └─ retire filters left over from the fixed-variant design

  └─ catalog_manager._update_or_insert_model_with_metadata()
        ├─ pipe_capabilities.image_output gate
        ├─ web_tools_supported = ... and not image_output
        ├─ _apply_list_filter_ids(meta_dict)       — writes filterIds
        └─ _apply_list_default_filter_ids(meta_dict) — writes defaultFilterIds

pipe(body, ...)
  └─ orchestrator._inject_image_modalities(body)
        ├─ no-op if model not in registry or no image in output_modalities
        ├─ pure-image: body["modalities"] = ["image"]
        └─ multimodal: body["modalities"] = ["image", "text"]

  └─ filter inlet (run by OWUI before pipe receives body)
        ├─ model gate: every id form OWUI produces, and no other model
        ├─ merge the chosen values into body.image_config, per key
        └─ JSON-typed values parsed only when they open a container

  └─ multimodal model (emits image and text)
  │     └─ chat-completions request → response with message.images[0]
  │     └─ chat_completions_adapter parses message.images
  │     └─ streaming_core materialises the entry → persists → file URL
  │     └─ streaming_core renders "![alt](file_url)"
  └─ image-only model
        └─ dedicated image request → response with inline base64
        └─ image adapter validates knobs against the model's endpoint record
        └─ image adapter persists each image → file URL
        └─ image adapter renders "![alt](file_url)"
  └─ OWUI renders inline image
```

Both branches emit the same `![alt](file_url)` markdown, which is what keeps
iterative editing working: the next request re-parses that markdown back into
an input image.

Key invariant: **both branches render the same markdown**. Multimodal
models keep the streaming path that has always handled them.

Key files:

- [`integrations/image_catalog.py`](../open_webui_openrouter_pipe/integrations/image_catalog.py)
  — TTL-gated catalog fetch + master-disable cleanup.
- [`integrations/image_client.py`](../open_webui_openrouter_pipe/integrations/image_client.py)
  — HTTP client for the image model catalog, the per-model endpoint record
  that publishes which knobs a model accepts, and image generation itself.
- [`integrations/image.py`](../open_webui_openrouter_pipe/integrations/image.py)
  — the adapter for image-only models: gates each requested knob against the
  model's published contract, reports the ones it withheld, persists the
  returned images and renders the markdown.
- [`integrations/provider_options.py`](../open_webui_openrouter_pipe/integrations/provider_options.py)
  — the single reader of a request's provider block, the per-transport set of
  provider keys OpenRouter documents, and the choice of which provider slug
  carries a value that cannot be duplicated across providers.
- [`integrations/image_help.py`](../open_webui_openrouter_pipe/integrations/image_help.py)
  — `_IMAGE_PER_MODEL_HELP_DATA` (per-model prose), `render_image_help()`
  (control list read from the model's endpoint record).
- [`filters/image_filter_renderer.py`](../open_webui_openrouter_pipe/filters/image_filter_renderer.py)
  — `build_image_model_filter_spec()` turns a model's endpoint record into
  its knob set; `render_image_model_filter_source()` renders one filter
  module from that spec.
- [`filters/filter_manager.py::ensure_openrouter_image_filter_function_ids`](../open_webui_openrouter_pipe/filters/filter_manager.py)
  — installs filter rows in OWUI Functions table; returns
  per-model filter id mapping.
- [`models/catalog_manager.py`](../open_webui_openrouter_pipe/models/catalog_manager.py)
  — `_apply_list_filter_ids`, `_apply_list_default_filter_ids`,
  `pipe_capabilities.image_output` gate, capability-gated
  `web_tools_supported` exclusion.
- [`models/registry.py::register_image_models`](../open_webui_openrouter_pipe/models/registry.py)
  — atomic registry merge with stale-norm cleanup; multimodal dedupe.
- [`requests/orchestrator.py::_inject_image_modalities`](../open_webui_openrouter_pipe/requests/orchestrator.py)
  — body modalities injection.
- [`api/transforms.py`](../open_webui_openrouter_pipe/api/transforms.py)
  — Pydantic `image_config: Optional[Dict[str, Any]]` field type fix.
- [`core/config.py`](../open_webui_openrouter_pipe/core/config.py)
  — 4 new valves + filter marker constant.

**Files NOT touched** (pre-existing, reused as-is):

- `chat_completions_adapter.py` — `message.images` parser.
- `streaming/streaming_core.py` — image materialization, file
  persistence, markdown rendering.
- `storage/multimodal.py` — `_persist_generated_image` and friends.
- The legacy `openrouter_image_gen` filter (OpenAI Responses-tool wiring).

---

## Limitations and non-goals

- **Synchronous only.** Image generation is a single chat-completions
  request — no polling lifecycle, no resume, no disconnect recovery.
  If the request fails or the user disconnects, the generation is lost.
  Re-submit to retry.
- **No streaming intermediate frames.** OpenRouter doesn't stream
  partial images; the response includes the full base64 image at once.
- **No batch generation.** One request, one image (or set of images
  the model emits per turn). For batch use, send multiple chats.
- **Multimodal models may emit text without an image.** GPT-5 Image
  and Gemini Image variants decide based on prompt. Be explicit in
  the prompt ("generate an image of...") if you want guaranteed
  image output.
- **No video output from these models.** Image-output models do not
  generate video. For video, use the
  [video-generation feature](openrouter_video_generation.md).
