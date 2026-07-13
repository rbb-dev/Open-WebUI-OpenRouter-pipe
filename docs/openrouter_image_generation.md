# OpenRouter Image Generation

This pipe exposes OpenRouter's thirty-one native image-output models as
selectable chat models in Open WebUI. You pick an image model in the chat
header (just like any other LLM), type a prompt, and the pipe submits a
chat-completions request with `modalities: ["image"]` (or
`["image", "text"]` for multimodal models), receives the generated image
inline as base64, persists it to OWUI file storage, and renders it inline
with `![alt](url)` markdown. No polling, no separate endpoint — image
generation is synchronous over `/api/v1/chat/completions`.

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
- [Per-model parameter reference](#per-model-parameter-reference)
- [Filter UserValve identifiers (master reference)](#filter-uservalve-identifiers-master-reference)
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
   prompt input). The `OR Image Filter` toggle is auto-attached and
   default-on. Model families with extra parameters also get their own
   filter rows: `Gemini Options` (Gemini Flash 3.x image),
   `Sourceful Options` (Riverflow V2 Pro/Fast), `Sourceful V2.5
   Options` (Riverflow 2.5 Pro/Fast — the single Sourceful filter for
   2.5, carrying fonts plus the 2.5-only knobs), `Recraft Options`
   (all Recraft models), `Recraft V3 Extras` (Recraft V3 only), and
   `Grok Imagine Options` (xAI Grok Imagine image models). All
   applicable filters are default-on.
4. (Optional) Click any filter's settings icon to set per-message
   overrides — aspect ratio, image size, custom fonts (Sourceful), etc.
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
AUTO_INSTALL_IMAGE_FILTERS = True       # creates the 7 filter rows
AUTO_ATTACH_IMAGE_FILTERS  = True       # attaches each filter to its models
AUTO_DEFAULT_IMAGE_FILTERS = True       # filter is on-by-default per chat
```

If the per-model filters do not appear in the Integrations menu, check:

- `AUTO_INSTALL_IMAGE_FILTERS` and `AUTO_ATTACH_IMAGE_FILTERS` are both
  `True`.
- The pipe has been called at least once with a logged-in user (the
  filters install during `pipes()` warmup).
- Open WebUI Admin → Functions lists seven entries:
  `OR Image Filter`, `Gemini Options`, `Sourceful Options`,
  `Sourceful V2.5 Options`, `Recraft Options`, `Recraft V3 Extras`,
  `Grok Imagine Options`.

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

| Model id | Display name | Output | Special knobs | Cost rate |
|----------|--------------|:----:|---------------|-----------|
| `openai/gpt-5-image` | OpenAI: GPT-5 Image | text + image | aspect_ratio, image_size | GPT-5 chat token economics |
| `openai/gpt-5-image-mini` | OpenAI: GPT-5 Image Mini | text + image | aspect_ratio, image_size | Cheaper GPT-5 Image tier |
| `openai/gpt-5.4-image-2` | OpenAI: GPT-5.4 Image 2 | text + image | aspect_ratio, image_size | Updated GPT-5.4 generation |
| `google/gemini-2.5-flash-image` | Google: Gemini 2.5 Flash Image | text + image | aspect_ratio, image_size | Standard Gemini multimodal |
| `google/gemini-3-pro-image-preview` | Google: Gemini 3 Pro Image (Preview) | text + image | aspect_ratio, image_size | Premium Gemini 3 with image |
| `google/gemini-3.1-flash-image-preview` | Google: Gemini 3.1 Flash Image (Preview) | text + image | + extended ratios (1:4/4:1/1:8/8:1), 0.5K size | Cost-optimized; 0.5K is ~50% cheaper than 1K |
| `openrouter/auto` | OpenRouter: Auto (Image Routing) | varies | aspect_ratio, image_size | Auto-routes to best image model |
| `microsoft/mai-image-2.5` | Microsoft: MAI-Image-2.5 | image only | aspect_ratio (7 of the 10 standard ratios), image_size | $5/M tokens via Azure AI Foundry |
| `sourceful/riverflow-v2-pro` | Sourceful: Riverflow V2 Pro | image only | + font_inputs (max 2, +$0.03 each), super_resolution_references (max 4, +$0.20 each — V2 only) | Premium Sourceful tier |
| `sourceful/riverflow-v2-fast` | Sourceful: Riverflow V2 Fast | image only | + font_inputs, super_resolution_references (V2 only) | Faster, cheaper Sourceful |
| `sourceful/riverflow-v2.5-pro` | Sourceful: Riverflow V2.5 Pro | image only | + font_inputs; scoring_prompt, scoring_rubric, background_mode, background_hex_color (2.5 extras). NO super_resolution_references | From $0.13/image (finalized per job) |
| `sourceful/riverflow-v2.5-fast` | Sourceful: Riverflow V2.5 Fast | image only | + font_inputs; scoring_prompt, scoring_rubric, background_mode, background_hex_color (2.5 extras). NO super_resolution_references | From $0.019/image (finalized per job) |
| `sourceful/riverflow-v2-max-preview` | Sourceful: Riverflow V2 Max (Preview) | image only | aspect_ratio, image_size | Preview tier — specs may shift |
| `sourceful/riverflow-v2-standard-preview` | Sourceful: Riverflow V2 Standard (Preview) | image only | aspect_ratio, image_size | Entry-tier Sourceful preview |
| `sourceful/riverflow-v2-fast-preview` | Sourceful: Riverflow V2 Fast (Preview) | image only | aspect_ratio, image_size | Fast preview tier |
| `black-forest-labs/flux.2-pro` | Black Forest Labs: FLUX.2 Pro | image only | aspect_ratio, image_size; seed support | Premium FLUX.2 |
| `black-forest-labs/flux.2-max` | Black Forest Labs: FLUX.2 Max | image only | aspect_ratio, image_size; seed support | Highest FLUX.2 tier |
| `black-forest-labs/flux.2-flex` | Black Forest Labs: FLUX.2 Flex | image only | aspect_ratio, image_size; seed support | Mid-tier FLUX.2 |
| `black-forest-labs/flux.2-klein-4b` | Black Forest Labs: FLUX.2 Klein 4B | image only | aspect_ratio, image_size; seed support | Smallest, cheapest FLUX |
| `bytedance-seed/seedream-4.5` | ByteDance Seed: Seedream 4.5 | image only | aspect_ratio, image_size; temperature, top_p | Image-only with sampling controls |
| `recraft/recraft-v3` | Recraft: Recraft V3 | image only | + strength, rgb_colors, background_rgb_color, **style** (V3 only), **text_layout** (V3 only) | Typography champion; only model with text-at-position |
| `recraft/recraft-v4` | Recraft: Recraft V4 | image only | + strength, rgb_colors, background_rgb_color | Design-taste rebuild; 1024x1024; ~10s/image |
| `recraft/recraft-v4-pro` | Recraft: Recraft V4 Pro | image only | + strength, rgb_colors, background_rgb_color | Print-ready 2048x2048 (~30s/image); $0.25/image |
| `recraft/recraft-v4-vector` | Recraft: Recraft V4 Vector | image only (SVG) | + strength, rgb_colors, background_rgb_color | True SVG output; scales without quality loss |
| `recraft/recraft-v4-pro-vector` | Recraft: Recraft V4 Pro Vector | image only (SVG) | + strength, rgb_colors, background_rgb_color | High-fidelity SVG finals |
| `recraft/recraft-v4.1` | Recraft: Recraft V4.1 | image only | + strength, rgb_colors, background_rgb_color | Aesthetic refresh of V4; 1024x1024; ~10s/image |
| `recraft/recraft-v4.1-pro` | Recraft: Recraft V4.1 Pro | image only | + strength, rgb_colors, background_rgb_color | Print-ready 2048x2048 with V4.1 aesthetics |
| `recraft/recraft-v4.1-vector` | Recraft: Recraft V4.1 Vector | image only (SVG) | + strength, rgb_colors, background_rgb_color | V4.1 aesthetics, SVG output |
| `recraft/recraft-v4.1-pro-vector` | Recraft: Recraft V4.1 Pro Vector | image only (SVG) | + strength, rgb_colors, background_rgb_color | Highest-fidelity SVG finals |
| `recraft/recraft-v4.1-utility` | Recraft: Recraft V4.1 Utility | image only | + strength, rgb_colors, background_rgb_color | General-purpose (non-aesthetic) tier; 1024x1024 |
| `recraft/recraft-v4.1-utility-pro` | Recraft: Recraft V4.1 Utility Pro | image only | + strength, rgb_colors, background_rgb_color | General-purpose at 2048x2048 |
| `x-ai/grok-imagine-image-quality` | xAI: Grok Imagine Image Quality | image only | + Grok aspect_ratio set (14 values incl. 9:19.5/9:20/1:2/auto), n (1-10 images per request) | $0.01/image |

Pick model selection rules of thumb:

- **Long-form text or precise text-at-position in images** → Recraft V3
  (the only model with `text_layout` for explicit placement; renders
  full sentences/paragraphs cleanly).
- **Custom typography (font files) on an image** → Sourceful Riverflow
  V2 or V2.5 Pro/Fast (only models with `font_inputs`).
- **Print-ready high-resolution finals** → Recraft V4/V4.1 Pro
  (2048x2048 with design-taste output) or FLUX.2 Max (4K).
- **Image-to-image super-resolution** → Sourceful Riverflow V2 Pro/Fast
  ONLY (`super_resolution_references` was dropped in Riverflow 2.5).
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

- **Filter attach**: still receives `OR Image Filter` (generic) so
  users can configure aspect ratio and image size.
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

This section is the same content the in-chat `help` command renders, in
written form. Skip to a model that matches your use case, or read them
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
- **Standard knobs:** 10 aspect ratios + 1K/2K/4K via the generic
  filter; defaults to 1:1 1K when unset.
- **Already in chat catalog** — generic image filter auto-attaches to
  expose aspect_ratio/image_size knobs.
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
- Same standard aspect_ratio + image_size knob set; no Sourceful-only
  or Gemini-only extensions.

### OpenAI: GPT-5.4 Image 2

> **id**: `openai/gpt-5.4-image-2` · **multimodal**

Updated GPT-5.4 generation of multimodal text+image output. Improved
prompt adherence and visual fidelity over GPT-5 Image.

- Successor to GPT-5 Image — same modalities + image_config schema,
  improved quality.
- Use for production deliverables that need the latest OpenAI image
  model.
- Same standard knob set; standard 10 aspect ratios + 1K/2K/4K sizes.

### Google: Gemini 2.5 Flash Image

> **id**: `google/gemini-2.5-flash-image` · **multimodal**

Google's standard Gemini multimodal text+image model. Best for
prompt-following tasks with cinematic composition and natural-looking
output. Outputs both text and image.

- Standard 10 aspect ratios + 1K/2K/4K. **No 0.5K or extended ratios on
  this variant** — those are Gemini 3.x Flash Image only.
- Multimodal: model decides emission based on prompt; be explicit.
- Strong at photoreal scenes and prompt-faithful composition.

### Google: Gemini 3 Pro Image (Preview)

> **id**: `google/gemini-3-pro-image-preview` · **multimodal**

Premium tier of Gemini 3 with native image output. Highest fidelity
Gemini image model OpenRouter exposes; best for hero shots and
high-detail outputs.

- Premium variant — higher cost than Flash; reserve for finals.
- Standard 10 aspect ratios + 1K/2K/4K (no 0.5K — that's Flash-only).
- Multimodal text+image output.

### Google: Gemini 3.1 Flash Image (Preview)

> **id**: `google/gemini-3.1-flash-image-preview` · **multimodal + Gemini Options filter**

Cost-optimized Gemini 3.1 with native image output AND unique extended
knobs: 4 extra aspect ratios (1:4, 4:1, 1:8, 8:1) for ultrawide/tall
layouts AND a 0.5K low-res tier for cheap iteration. **Only Gemini
variant with these extensions.**

- **Use the dedicated `Gemini Options` filter** for extended ratios
  (4:1, 1:4, 8:1, 1:8) and 0.5K size.
- Set aspect via the Gemini-extended valve OR the standard valve;
  the Gemini one wins on collision (deep-merge semantics — second
  filter's writes overwrite the first's).
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
- Standard knob set applies; provider-specific knobs (Gemini 0.5K,
  Sourceful font_inputs) likely ignored if not the selected provider.

### Microsoft: MAI-Image-2.5

> **id**: `microsoft/mai-image-2.5` · **pure-image-only, generic filter only**

Microsoft's high-quality image generation model served via Azure AI
Foundry — photorealistic and artistic output from text prompts with
optional reference-image input. Best for general-purpose photoreal
work on Azure-backed infrastructure with token-based pricing ($5/M
tokens) instead of per-image billing.

- Supports 7 aspect ratios (1:1 default, 4:3, 3:4, 16:9, 9:16, 3:2,
  2:3) — all available through the generic aspect ratio knob;
  4:5/5:4/21:9 are NOT supported by this model.
- Token-priced ($5/M) rather than per-image — long prompts cost
  proportionally more.
- Multimodal input: accepts reference images alongside the text prompt
  for editing/guidance.
- No model-specific extensions — the generic filter covers everything
  this model accepts.

### Sourceful: Riverflow V2 Pro

> **id**: `sourceful/riverflow-v2-pro` · **pure-image-only + Sourceful Options filter**

Sourceful's premium tier — pure image-only output with custom font
rendering and image-to-image super-resolution. Strongest for marketing
assets requiring exact text rendering at scale.

- **PURE-image-only** — does NOT output text. Filter writes
  `modalities=["image"]` for this model.
- **`font_inputs`** (max 2, +$0.03/font) renders custom typefaces in
  the image — supply `font_url` + matching text in the prompt.
- **`super_resolution_references`** (max 4, +$0.20/ref) requires input
  images in messages (image-to-image only).
- Both extensions exposed via `Sourceful Options` filter; cardinality
  caps validated at inlet (rejects 3+ font_inputs before submission).
- **4.5MB request size limit** — pass image URLs instead of base64 to
  avoid bloat.

### Sourceful: Riverflow V2 Fast

> **id**: `sourceful/riverflow-v2-fast` · **pure-image-only + Sourceful Options filter**

Faster, cheaper variant of Riverflow V2 — same Sourceful extensions
(`font_inputs`, `super_resolution_references`) at lower quality and
reduced cost. Best for iteration before committing to a Pro render.

- Same caveats as Riverflow V2 Pro: pure-image-only, 4.5MB request
  limit, image URLs preferred.
- Use Fast for prompt iteration and font/reference tuning; switch to
  Pro for finals.
- Same Sourceful-specific knobs (`font_inputs`,
  `super_resolution_references`) via dedicated filter.

### Sourceful: Riverflow V2.5 Pro

> **id**: `sourceful/riverflow-v2.5-pro` · **pure-image-only + Sourceful V2.5 Options filter**

The most powerful variant of Sourceful's Riverflow 2.5 lineup — a
unified text-to-image and image-to-image family. Best for top-tier
control and quality-sensitive outputs: brand assets, marketing finals,
and work that benefits from the new 2.5 self-scoring and background
controls. From $0.13/image (finalized per job at completion).

- PURE-image-only — does NOT output text.
- ONE dedicated filter — Sourceful V2.5 Options — carries every 2.5
  knob: font_inputs (max 2, +$0.03/font, carried over from V2) plus
  scoring_prompt + scoring_rubric (self-scored candidate selection)
  and background_mode (original/transparent/solid) +
  background_hex_color.
- super_resolution_references is NOT supported — Riverflow 2.5 dropped
  it (V2 Pro/Fast only); the knob does not exist on 2.5 models.
- Supports reasoning effort up to xhigh (low/medium/high/xhigh).
- Pricing is dynamic — the from-$0.13/image floor is finalized per job
  based on billable processing.

### Sourceful: Riverflow V2.5 Fast

> **id**: `sourceful/riverflow-v2.5-fast` · **pure-image-only + Sourceful V2.5 Options filter**

The speed-optimized variant of Sourceful's Riverflow 2.5 lineup — best
for production deployments and latency-critical workflows. Same
unified text-to-image and image-to-image family and the same 2.5
extras as Pro at a fraction of the cost. From $0.019/image (finalized
per job at completion).

- PURE-image-only — does NOT output text.
- Use Fast for iteration and high-volume production; switch to V2.5
  Pro for quality-sensitive finals.
- ONE dedicated filter — Sourceful V2.5 Options — with font_inputs
  plus the 2.5 extras (scoring_prompt, scoring_rubric,
  background_mode, background_hex_color).
- super_resolution_references is NOT supported (dropped in 2.5); the
  knob does not exist on 2.5 models.
- Supports reasoning effort low/medium/high (xhigh is Pro-only).

### Sourceful: Riverflow V2 Max (Preview)

> **id**: `sourceful/riverflow-v2-max-preview` · **pure-image-only**

Preview release of the highest-tier Riverflow variant. Higher fidelity
than Pro but preview status means specs may shift. Pure-image-only
output.

- Preview — quality and pricing may change without notice.
- **Does NOT support `font_inputs` / `super_resolution_references`** —
  those are Pro/Fast only on the v2 line.
- Standard 10 aspect ratios + 1K/2K/4K via the generic filter.

### Sourceful: Riverflow V2 Standard (Preview)

> **id**: `sourceful/riverflow-v2-standard-preview` · **pure-image-only**

Standard preview release of Riverflow V2 — entry-tier quality and
pricing. Pure-image-only.

- Preview status — specs may change.
- No Sourceful-specific extensions on this variant
  (`font_inputs` / `super_resolution_references` are Pro/Fast only).
- Standard knob set applies via generic filter.

### Sourceful: Riverflow V2 Fast (Preview)

> **id**: `sourceful/riverflow-v2-fast-preview` · **pure-image-only**

Preview release of the fastest Riverflow tier. Pure-image-only with
reduced quality versus Pro/Standard at lower cost.

- Preview — pricing/quality may shift.
- No Sourceful-specific extensions (`font_inputs` /
  `super_resolution_references` are Pro/Fast non-preview only).
- Standard knob set via generic filter.

### Black Forest Labs: FLUX.2 Pro

> **id**: `black-forest-labs/flux.2-pro` · **pure-image-only**

Black Forest Labs' premium FLUX.2 model — pure-image-only with strong
photorealism and prompt adherence. Best for high-quality deliverables.
**Supports seed for deterministic generation.**

- PURE-image-only — does NOT output text.
- Seed support enables deterministic regeneration with same prompt +
  seed.
- Standard 10 aspect ratios + 1K/2K/4K via generic filter.
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
- Standard knob set via generic filter.

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
- Standard knob set via generic filter; no model-specific extensions.

### Recraft: Recraft V3

> **id**: `recraft/recraft-v3` · **pure-image-only + Recraft Options + Recraft V3 Extras filters**

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

> **id**: `recraft/recraft-v4` · **pure-image-only + Recraft Options filter**

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

> **id**: `recraft/recraft-v4-pro` · **pure-image-only + Recraft Options filter**

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

> **id**: `recraft/recraft-v4-vector` · **pure-image-only (SVG) + Recraft Options filter**

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

> **id**: `recraft/recraft-v4-pro-vector` · **pure-image-only (SVG) + Recraft Options filter**

High-fidelity SVG counterpart to V4 Pro — ~2K-equivalent detail in
true vector output. Use V4 Vector for iteration; V4 Pro Vector for
final logo/brand deliverables.

- Same SVG caveats as V4 Vector (graphic prompts, undocumented color
  steering, rasterised i2i input).
- Higher fidelity, slower, costlier than V4 Vector — reserve for finals.

### Recraft: Recraft V4.1

> **id**: `recraft/recraft-v4.1` · **pure-image-only + Recraft Options filter**

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

> **id**: `recraft/recraft-v4.1-pro` · **pure-image-only + Recraft Options filter**

High-resolution counterpart to V4.1 — same aesthetic tuning at
2048x2048 (~4 MP), ~30s/image. Built for print-ready aesthetic work:
magazine layouts, posters, billboards, editorial illustration. Use
V4.1 for iteration, V4.1 Pro for finals.

- PURE-image-only; same knob set as V4.1.
- ~3x slower than V4.1 due to higher resolution — reserve for finals.
- Same human-subject limitations as the V4 family.

### Recraft: Recraft V4.1 Vector

> **id**: `recraft/recraft-v4.1-vector` · **pure-image-only (SVG) + Recraft Options filter**

Vector (SVG) variant of V4.1 — V4.1's aesthetic tuning with ~1K
equivalent detail and true `<svg>` output. Best for aesthetic-driven
logos, icon sets, and flat illustrations destined for vector editing.
Faster/cheaper than V4.1 Pro Vector for iteration.

- Same SVG caveats as the V4 vector variants.
- Use V4.1 Vector for iteration; V4.1 Pro Vector for finals.

### Recraft: Recraft V4.1 Pro Vector

> **id**: `recraft/recraft-v4.1-pro-vector` · **pure-image-only (SVG) + Recraft Options filter**

The highest-fidelity vector variant — V4.1 aesthetics, ~2K equivalent
detail, true SVG. Best for high-polish logos, editorial icon sets, and
brand assets that must scale and edit downstream.

- Same SVG caveats as the other vector variants.
- Try V4.1 Pro Vector first for final vector work; it carries the same
  aesthetic advantage over V4 Pro Vector that V4.1 has over V4.

### Recraft: Recraft V4.1 Utility

> **id**: `recraft/recraft-v4.1-utility` · **pure-image-only + Recraft Options filter**

Recraft's general-purpose V4.1 variant — drops the aesthetic-tuning
bias in exchange for broader subject coverage and cheaper generation.
Best for spot illustrations, diagrams, placeholder/stock imagery, and
any work where "on-brand aesthetics" is not the goal. 1024x1024.

- Pick Utility over regular V4.1 when you need versatility, not polish.
- Same knob set as V4.1; NO style or text_layout (V3 ONLY).
- Switch to regular V4.1 (aesthetic) or V4.1 Pro (print) when output
  quality matters.

### Recraft: Recraft V4.1 Utility Pro

> **id**: `recraft/recraft-v4.1-utility-pro` · **pure-image-only + Recraft Options filter**

High-resolution counterpart to V4.1 Utility — 2048x2048 (~4 MP)
general-purpose raster output. Use for general-purpose finals where
aesthetic polish is not the goal; otherwise prefer V4.1 Pro.

- ~3x slower than V4.1 Utility due to higher resolution.
- Same knob set and limitations as the rest of the V4.1 family.

### xAI: Grok Imagine Image Quality

> **id**: `x-ai/grok-imagine-image-quality` · **pure-image-only + Grok Imagine Options filter**

xAI's fast, high-fidelity image generation and editing model. Accepts
text prompts and optional reference images; produces photorealistic
outputs at 1K or 2K. Best for photoreal scenes, compositional control,
and workflows that need Grok-only tall phone-screen aspect ratios
(9:19.5, 9:20, 1:2, 2:1) or an `auto` ratio that lets the model pick
frame shape from the prompt.

- Use the Grok aspect ratio knob (14 values, including phone-tall and
  `auto`) for Grok-specific frames; the generic knob's 10-value set
  still works but won't expose the wide/tall variants.
- `n` fans out 1-10 variations per request — cost scales linearly.
  Pick n=1 (default) for iteration; bump to 3-5 for exploration.
- Multimodal input: pair the prompt with reference images for
  editing/style transfer.
- Charged per image output ($0.01/image at OpenRouter's listed rate).

---

## Per-model parameter reference

This section enumerates exactly which filter knobs each model exposes.

### Standard knobs (all image models — generic filter)

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Image aspect ratio | Literal | `""`, `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` | `""` = model default. |
| Image size | Literal | `""`, `1K`, `2K`, `4K` | `""` = model default (1K). |

### Gemini Flash 3.x image only (Gemini Options filter)

Adds 4 extra aspect ratios + a 0.5K size tier. Only attached to models
matching `^google/gemini-3.*flash-image.*$`.

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Image aspect ratio (Gemini extended) | Literal | `""`, `1:4`, `4:1`, `1:8`, `8:1` | Ultrawide/tall layouts. Overrides generic aspect_ratio when set. |
| Image size (Gemini-only 0.5K) | Literal | `""`, `0.5K` | Low-res tier (~50% cheaper than 1K). Overrides generic image_size when set. |

### Riverflow V2 Pro/Fast only (Sourceful Options filter)

Adds custom font rendering + image-to-image super-resolution. Only
attached to models matching `^sourceful/riverflow-v2-(pro|fast)$`
(Riverflow 2.5 has its own dedicated filter below).

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Font inputs (JSON array) | str (JSON) | `[{"font_url": "...", "text": "..."}]` | Max 2, +$0.03 each. Validated at inlet (rejects 3+). V2 and 2.5. |
| Super-resolution references (JSON array) | str (JSON) | `["url1", "url2", ...]` | Max 4, +$0.20 each. Image-to-image only (requires input images). Validated at inlet. V2-only parameter (2.5 dropped it). |

**Pre-validation:** the Sourceful filter rejects oversized arrays
**before** the HTTP call, surfacing a clear `ImageGenerationError`
instead of an opaque 400 from the provider.

### Riverflow 2.5 Pro/Fast only (Sourceful V2.5 Options filter)

The single Sourceful filter for 2.5 models — fonts carried over from
V2 plus the params 2.5 introduced. Only attached to models matching
`^sourceful/riverflow-v2\.5-(pro|fast)$`.

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Font inputs (JSON array) | str (JSON) | `[{"font_url": "...", "text": "..."}]` | Max 2, +$0.03 each. Validated at inlet (rejects 3+). Same knob as V2. |
| Scoring prompt | str | free text | Instruction the model uses to self-score candidates before returning the best one. |
| Scoring rubric | str | free text | Rubric describing what a good output looks like; pairs with the scoring prompt. |
| Background mode | Literal | `""`, `original`, `transparent`, `solid` | `transparent` removes the background (PNG alpha); `solid` fills with the hex color below. |
| Background hex color | str | `#RGB` / `#RRGGBB` | Requires Background mode `solid`. Validated at inlet (format + mode pairing). |

### Grok Imagine image models only (Grok Imagine Options filter)

Adds the Grok-specific 14-value aspect-ratio set and multi-image
count. Only attached to models matching `^x-ai/grok-imagine-image-`.

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Image aspect ratio (Grok Imagine) | Literal | `""`, 1:1, 3:4, 4:3, 9:16, 16:9, 2:3, 3:2, 9:19.5, 19.5:9, 9:20, 20:9, 1:2, 2:1, auto | Overrides the generic aspect_ratio when set. |
| Number of images (1-10) | int | `0`–`10` | `0` = skip (model default 1). Cost scales linearly per image. |

### Recraft (all variants — Recraft Options filter)

Adds image-to-image deviation control + color palette steering. Only
attached to models matching `^recraft/recraft-`.

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Strength (image-to-image) | float | `0.0` – `1.0` | `0.0` = skip / use model default (`0.5`). Lower = closer to input image; higher = more creative deviation. Image-to-image only. |
| RGB color palette (JSON array) | str (JSON) | `[[r,g,b], ...]` each 0-255 | Hints output palette. Multiple entries allowed. Validated at inlet (rejects malformed JSON, oversaturated components, wrong arity). |
| Background RGB color (JSON array) | str (JSON) | single `[r,g,b]` each 0-255 | Forces a specific background color. Combinable with `rgb_colors`. Validated at inlet. |

### Recraft V3 only (Recraft V3 Extras filter)

Adds artistic style presets + precise text placement at exact
positions. Only attached to `recraft/recraft-v3` exactly. V4 and V4 Pro
do NOT support these per OpenRouter docs; the filter no-ops if
manually attached to V4/V4 Pro.

| Knob | Type | Values | Notes |
|------|------|--------|-------|
| Recraft style | str | e.g. `"Photorealism"` | Artistic style preset name. See [Recraft styles list](https://www.recraft.ai/docs/api-reference/styles). Vector styles NOT supported via OpenRouter. Empty = no style override. |
| Text layout (JSON array) | str (JSON) | `[{text: str, bbox: [[x,y]×4]}]` | Place text strings at exact positions. `bbox` is 4 corner points in normalized 0.0–1.0 coords (order TL, TR, BR, BL). Validated at inlet (rejects malformed JSON, out-of-range bbox, wrong arity). |

**Pre-validation:** the Recraft filters reject malformed input (bad
JSON, RGB out of 0-255, bbox out of 0.0-1.0) **before** the HTTP call.

---

## Filter UserValve identifiers (master reference)

The per-model parameter tables above use friendly UI labels ("Image
aspect ratio"). Internally each maps to a Pydantic `UserValves`
field with an `IMAGE_*` identifier rendered into the filter source.

`Type` column reads as Pydantic field type. `Default` is the value
treated as "leave model default" (skipped from the request).

### Generic filter (always attached)

| Identifier | Type | Default | Maps to body field |
|------------|------|---------|---------------------|
| `IMAGE_ASPECT_RATIO` | `Literal["", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]` | `""` | `image_config.aspect_ratio` |
| `IMAGE_SIZE` | `Literal["", "1K", "2K", "4K"]` | `""` | `image_config.image_size` |

### Gemini Options filter (Gemini Flash 3.x image only)

| Identifier | Type | Default | Maps to body field |
|------------|------|---------|---------------------|
| `IMAGE_ASPECT_RATIO_EXTENDED` | `Literal["", "1:4", "4:1", "1:8", "8:1"]` | `""` | `image_config.aspect_ratio` (overrides generic) |
| `IMAGE_SIZE_GEMINI` | `Literal["", "0.5K"]` | `""` | `image_config.image_size` (overrides generic) |

### Sourceful Options filter (Riverflow V2 Pro/Fast only)

Riverflow 2.5 models do NOT get this filter — they use the dedicated
`Sourceful V2.5 Options` filter below (one Sourceful filter per
version, never two).

| Identifier | Type | Default | Maps to body field |
|------------|------|---------|---------------------|
| `IMAGE_FONT_INPUTS_JSON` | `str` (JSON array) | `""` | `image_config.font_inputs` |
| `IMAGE_SUPER_RESOLUTION_REFERENCES_JSON` | `str` (JSON array) | `""` | `image_config.super_resolution_references` (V2-only parameter — Riverflow 2.5 dropped it) |

### Sourceful V2.5 Options filter (Riverflow 2.5 Pro/Fast only)

The single Sourceful filter for 2.5 models: fonts carried over from
V2 plus everything 2.5 added. Intentionally no super-resolution knob.

| Identifier | Type | Default | Maps to body field |
|------------|------|---------|---------------------|
| `IMAGE_FONT_INPUTS_JSON` | `str` (JSON array) | `""` | `image_config.font_inputs` |
| `IMAGE_SCORING_PROMPT` | `str` | `""` | `image_config.scoring_prompt` |
| `IMAGE_SCORING_RUBRIC` | `str` | `""` | `image_config.scoring_rubric` |
| `IMAGE_BACKGROUND_MODE` | `Literal["", "original", "transparent", "solid"]` | `""` | `image_config.background_mode` |
| `IMAGE_BACKGROUND_HEX_COLOR` | `str` (`#RGB` or `#RRGGBB`) | `""` | `image_config.background_hex_color` (requires `IMAGE_BACKGROUND_MODE="solid"`; validated at inlet) |

### Grok Imagine Options filter (xAI Grok Imagine image models only)

| Identifier | Type | Default | Maps to body field |
|------------|------|---------|---------------------|
| `IMAGE_GROK_ASPECT_RATIO` | `Literal["", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "9:19.5", "19.5:9", "9:20", "20:9", "1:2", "2:1", "auto"]` | `""` | `image_config.aspect_ratio` (overrides generic) |
| `IMAGE_GROK_N` | `int` (`ge=0, le=10`) | `0` (skip sentinel) | `image_config.n` |

### Recraft Options filter (all Recraft models)

| Identifier | Type | Default | Maps to body field |
|------------|------|---------|---------------------|
| `IMAGE_STRENGTH` | `float` (`ge=0.0, le=1.0`) | `0.0` (skip sentinel) | `image_config.strength` |
| `IMAGE_RGB_COLORS_JSON` | `str` (JSON array) | `""` | `image_config.rgb_colors` |
| `IMAGE_BACKGROUND_RGB_JSON` | `str` (JSON array) | `""` | `image_config.background_rgb_color` |

### Recraft V3 Extras filter (Recraft V3 only)

| Identifier | Type | Default | Maps to body field |
|------------|------|---------|---------------------|
| `IMAGE_RECRAFT_STYLE` | `str` | `""` | `image_config.style` |
| `IMAGE_TEXT_LAYOUT_JSON` | `str` (JSON array) | `""` | `image_config.text_layout` |

### Conventions for "skip when default"

A valve set to its **default value** is **NOT** included in the
request body — the upstream provider's own default applies. This
matters because providers accept different defaults for the same
parameter, and forcing a value overrides them. Specifically:

- `Literal` and `str` valves: empty string `""` (after `.strip()`) → skipped.
- `float` valves (e.g. `IMAGE_STRENGTH`): `0.0` is the skip sentinel.
  If a user actually wants strength `0.0`, they can set `0.001` (visually
  identical effect; same convention as `VIDEO_SEED`).
- Non-default value → written to `body.image_config[<key>]`.

### Routing (how filters interact)

All image filters write to the same `body.image_config` dict via
**shallow merge**. If the generic filter sets `aspect_ratio=16:9` and
the Gemini Options filter sets `aspect_ratio=4:1`, the second filter's
write wins (per-key overwrite). This is intentional: the more-specific
filter overrides the generic.

**Model gates on extended filters:** the Gemini Options, Sourceful
Options, Sourceful V2.5 Options, Recraft Options, Recraft V3 Extras,
and Grok Imagine Options filters all check `body.model` against their
respective regex patterns at inlet time. If the filter is manually
attached to a non-matching model, the inlet returns the body unchanged
(defensive — protects against operator misconfiguration).

**One Sourceful filter per Riverflow version:** the Sourceful Options
filter attaches to V2 Pro/Fast only (`^sourceful/riverflow-v2-(pro|fast)$`)
and the Sourceful V2.5 Options filter to 2.5 Pro/Fast only — never
both on one model. 2.5 dropped `super_resolution_references` and added
the scoring/background params, so a shared filter would expose dead
knobs; instead each version's filter carries exactly the knobs its
models accept (fonts appear in both).

**Recraft V3 Extras silent no-op on V4/V4 Pro:** per OpenRouter docs
V4 and V4 Pro do NOT support `style` or `text_layout`. If a user has
the Recraft V3 Extras filter set with a style/text_layout value but
switches to V4 mid-session, the filter silently drops those params on
the V4 turn rather than erroring (since the user might switch back to
V3 later).

---

## The chat filter UI (UserValves)

Each filter's UserValves are visible to end users as form fields under
the per-filter settings icon in the Integrations menu. The display
names are intentionally short to fit OWUI's narrow UI:

| Filter ID | OWUI display name | UserValves shown |
|-----------|-------------------|------------------|
| `openrouter_image_filter_generic` | `OR Image Filter` | Image aspect ratio, Image size |
| `openrouter_image_filter_gemini` | `Gemini Options` | Image aspect ratio (Gemini extended), Image size (Gemini-only 0.5K) |
| `openrouter_image_filter_sourceful` | `Sourceful Options` | Font inputs (JSON array), Super-resolution references (JSON array) |
| `openrouter_image_filter_recraft` | `Recraft Options` | Strength (image-to-image), RGB color palette (JSON array), Background RGB color (JSON array) |
| `openrouter_image_filter_recraft_v3` | `Recraft V3 Extras` | Recraft style, Text layout (JSON array) |
| `openrouter_image_filter_sourceful_v25` | `Sourceful V2.5 Options` | Font inputs (JSON array), Scoring prompt, Scoring rubric, Background mode, Background hex color |
| `openrouter_image_filter_grok` | `Grok Imagine Options` | Image aspect ratio (Grok Imagine), Number of images (1-10) |

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

Typing the literal word `help` (no other text — case-sensitive,
exactly four characters) in a chat against any image model returns a
curated help blurb for that specific model. The renderer is
[`integrations/image_help.py::render_image_help()`](../open_webui_openrouter_pipe/integrations/image_help.py).

Help content is composed from `_IMAGE_PER_MODEL_HELP_DATA` entries:

```
# OpenAI: GPT-5 Image

OpenAI's flagship multimodal text+image model — generates both text
response AND inline images per turn. Best for chat-style image
generation where you want commentary alongside the visual.

## Tips & pitfalls
- Multimodal output: model decides when to emit images based on
  prompt — be explicit ("Generate an image of...") for reliability.
- Standard 10 aspect ratios + 1K/2K/4K sizes via image_config; defaults
  to 1:1 1K when unset.
- Already in chat catalog — generic image filter auto-attaches to
  expose aspect_ratio/image_size knobs.
- Pricing follows GPT-5 chat token economics; image output is included
  in completion tokens.

## Knobs
- `Image aspect ratio`: Frame shape (10 standard ratios from 1:1 to
  21:9). Empty = model default.
- `Image size`: Resolution tier (1K/2K/4K). Empty = model default (1K).
```

The `## Knobs` section is gated on the model — the Gemini Options
knobs only render for Gemini Flash 3.x image models, the Sourceful
Options knobs only render for Riverflow Pro/Fast.

If a model isn't in the curated dataset (newly added by OpenRouter
between catalog refreshes, for example), `help` falls back to the
catalog metadata — display name, description, output/input modalities.

---

## Output rendering and message format

Image responses follow the existing chat-completion image rendering
pipeline that has handled `gpt-5-image` and similar multimodal models
since well before this feature. **No new adapter, no new streaming
handler, no new storage helper.** The pipeline:

1. **OpenRouter response** comes back with `message.images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]`.
2. [`api/gateway/chat_completions_adapter.py:418-447`](../open_webui_openrouter_pipe/api/gateway/chat_completions_adapter.py)
   parses `message.images` and emits an `image_generation_call` item.
3. [`streaming/streaming_core.py:595-631`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
   `_materialize_image_entry()` recursively resolves dicts (`url`,
   `image_url`, `imageUrl`, `content_url`), decodes base64 fields
   (`b64_json`, `b64`, `base64`, `data`, `image_base64`, `imageB64`),
   validates size against `REMOTE_IMAGE_MAX_SIZE_MB`, persists via
   `_persist_generated_image`, returns `/api/v1/files/{stored}/content`.
4. [`streaming/streaming_core.py:633-661`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
   `_collect_image_output_urls()` resolves entries to a list of file
   URLs.
5. [`streaming/streaming_core.py:663-672`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
   `_render_image_markdown()` produces `![alt](url)` markdown that
   OWUI renders inline.
6. The renderer at [`streaming/streaming_core.py:1656-1692`](../open_webui_openrouter_pipe/streaming/streaming_core.py)
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

**Sourceful upcharges:** `font_inputs` adds +$0.03 per font, and
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
| `AUTO_INSTALL_IMAGE_FILTERS` | `True` | bool | Install the seven filter rows (generic, Gemini, Sourceful, Sourceful V2.5, Recraft, Recraft V3, Grok Imagine) in OWUI Functions table on `pipes()`. |
| `AUTO_ATTACH_IMAGE_FILTERS` | `True` | bool | Attach the appropriate filter combination to each image-output model: generic to all, Gemini to gemini-flash-image-preview models, Sourceful to Riverflow V2 Pro/Fast, Sourceful V2.5 to Riverflow 2.5 Pro/Fast (one Sourceful filter per version), Recraft to all Recraft models, Recraft V3 to recraft-v3 only, Grok Imagine to grok-imagine-image models. |
| `AUTO_DEFAULT_IMAGE_FILTERS` | `True` | bool | Keep attached image filters enabled by default per chat. Re-asserted on every catalog metadata sync. |

Related (existing) valves:

| Valve | Default | Purpose |
|-------|---------|---------|
| `MODEL_CATALOG_REFRESH_SECONDS` | `3600` | TTL governing how often the image catalog is re-fetched from `/api/v1/models?output_modalities=image`. |
| `REMOTE_IMAGE_MAX_SIZE_MB` | (multimodal section) | Cap on decoded image size before file persistence. |

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
- Admin → Functions has an entry named `OR Image Filter`.
- Restart the pipe to force a fresh `pipes()` cycle, or toggle
  `AUTO_ATTACH_IMAGE_FILTERS` off → save → on → save.

### `Sourceful Options` filter attached but knobs ignored on a non-Sourceful model

The Sourceful filters have model gates — `Sourceful Options` only
fires when the body's `model` matches
`^sourceful/riverflow-v2-(pro|fast)$`, and `Sourceful V2.5 Options`
only when it matches `^sourceful/riverflow-v2\.5-(pro|fast)$`; the
inlet otherwise returns the body unchanged. This is defensive behavior
protecting against operator misconfiguration. To use the knobs, you
must be on the matching Riverflow version (Pro or Fast).

Same applies to `Gemini Options` — only fires for
`^google/gemini-3.*flash-image.*$`.

### `font_inputs has 3 entries; max is 2.`

The Sourceful filter's pre-validation rejected the input. The provider
caps `font_inputs` at 2 and `super_resolution_references` at 4. This
error is raised at filter inlet **before** the HTTP call so the user
gets a clear message instead of an opaque 400 from the provider.
Reduce the array size to fit the cap.

### `IMAGE_FONT_INPUTS_JSON is not valid JSON`

The Sourceful filter parses the `IMAGE_FONT_INPUTS_JSON` /
`IMAGE_SUPER_RESOLUTION_REFERENCES_JSON` valve as JSON. If the user
input isn't valid JSON, the filter raises this error at inlet. Fix
the JSON syntax — for example:

```json
[{"font_url": "https://example.com/Inter.woff2", "text": "Hello"}]
```

### Aspect ratio not honored on `openrouter/auto`

Auto-routing means OpenRouter picks the underlying model. Some
providers may not honor all aspect ratios. The router maps to the
closest equivalent. To get exact aspect ratio, pick a specific model.

### Image generation succeeds but no image renders inline

Check:

- The OpenRouter response has `message.images` populated (not an
  empty list).
- `REMOTE_IMAGE_MAX_SIZE_MB` is large enough — if the decoded image
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
          ├─ install generic filter (always, lazily)
          ├─ install Gemini Options filter (if any Gemini Flash 3.x image model)
          ├─ install Sourceful Options filter (if any Sourceful Pro/Fast model)
          └─ each install in own try/except — partial failures isolated

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

  └─ filter inlets (run by OWUI before pipe receives body)
        ├─ generic: shallow-merge {aspect_ratio, image_size} into body.image_config
        ├─ Gemini Options: model gate; merge extended ratios + 0.5K
        └─ Sourceful Options: model gate; merge font_inputs + super_resolution_references
              └─ pre-validate cardinality and JSON shape; raise ImageGenerationError on fail

  └─ chat-completions request to OpenRouter → response with message.images[0]
  └─ chat_completions_adapter parses message.images
  └─ streaming_core._materialize_image_entry → _persist_generated_image → file URL
  └─ streaming_core._render_image_markdown → "![alt](file_url)"
  └─ OWUI renders inline image
```

Key invariant: **the existing image rendering pipeline is unmodified
by this feature**. Pure-image-only models work end-to-end through the
same `_materialize_image_entry` → `_persist_generated_image` →
`_render_image_markdown` path that has handled `gpt-5-image` since
well before this PR.

Key files:

- [`integrations/image_catalog.py`](../open_webui_openrouter_pipe/integrations/image_catalog.py)
  — TTL-gated catalog fetch + master-disable cleanup.
- [`integrations/image_client.py`](../open_webui_openrouter_pipe/integrations/image_client.py)
  — HTTP client for `/api/v1/models?output_modalities=image`.
- [`integrations/image_help.py`](../open_webui_openrouter_pipe/integrations/image_help.py)
  — `_IMAGE_PER_MODEL_HELP_DATA` (32 entries), `_IMAGE_KNOB_GATE`,
  `render_image_help()`.
- [`filters/image_filter_renderer.py`](../open_webui_openrouter_pipe/filters/image_filter_renderer.py)
  — generates filter source code for the seven variants
  (generic, Gemini Options, Sourceful Options, Sourceful V2.5 Options,
  Recraft Options, Recraft V3 Extras, Grok Imagine Options).
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

- `chat_completions_adapter.py:418-447` — `message.images` parser.
- `streaming/streaming_core.py:595-672` — image materialization, file
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
- **No image-to-image except via Sourceful `super_resolution_references`.**
  Standard chat-completion image generation is text-to-image. To use
  reference images, attach them as standard chat input — the pipeline
  passes them through to the model as user content.
- **No batch generation.** One request, one image (or set of images
  the model emits per turn). For batch use, send multiple chats.
- **Multimodal models may emit text without an image.** GPT-5 Image
  and Gemini Image variants decide based on prompt. Be explicit in
  the prompt ("generate an image of...") if you want guaranteed
  image output.
- **Auto-router (`openrouter/auto`) doesn't honor all knobs.**
  Provider-specific extensions (Sourceful `font_inputs`, Gemini 0.5K)
  are likely ignored if not the selected provider. Use specific
  models for guaranteed knob fidelity.
- **No video output from these models.** Image-output models do not
  generate video. For video, use the
  [video-generation feature](openrouter_video_generation.md).
