"""Per-model help text for OpenRouter image-output models.

Mirror of `video_help.py` shape — `_IMAGE_PER_MODEL_HELP_DATA` keyed by canonical
model id, with `_IMAGE_KNOB_GATE` resolving knob labels against a model's catalog
metadata. `render_image_help(model_id, image_model)` returns the curated help
when an entry exists, or falls back to a generic catalog dump.

**Wiring status (intentional)**: `render_image_help` is exported for future use
but is NOT currently invoked from any production code path. Image generation
flows through the standard chat-completions request — there is no dedicated
adapter (unlike video, where `VideoGenerationAdapter` intercepts the "help"
prompt at `integrations/video.py:62`). A future feature can call
`render_image_help(model_id, image_model_dict)` from a help command, model-info
popover, or orchestrator-level prompt interception. The function is unit-tested
to lock in the contract for that future caller.
"""

from __future__ import annotations

from typing import Any

_IMAGE_PER_MODEL_HELP_DATA: dict[str, dict[str, Any]] = {
    "openai/gpt-5-image": {
        "display_name": "OpenAI: GPT-5 Image",
        "best_known_for": (
            "OpenAI's flagship multimodal text+image model — generates both "
            "text response AND inline images per turn. Best for chat-style "
            "image generation where you want commentary alongside the visual."
        ),
        "tips_and_pitfalls": [
            "Multimodal output: model decides when to emit images based on prompt — be explicit (\"Generate an image of...\") for reliability.",
            "Standard 10 aspect ratios + 1K/2K/4K sizes via image_config; defaults to 1:1 1K when unset.",
            "Already in chat catalog — generic image filter auto-attaches to expose aspect_ratio/image_size knobs.",
            "Pricing follows GPT-5 chat token economics; image output is included in completion tokens.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios from 1:1 to 21:9). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default (1K).",
        },
    },
    "openai/gpt-5-image-mini": {
        "display_name": "OpenAI: GPT-5 Image Mini",
        "best_known_for": (
            "Cost-efficient variant of GPT-5 Image with the same multimodal "
            "text+image output. Best for high-volume image generation, "
            "drafts, and iteration where premium-tier quality isn't required."
        ),
        "tips_and_pitfalls": [
            "Same prompting style as GPT-5 Image — be explicit about wanting images in the prompt.",
            "Lower cost-per-token than GPT-5 Image; ideal for prototyping and bulk runs.",
            "Same standard aspect_ratio + image_size knob set; no Sourceful-only or Gemini-only extensions.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "openai/gpt-5.4-image-2": {
        "display_name": "OpenAI: GPT-5.4 Image 2",
        "best_known_for": (
            "Updated GPT-5.4 generation of multimodal text+image output. "
            "Improved prompt adherence and visual fidelity over GPT-5 Image."
        ),
        "tips_and_pitfalls": [
            "Successor to GPT-5 Image — same modalities + image_config schema, improved quality.",
            "Use for production deliverables that need the latest OpenAI image model.",
            "Same standard knob set; standard 10 aspect ratios + 1K/2K/4K sizes.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "google/gemini-2.5-flash-image": {
        "display_name": "Google: Gemini 2.5 Flash Image",
        "best_known_for": (
            "Google's standard Gemini multimodal text+image model. Best for "
            "prompt-following tasks with cinematic composition and natural-"
            "looking output. Outputs both text and image."
        ),
        "tips_and_pitfalls": [
            "Standard 10 aspect ratios + 1K/2K/4K. No 0.5K or extended ratios on this variant — those are Gemini 3.x Flash Image only (GA + preview).",
            "Multimodal: model decides emission based on prompt; be explicit.",
            "Strong at photoreal scenes and prompt-faithful composition.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "google/gemini-3-pro-image-preview": {
        "display_name": "Google: Gemini 3 Pro Image (Preview)",
        "best_known_for": (
            "Premium tier of Gemini 3 with native image output. Highest "
            "fidelity Gemini image model OpenRouter exposes; best for hero "
            "shots and high-detail outputs."
        ),
        "tips_and_pitfalls": [
            "Premium variant — higher cost than Flash; reserve for finals.",
            "Standard 10 aspect ratios + 1K/2K/4K (no 0.5K — that's Flash-only).",
            "Multimodal text+image output.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "google/gemini-3.1-flash-image-preview": {
        "display_name": "Google: Gemini 3.1 Flash Image (Preview)",
        "best_known_for": (
            "Cost-optimized Gemini 3.1 with native image output AND unique "
            "extended knobs: 4 extra aspect ratios (1:4, 4:1, 1:8, 8:1) for "
            "ultrawide/tall layouts AND a 0.5K low-res tier for cheap "
            "iteration. The Gemini 3.x Flash Image line (GA + preview) has these; Pro and 2.5 do not."
        ),
        "tips_and_pitfalls": [
            "Use the dedicated 'Gemini Options' filter for extended ratios (4:1, 1:4, 8:1, 1:8) and 0.5K size — these are Gemini 3.x Flash Image ONLY (GA + preview; not Pro or 2.5).",
            "Set aspect via the Gemini-extended valve OR the standard valve; the Gemini one wins on collision (deep-merge semantics).",
            "0.5K is ~50% cheaper than 1K — good for prompt iteration.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Image aspect ratio (Gemini extended)": "Gemini-only extended ratios for ultrawide/tall layouts (1:4, 4:1, 1:8, 8:1).",
            "Image size (Gemini-only 0.5K)": "Low-resolution tier for cheap iteration. Gemini Flash Image only.",
        },
    },
    "openrouter/auto": {
        "display_name": "OpenRouter: Auto (Image Routing)",
        "best_known_for": (
            "OpenRouter's automatic routing for image generation. Routes to "
            "the best available image model based on prompt. Useful when you "
            "want OpenRouter to pick rather than committing to a specific "
            "provider."
        ),
        "tips_and_pitfalls": [
            "Auto-routing — exact model used varies; check the response metadata for routed model id.",
            "Universal input modalities (text + image + audio + file + video) — flexible request shape.",
            "Standard knob set applies; provider-specific knobs (Gemini 0.5K, Sourceful font_inputs) likely ignored if not the selected provider.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape. Auto-router may not honor all values for all routed providers.",
            "Image size": "Resolution tier. Auto-router may map to closest equivalent.",
        },
    },
    "microsoft/mai-image-2.5": {
        "display_name": "Microsoft: MAI-Image-2.5",
        "best_known_for": (
            "Microsoft's high-quality image generation model served via Azure "
            "AI Foundry — photorealistic and artistic output from text prompts "
            "with optional reference-image input. Best for general-purpose "
            "photoreal work on Azure-backed infrastructure with token-based "
            "pricing ($5/M tokens) instead of per-image billing."
        ),
        "tips_and_pitfalls": [
            "Supports 7 aspect ratios (1:1 default, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3) — all available through the generic aspect ratio knob; 4:5/5:4/21:9 are NOT supported by this model.",
            "Token-priced ($5/M) rather than per-image — long prompts cost proportionally more.",
            "Multimodal input: accepts reference images alongside the text prompt for editing/guidance.",
            "No model-specific extensions — the generic filter covers everything this model accepts.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape — this model supports 7 of the 10 standard ratios (1:1, 4:3, 3:4, 16:9, 9:16, 3:2, 2:3). Empty = 1:1 default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default (1024px tier).",
        },
    },
    "sourceful/riverflow-v2-pro": {
        "display_name": "Sourceful: Riverflow V2 Pro",
        "best_known_for": (
            "Sourceful's premium tier — pure image-only output with custom "
            "font rendering and image-to-image super-resolution. Strongest "
            "for marketing assets requiring exact text rendering at scale."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text. Filter writes modalities=['image'] for this model.",
            "font_inputs (max 2, +$0.03/font) renders custom typefaces in the image — supply font_url + matching text in the prompt.",
            "super_resolution_references (max 4, +$0.20/ref) requires input images in messages (image-to-image only). V2 Pro/Fast only — Riverflow 2.5 dropped this parameter.",
            "Both extensions exposed via 'Sourceful Options' filter; cardinality caps validated at inlet (rejects 3+ font_inputs before submission).",
            "4.5MB request size limit — pass image URLs instead of base64 to avoid bloat.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Font inputs (JSON array)": "JSON array of {font_url, text} objects for custom typeface rendering. Max 2, +$0.03 each.",
            "Super-resolution references (JSON array)": "JSON array of URL strings for image-to-image upscaling. Max 4, +$0.20 each. Image-to-image only. V2 Pro/Fast only.",
        },
    },
    "sourceful/riverflow-v2-fast": {
        "display_name": "Sourceful: Riverflow V2 Fast",
        "best_known_for": (
            "Faster, cheaper variant of Riverflow V2 — same Sourceful "
            "extensions (font_inputs, super_resolution_references) at lower "
            "quality and reduced cost. Best for iteration before committing "
            "to a Pro render."
        ),
        "tips_and_pitfalls": [
            "Same caveats as Riverflow V2 Pro: pure-image-only, 4.5MB request limit, image URLs preferred.",
            "Use Fast for prompt iteration and font/reference tuning; switch to Pro for finals.",
            "Same Sourceful-specific knobs (font_inputs, super_resolution_references) via dedicated filter. super_resolution_references is V2 Pro/Fast only — dropped in Riverflow 2.5.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Font inputs (JSON array)": "JSON array of {font_url, text} objects. Max 2, +$0.03 each.",
            "Super-resolution references (JSON array)": "JSON array of URL strings for image-to-image upscaling. Max 4, +$0.20 each. Image-to-image only. V2 Pro/Fast only.",
        },
    },
    "sourceful/riverflow-v2.5-pro": {
        "display_name": "Sourceful: Riverflow V2.5 Pro",
        "best_known_for": (
            "The most powerful variant of Sourceful's Riverflow 2.5 lineup — "
            "a unified text-to-image and image-to-image family. Best for "
            "top-tier control and quality-sensitive outputs: brand assets, "
            "marketing finals, and work that benefits from the new 2.5 "
            "self-scoring and background controls. From $0.13/image "
            "(finalized per job at completion)."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "ONE dedicated filter — 'Sourceful V2.5 Options' — carries every 2.5 knob: font_inputs (max 2, +$0.03/font, carried over from V2) plus the new scoring_prompt + scoring_rubric (self-scored candidate selection) and background_mode (original/transparent/solid) + background_hex_color.",
            "super_resolution_references is NOT supported — Riverflow 2.5 dropped it (V2 Pro/Fast only); the knob does not exist on 2.5 models.",
            "Supports reasoning effort up to xhigh (low/medium/high/xhigh) through the standard reasoning controls.",
            "Pricing is dynamic — the quoted from-$0.13/image floor is finalized per job based on billable processing.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Font inputs (JSON array)": "JSON array of {font_url, text} objects for custom typeface rendering. Max 2, +$0.03 each.",
            "Scoring prompt": "Free-text instruction the model uses to self-score candidates before returning the best one.",
            "Scoring rubric": "Free-text rubric describing what a good output looks like; pairs with the scoring prompt.",
            "Background mode": "original keeps the generated background, transparent removes it (PNG alpha), solid fills with the hex color.",
            "Background hex color": "#RGB or #RRGGBB fill color. Requires background mode 'solid'.",
        },
    },
    "sourceful/riverflow-v2.5-fast": {
        "display_name": "Sourceful: Riverflow V2.5 Fast",
        "best_known_for": (
            "The speed-optimized variant of Sourceful's Riverflow 2.5 lineup "
            "— best for production deployments and latency-critical "
            "workflows. Same unified text-to-image and image-to-image family "
            "and the same 2.5 extras as Pro at a fraction of the cost. From "
            "$0.019/image (finalized per job at completion)."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "Use Fast for iteration and high-volume production; switch to V2.5 Pro for quality-sensitive finals.",
            "ONE dedicated filter — 'Sourceful V2.5 Options' — carries every 2.5 knob: font_inputs (max 2, +$0.03/font, carried over from V2) plus scoring_prompt + scoring_rubric and background_mode + background_hex_color.",
            "super_resolution_references is NOT supported — Riverflow 2.5 dropped it (V2 Pro/Fast only); the knob does not exist on 2.5 models.",
            "Supports reasoning effort low/medium/high (xhigh is Pro-only) through the standard reasoning controls.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Font inputs (JSON array)": "JSON array of {font_url, text} objects. Max 2, +$0.03 each.",
            "Scoring prompt": "Free-text instruction the model uses to self-score candidates before returning the best one.",
            "Scoring rubric": "Free-text rubric describing what a good output looks like; pairs with the scoring prompt.",
            "Background mode": "original keeps the generated background, transparent removes it (PNG alpha), solid fills with the hex color.",
            "Background hex color": "#RGB or #RRGGBB fill color. Requires background mode 'solid'.",
        },
    },
    "sourceful/riverflow-v2-max-preview": {
        "display_name": "Sourceful: Riverflow V2 Max (Preview)",
        "best_known_for": (
            "Preview release of the highest-tier Riverflow variant. Higher "
            "fidelity than Pro but preview status means specs may shift. "
            "Pure-image-only output."
        ),
        "tips_and_pitfalls": [
            "Preview — quality and pricing may change without notice.",
            "Does NOT support font_inputs / super_resolution_references — those are Pro/Fast only on the v2 line.",
            "Standard 10 aspect ratios + 1K/2K/4K via the generic filter.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "sourceful/riverflow-v2-standard-preview": {
        "display_name": "Sourceful: Riverflow V2 Standard (Preview)",
        "best_known_for": (
            "Standard preview release of Riverflow V2 — entry-tier quality "
            "and pricing. Pure-image-only."
        ),
        "tips_and_pitfalls": [
            "Preview status — specs may change.",
            "No Sourceful-specific extensions on this variant (font_inputs / super_resolution_references are Pro/Fast only).",
            "Standard knob set applies via generic filter.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "sourceful/riverflow-v2-fast-preview": {
        "display_name": "Sourceful: Riverflow V2 Fast (Preview)",
        "best_known_for": (
            "Preview release of the fastest Riverflow tier. Pure-image-only "
            "with reduced quality versus Pro/Standard at lower cost."
        ),
        "tips_and_pitfalls": [
            "Preview — pricing/quality may shift.",
            "No Sourceful-specific extensions (font_inputs / super_resolution_references are Pro/Fast non-preview only).",
            "Standard knob set via generic filter.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "black-forest-labs/flux.2-pro": {
        "display_name": "Black Forest Labs: FLUX.2 Pro",
        "best_known_for": (
            "Black Forest Labs' premium FLUX.2 model — pure-image-only with "
            "strong photorealism and prompt adherence. Best for high-quality "
            "deliverables. Supports seed for deterministic generation."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "Seed support enables deterministic regeneration with same prompt + seed.",
            "Standard 10 aspect ratios + 1K/2K/4K via generic filter.",
            "No Sourceful-only or Gemini-only extensions.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "black-forest-labs/flux.2-max": {
        "display_name": "Black Forest Labs: FLUX.2 Max",
        "best_known_for": (
            "Highest-tier FLUX.2 — best fidelity in the Black Forest Labs "
            "lineup. Pure-image-only with seed support. Reserve for hero "
            "shots and finals where Pro isn't enough."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "Seed enables deterministic regeneration.",
            "Most expensive FLUX tier — use for finals only.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "black-forest-labs/flux.2-flex": {
        "display_name": "Black Forest Labs: FLUX.2 Flex",
        "best_known_for": (
            "Mid-tier FLUX.2 balancing quality and cost. Pure-image-only "
            "with seed support. Best for general production work."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "Seed support; balanced cost-quality vs Pro/Max.",
            "Standard knob set via generic filter.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "black-forest-labs/flux.2-klein-4b": {
        "display_name": "Black Forest Labs: FLUX.2 Klein 4B",
        "best_known_for": (
            "Smallest FLUX.2 variant (4B parameters) — lowest cost in the "
            "FLUX lineup. Pure-image-only with seed support. Best for high-"
            "volume / draft work."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "Seed support; cheapest FLUX tier.",
            "Quality trades against cost — use for iteration, not finals.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "bytedance-seed/seedream-4.5": {
        "display_name": "ByteDance Seed: Seedream 4.5",
        "best_known_for": (
            "ByteDance Seed's image-only model. Pure-image-only output; "
            "supports temperature and top_p for controlled generation."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "Supports temperature/top_p (unusual for image models) — useful for varied outputs from same prompt.",
            "Standard knob set via generic filter; no model-specific extensions.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
        },
    },
    "recraft/recraft-v3": {
        "display_name": "Recraft: Recraft V3",
        "best_known_for": (
            "Recraft's typography champion — the only AI image model that can "
            "render long-form text (full sentences and paragraphs) reliably AND "
            "place text at exact positions inside the image. 20B parameters, "
            "released Oct 2024, held #1 on the Artificial Analysis benchmark for "
            "5+ consecutive months at launch (beating Midjourney/DALL-E/FLUX). "
            "Used in production by Shopify and Salesforce. Pure-image-only at ~1K "
            "resolution. Best for posters, signage, packaging, marketing assets "
            "with embedded copy."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text in chat.",
            "ONLY Recraft variant with `style` and `text_layout`. V4 / V4 Pro lack both.",
            "For text rendering: put exact wording in quotes in your prompt AND use `text_layout` for precise placement (V3-exclusive feature).",
            "Style names: see https://www.recraft.ai/docs/api-reference/styles. Vector styles NOT supported via OpenRouter.",
            "text_layout: array of {text, bbox} where bbox is 4 [x,y] corners in 0-1 coords (order: TL, TR, BR, BL).",
            "Image-to-image: only one input image supported. Use `strength` 0.0-1.0 to control deviation (default 0.5).",
            "If you need newer composition / cleaner geometry → V4 / V4 Pro (but lose text_layout + style).",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5). Lower = closer to input image.",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255). Hints the output palette.",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255). Forces a specific background color.",
            "Recraft style": "Artistic style preset (V3 only), e.g. \"Photorealism\". Empty = no style override.",
            "Text layout (JSON array)": "Place text at exact positions (V3 ONLY). Each entry: {text, bbox: 4 [x,y] corners in 0-1 coords}.",
        },
    },
    "recraft/recraft-v4": {
        "display_name": "Recraft: Recraft V4",
        "best_known_for": (
            "Recraft's Feb 2026 ground-up rebuild — \"design taste meets image "
            "generation.\" 1024x1024 raster output, ~10s/image. Topped the "
            "Hugging Face Text-to-Image Arena (blind human preference) over "
            "Midjourney V8, DALL-E 3, FLUX, and Stable Diffusion. Strengths: "
            "balanced composition, cohesive color, clean readable embedded text "
            "(short / mid-length), and outputs that feel deliberate rather than "
            "stock-like. Best for infographics, signage, packaging, branded "
            "social/web assets, and rapid iteration."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "Does NOT support `style` or `text_layout` — those are V3 ONLY. For long-form text or precise placement use V3.",
            "Has `strength` + `rgb_colors` + `background_rgb_color` (3 Recraft image_config params).",
            "Image-to-image: only one input image supported.",
            "V4 limitations (per Recraft): photorealistic human faces and hands can be unreliable; not the right tool for editorial portraiture.",
            "Use V4 for fast iteration and social/web assets; switch to V4 Pro for print-ready finals at 2K.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255).",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255).",
        },
    },
    "recraft/recraft-v4-pro": {
        "display_name": "Recraft: Recraft V4 Pro",
        "best_known_for": (
            "Premium V4 — same design taste, 2x resolution. Outputs at 2048x2048 "
            "(~4 megapixels), ~30s/image. Built for print-ready work where fine "
            "detail matters: magazine layouts, posters, billboards, packaging, "
            "editorial illustration. Same prompt accuracy and creative judgment "
            "as V4 but with sharper geometry, finer textures, and better "
            "anatomy/realism in complex compositions. Flat $0.25 per image on "
            "OpenRouter."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "Same image_config knobs as V4 (strength + rgb_colors + background_rgb_color); NO style or text_layout (those are V3 ONLY).",
            "~3x slower than V4 due to higher resolution — reserve for finals, not iteration.",
            "$0.25 per image — flat per-image fee, not per-token.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4; not ideal for portraiture.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255).",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255).",
        },
    },
    "recraft/recraft-v4-pro-vector": {
        "display_name": "Recraft: Recraft V4 Pro Vector",
        "best_known_for": (
            "Vector (SVG) variant of V4 Pro — produces resolution-independent "
            "SVG markup instead of raster pixels. Same design taste as V4 Pro, "
            "scaled to ~2K equivalent detail. Best for logos, icons, infographics, "
            "and any asset that needs to be scaled or edited downstream in vector "
            "tools (Illustrator, Figma, Inkscape). Output is true `<svg>` markup "
            "embedded in a `data:image/svg+xml;base64,...` URL."
        ),
        "tips_and_pitfalls": [
            "Output is SVG, not PNG/JPEG — scales infinitely without quality loss.",
            "Prefer simple, graphic prompts (logos, icons, flat illustrations) over photoreal subjects; SVG cannot represent photographic detail.",
            "`rgb_colors` and `background_rgb_color` are sent through, but how the vector model honors them is not documented — verify visually if you rely on them.",
            "`strength` for image-to-image works but the input is rasterised internally; SVG comes from the model, not from the input.",
            "OpenRouter returns the SVG inline as base64; OWUI renders it natively in the chat — no rasterisation on our side.",
            "Same human-subject limitations as V4 Pro.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default. Vector output scales freely, but the model still picks an internal canvas.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255). Effect on vector output is undocumented.",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255). Effect on vector output is undocumented.",
        },
    },
    "recraft/recraft-v4-vector": {
        "display_name": "Recraft: Recraft V4 Vector",
        "best_known_for": (
            "Vector (SVG) variant of V4 — same design taste at ~1K equivalent "
            "detail, output as scalable SVG markup. Best for logos, icons, flat "
            "illustrations, and any asset destined for vector editing or "
            "infinite scaling. Faster and cheaper than V4 Pro Vector for "
            "iteration; reserve Pro Vector for finals."
        ),
        "tips_and_pitfalls": [
            "Output is SVG, not PNG/JPEG — scales infinitely without quality loss.",
            "Prefer simple, graphic prompts (logos, icons, flat illustrations) over photoreal subjects; SVG cannot represent photographic detail.",
            "`rgb_colors` and `background_rgb_color` are sent through, but how the vector model honors them is not documented — verify visually if you rely on them.",
            "`strength` for image-to-image works but the input is rasterised internally; SVG comes from the model.",
            "OpenRouter returns the SVG inline as base64; OWUI renders it natively in the chat.",
            "Use V4 Vector for iteration; V4 Pro Vector for higher-fidelity finals.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default. Vector output scales freely.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255). Effect on vector output is undocumented.",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255). Effect on vector output is undocumented.",
        },
    },
    "recraft/recraft-v4.1": {
        "display_name": "Recraft: Recraft V4.1",
        "best_known_for": (
            "V4.1 is Recraft's May 2026 aesthetic refresh of V4 — same 1024x1024 "
            "raster output, same image_config surface, but tuned for stronger "
            "composition, color cohesion, and visual polish. Best for marketing "
            "assets, social posts, hero imagery, and any work where the V4 "
            "output felt almost-but-not-quite-right aesthetically. Same speed "
            "envelope as V4 (~10s/image)."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "Same image_config knobs as V4 (strength + rgb_colors + background_rgb_color); NO style or text_layout (those are V3 ONLY).",
            "Drop-in successor to V4 — try V4.1 first; fall back to V4 if its aesthetic doesn't suit a specific brand.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4; not ideal for portraiture.",
            "For general-purpose / cost-sensitive work without aesthetic emphasis, prefer the V4.1 Utility variants.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255).",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255).",
        },
    },
    "recraft/recraft-v4.1-pro": {
        "display_name": "Recraft: Recraft V4.1 Pro",
        "best_known_for": (
            "V4.1 Pro is the high-resolution counterpart to V4.1 — same aesthetic "
            "tuning, 2048x2048 raster output (~4 MP), ~30s/image. Built for "
            "print-ready aesthetic work: magazine layouts, posters, billboards, "
            "editorial illustration where V4 Pro felt close but the polish was "
            "off. Use V4.1 for iteration, V4.1 Pro for finals."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "Same image_config knobs as V4.1 (strength + rgb_colors + background_rgb_color); NO style or text_layout (V3 ONLY).",
            "~3x slower than V4.1 due to higher resolution — reserve for finals.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4.1; not ideal for portraiture.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255).",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255).",
        },
    },
    "recraft/recraft-v4.1-pro-vector": {
        "display_name": "Recraft: Recraft V4.1 Pro Vector",
        "best_known_for": (
            "Vector (SVG) variant of V4.1 Pro — V4.1's aesthetic tuning, ~2K "
            "equivalent detail, true `<svg>` output. Best for high-polish logos, "
            "editorial icon sets, and brand assets that need to scale and edit "
            "downstream. OpenRouter returns the SVG inline as a "
            "`data:image/svg+xml;base64,...` URL; OWUI renders it natively."
        ),
        "tips_and_pitfalls": [
            "Output is SVG, not PNG/JPEG — scales infinitely without quality loss.",
            "Prefer simple, graphic prompts (logos, icons, flat illustrations) over photoreal subjects.",
            "`rgb_colors` and `background_rgb_color` are sent through, but how the vector model honors them is not documented — verify visually if you rely on them.",
            "`strength` for image-to-image works but the input is rasterised internally; output is SVG either way.",
            "Use V4.1 Vector for iteration; V4.1 Pro Vector for finals.",
            "Same aesthetic tuning advantage over V4 Pro Vector — try V4.1 Pro Vector first for vector work.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default. Vector output scales freely.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255). Effect on vector output is undocumented.",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255). Effect on vector output is undocumented.",
        },
    },
    "recraft/recraft-v4.1-utility": {
        "display_name": "Recraft: Recraft V4.1 Utility",
        "best_known_for": (
            "Recraft's general-purpose V4.1 variant — drops the aesthetic-tuning "
            "bias of the regular V4.1 in exchange for broader subject coverage "
            "and faster/cheaper generation. Best for spot illustrations, "
            "diagrams, placeholder/stock imagery, and any work where 'on-brand "
            "aesthetics' is not the goal. 1024x1024 raster output."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "Pick Utility over regular V4.1 when you need versatility, not aesthetic polish.",
            "Same image_config knobs as V4.1 (strength + rgb_colors + background_rgb_color); NO style or text_layout (V3 ONLY).",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4.1.",
            "Use Utility for fast/cheap work; switch to regular V4.1 (aesthetic) or V4.1 Pro (print) when output quality matters.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255).",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255).",
        },
    },
    "recraft/recraft-v4.1-utility-pro": {
        "display_name": "Recraft: Recraft V4.1 Utility Pro",
        "best_known_for": (
            "High-resolution counterpart to V4.1 Utility — 2048x2048 (~4 MP) "
            "general-purpose raster output. Same versatility / non-aesthetic "
            "bias as the base Utility variant, with 2x the resolution for "
            "larger placements. Use for general-purpose finals where aesthetic "
            "polish is not the goal."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "Same image_config knobs as V4.1 Utility (strength + rgb_colors + background_rgb_color); NO style or text_layout (V3 ONLY).",
            "~3x slower than V4.1 Utility due to higher resolution — reserve for finals.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4.1.",
            "Use Utility Pro when you need higher resolution but not aesthetic tuning; otherwise prefer V4.1 Pro.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255).",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255).",
        },
    },
    "recraft/recraft-v4.1-vector": {
        "display_name": "Recraft: Recraft V4.1 Vector",
        "best_known_for": (
            "Vector (SVG) variant of V4.1 — V4.1's aesthetic tuning, ~1K "
            "equivalent detail, true `<svg>` output. Best for aesthetic-driven "
            "logos, icon sets, and flat illustrations destined for vector "
            "editing. Faster/cheaper than V4.1 Pro Vector for iteration."
        ),
        "tips_and_pitfalls": [
            "Output is SVG, not PNG/JPEG — scales infinitely without quality loss.",
            "Prefer simple, graphic prompts (logos, icons, flat illustrations) over photoreal subjects.",
            "`rgb_colors` and `background_rgb_color` are sent through, but how the vector model honors them is not documented — verify visually if you rely on them.",
            "`strength` for image-to-image works but the input is rasterised internally; output is SVG either way.",
            "OpenRouter returns the SVG inline as base64; OWUI renders it natively.",
            "Use V4.1 Vector for iteration; V4.1 Pro Vector for higher-fidelity finals.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default. Vector output scales freely.",
            "Strength (image-to-image)": "0.0-1.0; 0.0 = skip / use model default (0.5).",
            "RGB color palette (JSON array)": "JSON array of [r,g,b] arrays (each 0-255). Effect on vector output is undocumented.",
            "Background RGB color (JSON array)": "Single [r,g,b] array (each 0-255). Effect on vector output is undocumented.",
        },
    },
    "x-ai/grok-imagine-image-quality": {
        "display_name": "xAI: Grok Imagine Image Quality",
        "best_known_for": (
            "xAI's fast, high-fidelity image generation and editing model. "
            "Accepts text prompts and optional reference images; produces "
            "photorealistic outputs at 1K or 2K. Best for photoreal scenes, "
            "compositional control, and workflows that need Grok-only tall "
            "phone-screen aspect ratios (9:19.5, 9:20, 1:2, 2:1) or an `auto` "
            "ratio that lets the model pick frame shape from prompt."
        ),
        "tips_and_pitfalls": [
            "Use the Grok aspect ratio knob (14 values, including phone-tall and `auto`) for Grok-specific frames. The generic knob's 10-value set still works but won't expose the wide/tall variants.",
            "`n` lets you fan out 1-10 variations per request — cost scales linearly. Pick `n=1` (default) for iteration; bump to 3-5 for exploration.",
            "Multimodal input: pair the prompt with reference images for editing/style transfer.",
            "Charged per image output ($0.01/image at OpenRouter's listed rate).",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Generic 10-value ratio (kept for compatibility). Prefer the Grok-specific knob below for Grok-only frames.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Image aspect ratio (Grok Imagine)": "Grok-supported 14 values including tall phone formats (9:19.5, 9:20, 1:2, 2:1) and `auto`. Overrides the generic aspect ratio when set.",
            "Number of images (1-10)": "Number of images per request. 0 = skip (default 1). Cost scales linearly.",
        },
    },
}

# Public re-export name (mirror of VIDEO_HELP_BY_MODEL convention).
IMAGE_HELP_BY_MODEL = _IMAGE_PER_MODEL_HELP_DATA


# Maps human-readable knob labels to gate predicates resolved against the
# model's catalog metadata. None means "always show". String values match
# specific entries in `allowed_passthrough_parameters`/architecture.
_IMAGE_KNOB_GATE: dict[str, str | None] = {
    "Image aspect ratio": None,  # always shown when generic filter attached
    "Image size": None,  # always shown when generic filter attached
    "Image aspect ratio (Gemini extended)": "gemini_extended",
    "Image size (Gemini-only 0.5K)": "gemini_extended",
    "Font inputs (JSON array)": "sourceful_extended",
    "Super-resolution references (JSON array)": "sourceful_v2_superres",
    "Scoring prompt": "sourceful_v25",
    "Scoring rubric": "sourceful_v25",
    "Background mode": "sourceful_v25",
    "Background hex color": "sourceful_v25",
    "Strength (image-to-image)": "recraft_common",
    "RGB color palette (JSON array)": "recraft_common",
    "Background RGB color (JSON array)": "recraft_common",
    "Recraft style": "recraft_v3_only",
    "Text layout (JSON array)": "recraft_v3_only",
    "Image aspect ratio (Grok Imagine)": "grok_imagine",
    "Number of images (1-10)": "grok_imagine",
}


def _is_gemini_extended_ratio_model(model_id: str) -> bool:
    """Match models eligible for Gemini-extended knobs (Flash 3.x — not Pro or 2.5)."""
    import re
    return bool(re.match(r"^~?google/gemini-3.*flash-image.*$", model_id or ""))


def _is_sourceful_pro_or_fast(model_id: str) -> bool:
    """Match models eligible for Sourceful-extended knobs (font_inputs — V2 and newer)."""
    import re
    return bool(re.match(r"^~?sourceful/riverflow-v\d+(\.\d+)?-(pro|fast)$", model_id or ""))


def _is_sourceful_v2_superres(model_id: str) -> bool:
    """Match models eligible for super_resolution_references (V2 Pro/Fast ONLY — dropped in 2.5)."""
    import re
    return bool(re.match(r"^~?sourceful/riverflow-v2-(pro|fast)$", model_id or ""))


def _is_sourceful_v25(model_id: str) -> bool:
    """Match models eligible for Riverflow 2.5 extras (scoring + background controls)."""
    import re
    return bool(re.match(r"^~?sourceful/riverflow-v2\.5-(pro|fast)$", model_id or ""))


def _is_recraft(model_id: str) -> bool:
    """Match models eligible for Recraft common knobs (strength, rgb_colors, background_rgb_color)."""
    import re
    return bool(re.match(r"^~?recraft/recraft-", model_id or ""))


def _is_recraft_v3(model_id: str) -> bool:
    """Match models eligible for Recraft V3-only knobs (style, text_layout)."""
    import re
    return bool(isinstance(model_id, str) and re.match(r"^~?recraft/recraft-v3\Z", model_id))


def _is_grok_imagine_image(model_id: str) -> bool:
    """Match models eligible for Grok Imagine-specific knobs (14 aspect ratios, n)."""
    import re
    return bool(re.match(r"^~?x-ai/grok-imagine-image-", model_id or ""))


def _image_knob_is_active(knob: str, model_id: str) -> bool:
    gate = _IMAGE_KNOB_GATE.get(knob)
    if gate is None:
        return True
    if gate == "gemini_extended":
        return _is_gemini_extended_ratio_model(model_id)
    if gate == "sourceful_extended":
        return _is_sourceful_pro_or_fast(model_id)
    if gate == "sourceful_v2_superres":
        return _is_sourceful_v2_superres(model_id)
    if gate == "sourceful_v25":
        return _is_sourceful_v25(model_id)
    if gate == "recraft_common":
        return _is_recraft(model_id)
    if gate == "recraft_v3_only":
        return _is_recraft_v3(model_id)
    if gate == "grok_imagine":
        return _is_grok_imagine_image(model_id)
    return False


def _image_render_template(model_id: str, image_model: dict[str, Any] | None) -> str:
    entry = _IMAGE_PER_MODEL_HELP_DATA.get(model_id)
    if not entry:
        return _image_render_catalog_fallback(model_id, image_model)
    parts: list[str] = []
    parts.append(f"# {entry['display_name']}")
    parts.append("")
    parts.append(entry.get("best_known_for", ""))
    parts.append("")
    tips = entry.get("tips_and_pitfalls") or []
    if tips:
        parts.append("## Tips & pitfalls")
        for tip in tips:
            parts.append(f"- {tip}")
        parts.append("")
    knob_desc = entry.get("knob_descriptions") or {}
    if knob_desc:
        parts.append("## Knobs")
        for label, desc in knob_desc.items():
            if _image_knob_is_active(label, model_id):
                parts.append(f"- `{label}`: {desc}")
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _image_render_catalog_fallback(model_id: str, image_model: dict[str, Any] | None) -> str:
    if not isinstance(image_model, dict):
        image_model = {}
    name = image_model.get("name") or model_id
    description = image_model.get("description") or "(no description)"
    arch = image_model.get("architecture") or {}
    out_mods = arch.get("output_modalities") or []
    in_mods = arch.get("input_modalities") or []
    parts = [
        f"# {name}",
        "",
        description,
        "",
        f"- **Output modalities**: {', '.join(out_mods) or '(none)'}",
        f"- **Input modalities**: {', '.join(in_mods) or '(none)'}",
        "",
        "_No curated help available for this model. Catalog metadata shown above._",
        "",
    ]
    return "\n".join(parts)


def render_image_help(model_id: str, image_model: dict[str, Any] | None = None) -> str:
    """Render help text for an image-output model. Falls back to catalog
    metadata if no curated entry exists for the model id."""
    return _image_render_template((model_id or "").strip(), image_model)
