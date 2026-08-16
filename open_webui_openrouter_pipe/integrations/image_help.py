from __future__ import annotations

import math
from typing import Any, NamedTuple

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
            "Pricing follows GPT-5 chat token economics; image output is included in completion tokens.",
        ],
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
        ],
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
        ],
    },
    "google/gemini-2.5-flash-image": {
        "display_name": "Google: Gemini 2.5 Flash Image",
        "best_known_for": (
            "Google's standard Gemini multimodal text+image model. Best for "
            "prompt-following tasks with cinematic composition and natural-"
            "looking output. Outputs both text and image."
        ),
        "tips_and_pitfalls": [
            "Multimodal: model decides emission based on prompt; be explicit.",
            "Strong at photoreal scenes and prompt-faithful composition.",
        ],
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
            "Multimodal text+image output.",
        ],
    },
    "google/gemini-3.1-flash-image-preview": {
        "display_name": "Google: Gemini 3.1 Flash Image (Preview)",
        "best_known_for": (
            "Cost-optimized Gemini 3.1 with native image output AND unique "
            "extended knobs: 4 extra aspect ratios (1:4, 4:1, 1:8, 8:1) for "
            "ultrawide/tall layouts AND a 512 low-res tier for cheap "
            "iteration. The Gemini 3.x Flash Image line (GA + preview) has these; Pro and 2.5 do not."
        ),
        "tips_and_pitfalls": [
            "512 renders far fewer pixels than 1K, and this model bills by token, so an iteration pass at 512 costs materially less.",
        ],
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
        ],
    },
    "qwen/qwen-image-3": {
        "display_name": "Qwen: Qwen Image 3",
        "best_known_for": (
            "Qwen's unified generation-and-editing model. Renders text and detail "
            "down to about 10px, so it holds up on small type, labels and fine "
            "linework where most models blur. Good general-purpose choice when the "
            "picture has to carry legible words."
        ),
        "tips_and_pitfalls": [
            "Small text is its strength — quote the exact wording you want in the prompt.",
            "Generates and edits with the same model, so a follow-up like \"make the sign green\" works on the picture it just made.",
        ],
    },
    "qwen/qwen-image-3-pro": {
        "display_name": "Qwen: Qwen Image 3 Pro",
        "best_known_for": (
            "The larger Qwen Image 3. Same fine-detail and small-text strengths "
            "with more world knowledge behind them, so prompts that lean on real "
            "places, products or conventions come out closer to right. Costs more "
            "per image than the base model."
        ),
        "tips_and_pitfalls": [
            "Worth the step up when the prompt depends on knowing something, not just rendering it.",
            "For iteration, draft on the base model and finish here.",
        ],
    },
    "krea/krea-2-large": {
        "display_name": "Krea: Krea 2 Large",
        "best_known_for": (
            "Krea's largest model, more than twice the size of Krea 2 Medium. Its "
            "lighter post-training leaves images looking rawer and less "
            "house-styled than most models — closer to photography than to "
            "illustration. Best when the polished AI look is the problem."
        ),
        "tips_and_pitfalls": [
            "Less post-training means less smoothing: expect texture, grain and imperfection rather than a clean render.",
            "If output looks too raw for the brief, Krea 2 Medium is the more finished sibling.",
        ],
    },
    "krea/krea-2-medium": {
        "display_name": "Krea: Krea 2 Medium",
        "best_known_for": (
            "Krea's balanced model and the sensible default of the family. Heavier "
            "post-training gives stable, consistent output across a wide range of "
            "prompts at a lower cost than Krea 2 Large."
        ),
        "tips_and_pitfalls": [
            "The most predictable of the three Krea models — good for work that has to look consistent across a set.",
            "Reach for Large when you want rawer texture, Turbo when you want speed.",
        ],
    },
    "krea/krea-2-medium-turbo": {
        "display_name": "Krea: Krea 2 Medium Turbo",
        "best_known_for": (
            "A distilled, speed-focused Krea 2 Medium. Built for rapid iteration "
            "and graphic-design exploration where getting twenty options quickly "
            "matters more than getting one perfect."
        ),
        "tips_and_pitfalls": [
            "Explore here, then re-run the prompt you settled on through Krea 2 Medium or Large for the final.",
            "Distillation trades some fidelity for speed — fine detail is where you will notice it.",
        ],
    },
    "openai/gpt-image-1": {
        "display_name": "OpenAI: GPT Image 1",
        "best_known_for": (
            "OpenAI's image model on their dedicated Images API. Accurate text "
            "rendering, transparent backgrounds, and editing with up to sixteen "
            "reference images — the widest reference support of any model here."
        ),
        "tips_and_pitfalls": [
            "Transparent backgrounds come from the background setting, not from asking for them in the prompt.",
            "For editing, attach the references to your message; it reads far more of them than most models.",
        ],
    },
    "openai/gpt-image-1-mini": {
        "display_name": "OpenAI: GPT Image 1 Mini",
        "best_known_for": (
            "The cheaper, faster GPT Image 1. Same API and the same controls, at "
            "reduced latency and cost. Best for volume work and for iterating "
            "before committing to the full model."
        ),
        "tips_and_pitfalls": [
            "Quality gap shows most on fine text and complex composition; for plain subjects it is hard to tell apart.",
        ],
    },
    "openai/gpt-image-2": {
        "display_name": "OpenAI: GPT Image 2",
        "best_known_for": (
            "OpenAI's newest image model, on the dedicated Images API. "
            "High-fidelity generation and editing, and the current default choice "
            "among the OpenAI image models unless you need GPT Image 1's "
            "sixteen-reference editing specifically."
        ),
        "tips_and_pitfalls": [
            "Answers only on the image endpoint — it is not available as a chat model.",
        ],
    },
    "google/gemini-3-pro-image": {
        "display_name": "Google: Nano Banana Pro (Gemini 3 Pro Image)",
        "best_known_for": (
            "Google's most capable image model, built on Gemini 3 Pro. Its "
            "advantage is reasoning: prompts that describe a situation rather than "
            "a picture — diagrams, annotated scenes, images that have to be "
            "internally consistent — come out markedly better than from faster models."
        ),
        "tips_and_pitfalls": [
            "Give it the reasoning to do: describe what the image must be true about, not only what it should look like.",
            "The slowest and priciest Gemini image model; use Flash for iteration and come here for the final.",
        ],
    },
    "google/gemini-3.1-flash-image": {
        "display_name": "Google: Nano Banana 2 (Gemini 3.1 Flash Image)",
        "best_known_for": (
            "Google's latest generation-and-editing model, delivering close to "
            "Pro-level quality at Flash speed and price. The general-purpose "
            "recommendation in the Gemini image line."
        ),
        "tips_and_pitfalls": [
            "Edits conversationally: describe the change rather than re-describing the whole image.",
        ],
    },
    "google/gemini-3.1-flash-lite-image": {
        "display_name": "Google: Nano Banana 2 Lite (Gemini 3.1 Flash Lite Image)",
        "best_known_for": (
            "The fastest and cheapest Gemini image model, built for high-velocity "
            "pipelines and rapid visual exploration. Best when you want many "
            "options quickly and will refine the winner elsewhere."
        ),
        "tips_and_pitfalls": [
            "Trades detail for speed — good for composition and layout exploration, less so for a finished asset.",
        ],
    },
    "microsoft/mai-image-2.5-pro": {
        "display_name": "Microsoft: MAI-Image-2.5 Pro",
        "best_known_for": (
            "The larger MAI-Image-2.5, served via Azure AI Foundry. Photorealistic "
            "and artistic output with the same token-based pricing as the base "
            "model, at higher quality and cost."
        ),
        "tips_and_pitfalls": [
            "Token-priced rather than per-image, so long prompts cost proportionally more.",
            "Accepts reference images alongside the prompt for editing and guidance.",
        ],
    },
    "microsoft/mai-image-2.5": {
        "display_name": "Microsoft: MAI-Image-2.5",
        "best_known_for": (
            "Microsoft's high-quality image generation model served via Azure "
            "AI Foundry — photorealistic and artistic output from text prompts "
            "with optional reference-image input. Best for general-purpose "
            "photoreal work on Azure-backed infrastructure, billed by token "
            "rather than by picture."
        ),
        "tips_and_pitfalls": [
            "Token-priced rather than per-image, so a long prompt costs more than a short one for the same picture.",
            "Multimodal input: accepts reference images alongside the text prompt for editing/guidance.",
        ],
    },
    "sourceful/riverflow-v2-pro": {
        "display_name": "Sourceful: Riverflow V2 Pro",
        "best_known_for": (
            "Sourceful's premium tier — pure image-only output with custom "
            "font rendering and image-to-image super-resolution. Strongest "
            "for marketing assets requiring exact text rendering at scale."
        ),
        "tips_and_pitfalls": [
            "4.5MB request size limit — pass image URLs instead of base64 to avoid bloat.",
        ],
    },
    "sourceful/riverflow-v2-fast": {
        "display_name": "Sourceful: Riverflow V2 Fast",
        "best_known_for": (
            "Faster, cheaper variant of Riverflow V2 — same Sourceful "
            "quality and reduced cost. Best for iteration before committing "
            "to a Pro render."
        ),
        "tips_and_pitfalls": [
            "Same caveats as Riverflow V2 Pro: pure-image-only, 4.5MB request limit, image URLs preferred.",
            "Use Fast for prompt iteration and font/reference tuning; switch to Pro for finals.",
        ],
    },
    "sourceful/riverflow-v2.5-pro": {
        "display_name": "Sourceful: Riverflow V2.5 Pro",
        "best_known_for": (
            "The most powerful variant of Sourceful's Riverflow 2.5 lineup — "
            "a unified text-to-image and image-to-image family. Best for "
            "top-tier control and quality-sensitive outputs: brand assets, "
            "marketing finals, and work that benefits from the new 2.5 "
            "self-scoring and background controls. Priced per image, rising "
            "with the output size you ask for."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "Pricing is dynamic: the published per-image rate is a starting point, and the final charge is settled per job from the processing it actually took.",
        ],
    },
    "sourceful/riverflow-v2.5-fast": {
        "display_name": "Sourceful: Riverflow V2.5 Fast",
        "best_known_for": (
            "The speed-optimized variant of Sourceful's Riverflow 2.5 lineup "
            "— best for production deployments and latency-critical "
            "workflows. Same unified text-to-image and image-to-image family "
            "and the same 2.5 extras as Pro at a fraction of the cost, with "
            "the charge settled per job at completion."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only — does NOT output text.",
            "Use Fast for iteration and high-volume production; switch to V2.5 Pro for quality-sensitive finals.",
        ],
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
        ],
    },
    "sourceful/riverflow-v2-standard-preview": {
        "display_name": "Sourceful: Riverflow V2 Standard (Preview)",
        "best_known_for": (
            "Standard preview release of Riverflow V2 — entry-tier quality "
            "and pricing. Pure-image-only."
        ),
        "tips_and_pitfalls": [
            "Preview status — specs may change.",
        ],
    },
    "sourceful/riverflow-v2-fast-preview": {
        "display_name": "Sourceful: Riverflow V2 Fast (Preview)",
        "best_known_for": (
            "Preview release of the fastest Riverflow tier. Pure-image-only "
            "with reduced quality versus Pro/Standard at lower cost."
        ),
        "tips_and_pitfalls": [
            "Preview — pricing/quality may shift.",
        ],
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
            "No Sourceful-only or Gemini-only extensions.",
        ],
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
        ],
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
        ],
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
            "If you need newer composition / cleaner geometry → V4 / V4 Pro (but lose text_layout + style).",
        ],
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
            "Image-to-image: only one input image supported.",
            "V4 limitations (per Recraft): photorealistic human faces and hands can be unreliable; not the right tool for editorial portraiture.",
            "Use V4 for fast iteration and social/web assets; switch to V4 Pro for print-ready finals at 2K.",
        ],
    },
    "recraft/recraft-v4-pro": {
        "display_name": "Recraft: Recraft V4 Pro",
        "best_known_for": (
            "Premium V4 — same design taste at a higher resolution. Outputs at 2048x2048 "
            "(~4 megapixels), ~30s/image. Built for print-ready work where fine "
            "detail matters: magazine layouts, posters, billboards, packaging, "
            "editorial illustration. Same prompt accuracy and creative judgment "
            "as V4 but with sharper geometry, finer textures, and better "
            "anatomy/realism in complex compositions. Billed at a flat rate "
            "per image on OpenRouter."
        ),
        "tips_and_pitfalls": [
            "PURE-image-only.",
            "~3x slower than V4 due to higher resolution — reserve for finals, not iteration.",
            "Flat per-image fee rather than per-token, so prompt length does not change what a render costs.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4; not ideal for portraiture.",
        ],
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
            "OpenRouter returns the SVG inline as base64; OWUI renders it natively in the chat — no rasterisation on our side.",
            "Same human-subject limitations as V4 Pro.",
        ],
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
            "OpenRouter returns the SVG inline as base64; OWUI renders it natively in the chat.",
            "Use V4 Vector for iteration; V4 Pro Vector for higher-fidelity finals.",
        ],
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
            "Drop-in successor to V4 — try V4.1 first; fall back to V4 if its aesthetic doesn't suit a specific brand.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4; not ideal for portraiture.",
            "For general-purpose / cost-sensitive work without aesthetic emphasis, prefer the V4.1 Utility variants.",
        ],
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
            "~3x slower than V4.1 due to higher resolution — reserve for finals.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4.1; not ideal for portraiture.",
        ],
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
            "Use V4.1 Vector for iteration; V4.1 Pro Vector for finals.",
            "Same aesthetic tuning advantage over V4 Pro Vector — try V4.1 Pro Vector first for vector work.",
        ],
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
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4.1.",
            "Use Utility for fast/cheap work; switch to regular V4.1 (aesthetic) or V4.1 Pro (print) when output quality matters.",
        ],
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
            "~3x slower than V4.1 Utility due to higher resolution — reserve for finals.",
            "Image-to-image: only one input image supported.",
            "Same human-subject limitations as V4.1.",
            "Use Utility Pro when you need higher resolution but not aesthetic tuning; otherwise prefer V4.1 Pro.",
        ],
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
            "OpenRouter returns the SVG inline as base64; OWUI renders it natively.",
            "Use V4.1 Vector for iteration; V4.1 Pro Vector for higher-fidelity finals.",
        ],
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
            "Multimodal input: pair the prompt with reference images for editing/style transfer.",
            "Charged per generated image, at a higher rate for 2K than for 1K, and reference images you supply are charged on top.",
        ],
    },
}

# Public re-export name (mirror of VIDEO_HELP_BY_MODEL convention).
IMAGE_HELP_BY_MODEL = _IMAGE_PER_MODEL_HELP_DATA


_IMAGE_BILLABLE_LABELS: tuple[tuple[str, str], ...] = (
    ("output_image", "Each image it makes"),
    ("input_image", "Each image you supply"),
    ("input_reference", "Each reference you supply"),
    ("input_font", "Each font you supply"),
    ("input_text", "Your prompt text"),
)

_IMAGE_PRICE_UNITS: tuple[tuple[str, str, int], ...] = (
    ("image", "per image", 1),
    ("megapixel", "per megapixel", 1),
    ("token", "per million tokens", 1000000),
)

_IMAGE_VARIANT_ORDER: tuple[str, ...] = ("", "1k", "2k", "4k")

_IMAGE_NO_PRICE_LINE = (
    "OpenRouter publishes no price for this model. Check what it charges on OpenRouter "
    "before running a batch."
)

_IMAGE_TOKEN_NOTE = (
    "This model bills by token rather than by picture, and how many tokens a picture "
    "comes to is not published, so what one image costs cannot be worked out from these "
    "rates."
)

_IMAGE_COST_CLOSING = (
    "The cost of each generation is reported on the status line when it finishes."
)


class _ImageCharge(NamedTuple):
    order: tuple[int, int]
    label: str
    price: str
    provider: str
    per_token: bool


def _image_amount_text(amount: float) -> str:
    text = f"{amount:.10f}".rstrip("0")
    if text.endswith("."):
        return text + "00"
    if len(text.split(".", 1)[1]) < 2:
        return text + "0"
    return text


def _image_cost_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None


def _image_price_text(unit: str, amount: float) -> str:
    for name, label, factor in _IMAGE_PRICE_UNITS:
        if unit == name:
            return f"${_image_amount_text(amount * factor)} {label}"
    return ""


def _image_provider_name(record: dict[str, Any]) -> str:
    for key in ("provider_name", "provider_slug"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _image_unnamed_charge(billable: str, unit: str) -> str:
    named = billable or "an item it does not name"
    priced = unit or "a unit it does not state"
    return (
        f'OpenRouter publishes a charge for "{named}" here, priced in "{priced}". Check '
        "this model's rates on OpenRouter for what that comes to."
    )


def _image_charges(records: list[dict[str, Any]]) -> tuple[list[_ImageCharge], list[str]]:
    labels = dict(_IMAGE_BILLABLE_LABELS)
    ranked = [name for name, _ in _IMAGE_BILLABLE_LABELS]
    charges: list[_ImageCharge] = []
    unnamed: list[str] = []
    for record in records:
        published = record.get("pricing")
        if not isinstance(published, list):
            continue
        provider = _image_provider_name(record)
        for item in published:
            if not isinstance(item, dict):
                continue
            billable = str(item.get("billable") or "").strip().lower()
            unit = str(item.get("unit") or "").strip().lower()
            variant = str(item.get("variant") or "").strip().lower()
            amount = _image_cost_number(item.get("cost_usd"))
            price = _image_price_text(unit, amount) if amount is not None else ""
            label = labels.get(billable, "")
            if not label or not price:
                unnamed.append(_image_unnamed_charge(billable, unit))
                continue
            tier = (
                _IMAGE_VARIANT_ORDER.index(variant)
                if variant in _IMAGE_VARIANT_ORDER
                else len(_IMAGE_VARIANT_ORDER)
            )
            charges.append(
                _ImageCharge(
                    (ranked.index(billable), tier),
                    f"{label} ({variant.upper()})" if variant else label,
                    price,
                    provider,
                    unit == "token",
                )
            )
    return charges, unnamed


def _image_price_lines(charges: list[_ImageCharge]) -> list[str]:
    lines: list[str] = []
    for label in dict.fromkeys(charge.label for charge in sorted(charges, key=lambda c: c.order)):
        offered: dict[str, list[str]] = {}
        for charge in charges:
            if charge.label == label:
                offered.setdefault(charge.price, []).append(charge.provider)
        if len(offered) == 1:
            lines.append(f"- {label}: {next(iter(offered))}")
            continue
        for price, providers in offered.items():
            serving = ", ".join(sorted({name for name in providers if name}))
            lines.append(f"- {label}: {price} via {serving}" if serving else f"- {label}: {price}")
    return lines


def _image_cost_section(records: list[dict[str, Any]]) -> list[str]:
    charges, unnamed = _image_charges(records)
    body = [*_image_price_lines(charges), *unnamed] if (charges or unnamed) else [_IMAGE_NO_PRICE_LINE]
    if any(charge.per_token for charge in charges):
        body.extend(["", _IMAGE_TOKEN_NOTE])
    body.extend(["", _IMAGE_COST_CLOSING])
    return ["## Cost", "", *body]


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


def render_image_help(
    model_id: str,
    image_model: dict[str, Any] | None = None,
    *,
    endpoint_record: list[dict[str, Any]] | dict[str, Any] | None = None,
) -> str:
    """Describe a model, and list the controls its own contract publishes.

    The prose is hand-written per model -- what it is good at, how to prompt it. The
    control list is not: it is built from the same spec the model's filter is built from,
    so help cannot name a control the chat UI does not draw, which is what a second
    hand-maintained table did until it described seven filters that no longer exist.
    """
    rendered = _image_render_template((model_id or "").strip(), image_model)
    if endpoint_record is None:
        return rendered

    from ..filters.image_filter_renderer import (
        _SCHEMA_ONLY_CAVEAT,
        ALWAYS_ON_CONTROLS,
        IMAGE_KNOB_TITLES,
        _image_shared_by_some,
        _published_records,
        build_image_model_filter_spec,
    )

    spec = build_image_model_filter_spec(model_id, image_model, endpoint_record)
    lines = [f"{rendered.rstrip()}", ""]
    lines.extend(_image_cost_section(_published_records(endpoint_record)))
    lines.extend(["", "## Controls"])
    if not spec.knob_count:
        if spec.published_anything:
            lines.append(
                "- The companies serving this model publish different settings, so none "
                "can be offered without knowing which one will take the request. It "
                "generates with its own defaults."
            )
        else:
            lines.append(
                "- This model publishes no adjustable settings, so it generates with its "
                "own defaults."
            )
        return "\n".join(lines) + "\n"

    for _name, _annotation, _default, title, description in ALWAYS_ON_CONTROLS:
        lines.append(f"- **{title}** — {description}".replace("  ", " "))
    also_offered = dict(spec.narrowed)
    for name, values in spec.enums:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        also = also_offered.get(name, ())
        offered = ", ".join(str(value) for value in (*values, *also))
        caveat = f" {_image_shared_by_some(also)}" if also else ""
        lines.append(
            f"- **{title}** — {description} Choices: {offered}.{caveat}".replace("  ", " ")
        )
    for name in spec.schema_only:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        lines.append(f"- **{title}** — {description} {_SCHEMA_ONLY_CAVEAT}".replace("  ", " "))
    for name, low, high in spec.ranges:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        lines.append(f"- **{title}** — {description} Accepts {low} to {high}.".replace("  ", " "))
    for name in spec.supported:
        title, description = IMAGE_KNOB_TITLES.get(name, (name, ""))
        lines.append(f"- **{title}** — {description}".replace("  ", " "))
    for name in spec.passthrough:
        lines.append(f"- **{name}** — a setting this model's provider accepts.")
    return "\n".join(lines) + "\n"
