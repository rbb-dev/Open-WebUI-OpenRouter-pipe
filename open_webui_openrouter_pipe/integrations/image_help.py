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
            "Standard 10 aspect ratios + 1K/2K/4K. No 0.5K or extended ratios on this variant — those are Gemini 3.1 Flash Image Preview only.",
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
            "iteration. Only Gemini variant with these extensions."
        ),
        "tips_and_pitfalls": [
            "Use the dedicated 'Gemini Options' filter for extended ratios (4:1, 1:4, 8:1, 1:8) and 0.5K size — these are Gemini Flash Image Preview ONLY.",
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
            "super_resolution_references (max 4, +$0.20/ref) requires input images in messages (image-to-image only).",
            "Both extensions exposed via 'Sourceful Options' filter; cardinality caps validated at inlet (rejects 3+ font_inputs before submission).",
            "4.5MB request size limit — pass image URLs instead of base64 to avoid bloat.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Font inputs (JSON array)": "JSON array of {font_url, text} objects for custom typeface rendering. Max 2, +$0.03 each.",
            "Super-resolution references (JSON array)": "JSON array of URL strings for image-to-image upscaling. Max 4, +$0.20 each. Image-to-image only.",
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
            "Same Sourceful-specific knobs (font_inputs, super_resolution_references) via dedicated filter.",
        ],
        "knob_descriptions": {
            "Image aspect ratio": "Frame shape (10 standard ratios). Empty = model default.",
            "Image size": "Resolution tier (1K/2K/4K). Empty = model default.",
            "Font inputs (JSON array)": "JSON array of {font_url, text} objects. Max 2, +$0.03 each.",
            "Super-resolution references (JSON array)": "JSON array of URL strings for image-to-image upscaling. Max 4, +$0.20 each. Image-to-image only.",
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
    "Super-resolution references (JSON array)": "sourceful_extended",
}


def _is_gemini_flash_image_preview(model_id: str) -> bool:
    """Match models eligible for Gemini-extended knobs."""
    import re
    return bool(re.match(r"^google/gemini-.*flash-image.*-preview$", model_id or ""))


def _is_sourceful_pro_or_fast(model_id: str) -> bool:
    """Match models eligible for Sourceful-extended knobs (font_inputs, super_resolution_references)."""
    import re
    return bool(re.match(r"^sourceful/riverflow-v\d+(\.\d+)?-(pro|fast)$", model_id or ""))


def _image_knob_is_active(knob: str, model_id: str) -> bool:
    gate = _IMAGE_KNOB_GATE.get(knob)
    if gate is None:
        return True
    if gate == "gemini_extended":
        return _is_gemini_flash_image_preview(model_id)
    if gate == "sourceful_extended":
        return _is_sourceful_pro_or_fast(model_id)
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
