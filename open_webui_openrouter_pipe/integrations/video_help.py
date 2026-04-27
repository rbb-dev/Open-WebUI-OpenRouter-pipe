from __future__ import annotations

from typing import Any


_PER_MODEL_HELP_DATA: dict[str, dict[str, Any]] = {
    "google/veo-3.1-fast": {
        "display_name": "Google: Veo 3.1 Fast",
        "best_known_for": (
            "Google DeepMind's speed-and-cost-optimised tier of Veo 3.1, generating 4-, 6-, "
            "or 8-second clips up to 4K with native synchronised audio at roughly 2× the "
            "speed and a fraction of the price of full Veo 3.1. Editor blind tests put "
            "quality within ~1–8% of the full tier while costs run ~60% lower, making it "
            "the workhorse choice for drafting, A/B-testing creative concepts, batch ad "
            "and social content, and image-to-video work where dialogue and SFX must land "
            "in sync."
        ),
        "tips_and_pitfalls": [
            "Front-load one clear shot: cinematography + subject + action + context + style/audio in plain prose, one idea per clip — multi-subject scenes and crowded actions remain a weak spot.",
            "Use first/last frame for controlled transitions: describe the transformation between the two stills (e.g. \"camera arcs 180°, lighting shifts cool→warm\"), not the stills themselves; mismatched aspect/lighting causes identity pops.",
            "Audio is generated from prompt cues — describe ambient sound, dialogue, and SFX explicitly, otherwise you get generic ambience; turning audio off both saves cost and avoids out-of-sync lip movement on talking heads.",
            "At 8s, faces and logos can drift partway through — lock with reference text, reuse a seed when iterating, and use the negative prompt to exclude common failures (\"no text overlays, no extra fingers, no warped logos\").",
        ],
        "knob_descriptions": {
            "Duration": "Picks clip length in seconds — 4 (quick beat), 6 (mid-shot), or 8 (full scene with audio arc); longer durations cost proportionally more and tax motion consistency harder.",
            "Aspect ratio": "Chooses 16:9 for landscape (YouTube/TV) or 9:16 for vertical (Reels/TikTok/Shorts); Veo composes natively for the chosen ratio rather than cropping.",
            "Resolution": "Selects 720p, 1080p, or 4K output tier, which also drives the OpenRouter price SKU (720p cheapest, 4K most expensive) and generation time.",
            "Size": "Pins exact pixel dimensions (e.g., 1920×1080, 2160×3840) when you need a specific canvas instead of letting aspect_ratio + resolution decide.",
            "Frames": "Controls image conditioning — auto/none for pure text-to-video, first_only to animate from a starting still, or first_last to interpolate a controlled transition between two stills.",
            "Negative prompt": "Free-text list of things to exclude (e.g., \"no text, no extra limbs, no logos\") — Veo 3.1 honours negation explicitly per the DeepMind prompt guide.",
            "Audio": "Toggles native synchronised audio generation; off uses the cheaper no-audio price SKU and skips dialogue/SFX, model_default lets the model decide.",
            "Seed": "Integer for deterministic regeneration — same prompt + same seed yields a near-identical clip, useful for iterating prompt tweaks without identity drift.",
            "Provider options JSON": "Escape hatch to inject raw OpenRouter/Google parameters that the typed valves don't expose, for advanced or future fields.",
            "Person generation": "Safety gate for human/face content — \"allow_all\" (broadest), \"allow_adult\" (default per Vertex AI docs, adults only), \"dont_allow\" (no people); blank uses model default.",
            "Conditioning scale": "Float weight (0–1) that biases how strongly reference/frame images steer the output versus the text prompt; 0 leaves the model at its default balance.",
            "Enhance prompt": "Asks the provider to auto-rewrite/expand the prompt before generation — on for richer cinematic detail, off to send your prompt verbatim, model_default to defer.",
        },
    },
    "google/veo-3.1-lite": {
        "display_name": "Google: Veo 3.1 Lite",
        "best_known_for": (
            "Google's most cost-effective Veo 3.1 tier, positioned for high-volume video "
            "applications and rapid iteration where cost-per-clip is the deciding factor. "
            "Unlike Veo 3.1 Fast, it does not sacrifice generation speed for the lower "
            "price — it matches Fast's latency at less than half the cost — making it the "
            "go-to pick for batch pipelines, social automation, and consumer-app "
            "integrations. Tradeoffs are a hard cap at 1080p (no 4K), no video extension, "
            "and slightly less polished visual fidelity, but it retains native synchronised "
            "audio."
        ),
        "tips_and_pitfalls": [
            "Upgrade to Veo 3.1 (full) when you need a final hero cut, 4K output, or video extension — Lite caps at 1080p and cannot extend an existing clip.",
            "Lite is tuned for \"Cinematic Control\" prompts — explicit camera directives like \"slow pan\", \"low-angle tilt\", and named lighting setups land more reliably than vague mood descriptors.",
            "For complex multi-subject scenes or fine character consistency, expect more retries than the full tier — generate a small batch with different seeds rather than over-engineering one prompt.",
            "Download outputs immediately: Google retains generated video URIs for only ~2 days before they expire.",
        ],
        "knob_descriptions": {
            "Duration": "Picks the clip length in seconds from the model's supported set (4, 6, or 8); longer clips cost proportionally more per second.",
            "Aspect ratio": "Selects landscape 16:9 or portrait 9:16 framing — the only two orientations Lite supports.",
            "Resolution": "Chooses 720p (cheapest, $0.05/s with audio) or 1080p ($0.08/s with audio); 4K is not available on this tier.",
            "Size": "Pins exact pixel dimensions (1280×720, 720×1280, 1920×1080, or 1080×1920) when you need a specific output size rather than just a resolution+ratio pair.",
            "Frames": "Controls image-to-video conditioning — auto/none for pure text-to-video, first_only to anchor the opening frame, or first_last to interpolate between a starting and ending image.",
            "Negative prompt": "Free-text list of things to avoid in the output (e.g. \"blurry, watermark, distorted hands\"), passed through as negativePrompt to Vertex.",
            "Audio": "Toggles native synchronised audio generation (ambient sound, SFX, dialogue, music); disabling it drops the price to $0.03/s at 720p or $0.05/s at 1080p.",
            "Seed": "Sets an integer seed for reproducibility — Google notes it improves determinism but does not strictly guarantee identical outputs across runs.",
            "Provider options JSON": "Raw escape hatch for sending arbitrary google-vertex provider fields not covered by the named valves above.",
            "Person generation": "Controls whether humans may appear in output — allow_all, allow_adult, or dont_allow; in EU/UK/CH/MENA only allow_adult is permitted for Veo 3.1.",
            "Conditioning scale": "Float that biases how strongly the model adheres to your input image(s) versus the text prompt when using first/last frame conditioning.",
            "Enhance prompt": "Lets Vertex auto-rewrite your prompt for better results (on), keep it verbatim (off), or use the provider default (model_default).",
        },
    },
    "google/veo-3.1": {
        "display_name": "Google: Veo 3.1",
        "best_known_for": (
            "Google DeepMind's flagship video model, positioned for production-quality "
            "output where visual fidelity is the priority — commercial deliverables, hero "
            "shots, and cinematic sequences. Its standout trait is jointly-diffused native "
            "audio: dialogue, SFX, and ambience are generated alongside the video in a "
            "single pass with lip-sync within roughly 120ms. Compared to Fast and Lite, "
            "the full tier delivers sharper motion, stronger prompt adherence, finer "
            "texture/lighting detail, and access to 4K output, at a higher cost per second. "
            "It also leads MovieGenBench evaluations on overall preference and prompt-"
            "following accuracy."
        ),
        "tips_and_pitfalls": [
            "Write prompts like a shot list: structure as Camera/Lens, Subject, Action, Environment, Lighting, Style, Audio — Veo 3.1 responds far better to film-industry vocabulary than conversational prose.",
            "Specify audio explicitly. If you leave dialogue, SFX, or ambience undefined, Veo defaults to rushed reads, mismatched ambience, or unwanted on-screen subtitles — quote dialogue with \"(no subtitles)\" to suppress captions.",
            "Known weak spots: in-video text rendering is unreliable, hands and limbs can warp, multi-subject scenes drift, and exact object counts break down past ~15 items — use a negative prompt covering \"no warping, no duplicate limbs, no face distortion, no floating objects\" and prefer \"a small group\" over hard numbers.",
            "Keep one dominant action per 8-second clip; conflicting simultaneous actions destabilise physics. For longer narratives, generate separate clips and stitch with last-frame conditioning rather than overloading one prompt.",
        ],
        "knob_descriptions": {
            "Duration": "Length of the generated clip in seconds; Veo 3.1 supports 4, 6, or 8 seconds, with cost scaling per second.",
            "Aspect ratio": "Sets framing as 16:9 (landscape) or 9:16 (vertical); pick 9:16 for mobile/social and 16:9 for cinematic or hero content.",
            "Resolution": "Chooses the output detail tier — 720p, 1080p, or 4K — where 4K roughly doubles the per-second price and is the premium-only capability for this model.",
            "Size": "Locks the exact pixel dimensions (e.g. 1920×1080, 2160×3840) when you need a specific frame size rather than just an aspect/resolution pair.",
            "Frames": "Lets you anchor generation with a first_frame and/or last_frame image, ideal for image-to-video starts and for stitching shots into longer continuous scenes.",
            "Negative prompt": "Free-text list of things to suppress (e.g., \"motion blur, warped hands, on-screen text\") — the primary lever for cleaning up Veo's known artifacts.",
            "Audio": "Toggles native synchronised audio generation; turning it off cuts cost roughly in half ($0.20 vs $0.40 per second at 1080p) but you lose Veo 3.1's signature joint-diffusion soundtrack.",
            "Seed": "A 32-bit integer that makes generation reproducible — reuse the same seed plus prompt to get consistent results when iterating on small prompt changes.",
            "Provider options JSON": "An escape hatch for passing raw provider-specific fields to the Vertex/Gemini backend that aren't surfaced as dedicated valves.",
            "Person generation": "Safety control for human subjects; \"allow_adult\" (default) permits adult faces and bodies, while \"dont_allow\" refuses any people/faces.",
            "Conditioning scale": "Adjusts how strictly Veo 3.1 follows the prompt versus exploring creatively — exposed as a passthrough but documented behaviour may be provider-internal.",
            "Enhance prompt": "Asks the backend to auto-rewrite your prompt for richer cinematic detail; per Google's Vertex docs this flag is officially Veo 2-only, so on Veo 3.1 it may be a no-op.",
        },
    },
    "kwaivgi/kling-video-o1": {
        "display_name": "Kling: Video O1",
        "best_known_for": (
            "Kuaishou's Kling Video O1 is best known for cinematic, film-grade output with "
            "strong character and subject consistency, \"director-like memory\" that locks "
            "identities across shots, and physics-aware human motion (weight, momentum, "
            "fabric, water) that holds up better than most peers. It shines at reference-"
            "driven workflows — mixing characters/props across multi-shot sequences — and "
            "at previsualisation, marketing assets, and short narrative clips where camera "
            "language (tracking, push-in, aerial) matters."
        ),
        "tips_and_pitfalls": [
            "Write like a director, not a tagger: lead with camera (wide / slow dolly-in / tracking) and motivate the camera move narratively — Kling responds to cinematic intent more than object lists.",
            "Be explicit about motion physics and end state: describe how a body or fabric moves and how the shot resolves; vague motion or missing end-states cause stalls and rubbery limbs.",
            "Use the negative prompt as guardrails (e.g. \"blurry text, extra fingers, warped face\") rather than burying don'ts in the main prompt — Kling honours negatives well.",
            "No seed control is exposed, so don't expect bit-exact repeats; lock look via reference frames (first/last) and tight prompt language instead, and avoid on-screen text (Kling renders text poorly).",
        ],
        "knob_descriptions": {
            "Duration": "Picks clip length in seconds — Kling O1 only accepts 5s or 10s, and pricing scales linearly per second.",
            "Aspect ratio": "Chooses the frame shape (16:9 landscape, 9:16 vertical, 1:1 square) to match the platform you're delivering to.",
            "Resolution": "Sets output quality tier; Kling O1 currently outputs only 720p, so this is effectively fixed.",
            "Size": "Selects the exact pixel dimensions tied to your aspect (1280×720, 720×1280, or 720×720); usually leave on auto so it follows the aspect ratio.",
            "Frames": "Lets you pin a first_frame and/or last_frame image to anchor the opening or closing pose, useful for continuity across shots or for image-to-video starts.",
            "Negative prompt": "Free-text list of things to avoid (artifacts, distorted faces, text, unwanted styles); Kling treats this as hard guardrails and it's the main quality lever here.",
            "Audio": "Toggles Kling's native audio generation (ambient sound / effects) along with the video — turn off if you plan to score the clip externally.",
            "Provider options JSON": "Escape hatch for raw OpenRouter/Kling provider parameters not surfaced as valves; leave empty unless OpenRouter docs call out a specific override you need.",
        },
    },
    "minimax/hailuo-2.3": {
        "display_name": "MiniMax: Hailuo 2.3",
        "best_known_for": (
            "MiniMax's flagship video model, best known for state-of-the-art human physics "
            "and character motion — fluid full-body choreography, accurate limb tracking, "
            "and lifelike facial micro-expressions that read as genuine emotion rather "
            "than uncanny animation. It excels at physics-heavy dynamics (rigid body, "
            "fluids, cloth, fire) where most rivals fall apart, and renders cinematic "
            "1080p output with strong camera control and stylisation across photoreal, "
            "anime, illustration, and ink-wash looks."
        ),
        "tips_and_pitfalls": [
            "No audio: Hailuo 2.3 is silent — you must add dialogue, SFX, and music in post; the catalog confirms generate_audio=false.",
            "First-frame only: 2.3 dropped last-frame conditioning that 2.0 had, so you can anchor the opening still but cannot pin the ending — plan motion to flow forward from the first frame.",
            "Hailuo rewards specific physical and emotional direction (e.g. \"tight smile turning to laughter,\" \"cloth catches the wind, then settles\") far more than other models — vague prompts under-use its physics strengths.",
            "prompt_optimizer rewrites/expands your prompt for better adherence; turn it off only when you have a deliberately precise prompt. fast_pretreatment speeds up that step at a small quality cost — useful for batch runs, otherwise leave at default.",
        ],
        "knob_descriptions": {
            "Duration": "Length of the generated clip in seconds; Hailuo 2.3 supports either 6s or 10s.",
            "Aspect ratio": "Frame shape of the output; this model is locked to 16:9 widescreen.",
            "Resolution": "Vertical pixel count of the render; this model only outputs 1080p (full HD).",
            "Size": "Exact pixel dimensions of the output frame; fixed at 1920×1080.",
            "Frames": "Optional reference images; Hailuo 2.3 accepts only a first_frame image to anchor the opening shot and does not support a last frame.",
            "Provider options JSON": "Free-form passthrough for OpenRouter provider routing (not for video parameters themselves).",
            "Prompt optimizer": "Enables MiniMax's server-side prompt rewriter that expands and refines your prompt for better motion and adherence; leave on for short or casual prompts, set off for verbatim.",
            "Fast pretreatment": "Only meaningful when the prompt optimiser is active — runs a quicker, lighter optimisation pass to cut latency (handy for batch generation) at a small loss of fine-tuning quality.",
        },
    },
    "alibaba/wan-2.7": {
        "display_name": "Alibaba: Wan 2.7",
        "best_known_for": (
            "Alibaba Tongyi Lab's flagship multimodal video model, unifying text, image, "
            "audio, and video conditioning in a single 27B-parameter Diffusion Transformer "
            "with Flow Matching. Its standout capability is true multi-reference control: "
            "lock subject identity, vocal timbre, props, and visual style across new "
            "scenes by feeding up to five reference videos plus reference image grids. "
            "It also adds last-frame anchoring (FLF2V), native audio-synced lip generation "
            "across languages, and a \"Thinking Mode\" planner that improves coherence on "
            "dialogue- and character-led shots — at the cost of weaker fast-motion physics "
            "than Seedance 2.0."
        ),
        "tips_and_pitfalls": [
            "Pick the right reference channel: use the images array to lock appearance, wardrobe, props (it works like a 9-image storyboard grid in 2.7); use the video / videos passthrough for motion style, camera language, or vocal timbre transfer.",
            "For talking-head and dialogue clips, supply an audio reference — Wan 2.7's automatic lip-sync matches mouth shapes to the supplied speech in the target language, a headline upgrade over 2.6.",
            "Wan 2.7 is tuned for character-led, narrative content; for fast sports/action shots its physics still trails Seedance 2.0 and Runway Gen-4, so add explicit motion verbs and a negative prompt against blur/morphing.",
            "Wan 2.7's instruction-following changed vs 2.6, so prompts calibrated on 2.6 may drift; lean on prompt_extend = on when prompts are short, but turn it off when you've already written a precise multi-shot storyboard.",
        ],
        "knob_descriptions": {
            "Duration": "Sets clip length in seconds (2–10 here); longer durations let Wan 2.7's full-attention DiT carry character identity further, but costs scale linearly at $0.10/sec.",
            "Aspect ratio": "Picks the canvas shape (16:9, 9:16, 1:1, 4:3, 3:4); 9:16 is the right choice for the talking-head / lip-sync workflows Wan 2.7 is tuned for.",
            "Resolution": "Selects 720p or 1080p output; 1080p is the model's native ceiling — there is no 4K, so upscale in post if you need it.",
            "Size": "Forces an explicit pixel size (e.g. 1920×1080, 1440×1080); use this when you need a specific frame size that the aspect-ratio preset doesn't expose.",
            "Frames": "Lets you pin a first_frame and/or last_frame image so Wan 2.7 interpolates the motion between your two keyframes — this is the FLF2V control added in 2.7.",
            "Negative prompt": "Free-text list of things to suppress (e.g. \"blurry, extra fingers, morphing\"); useful on Wan 2.7 to push back on residual fast-motion artefacts.",
            "Audio": "Toggles native audio generation — Wan 2.7 bakes synchronised speech, ambience, and effects into the clip rather than dubbing them in post.",
            "Seed": "Fixes the RNG so the same prompt + references reproduce the same clip; essential when iterating on multi-shot sequences that need to match.",
            "Provider options JSON": "Raw escape hatch for any OpenRouter provider parameter not surfaced as a dedicated valve; useful for niche flags Alibaba may add post-launch.",
            "Audio reference URL": "Sends an audio clip in the audio passthrough so Wan 2.7 conditions the character's voice timbre and lip motion on your reference — the basis for its multi-language lip-sync feature.",
            "Last image URL": "Convenience field for the last_image passthrough that anchors the closing frame, used together with first-frame to compose a controlled in-between motion arc.",
            "Reference video URL": "Sends a single video to the video passthrough for reference-to-video (Wan2.7-r2v); the model copies motion style, camera moves, and/or vocal identity from this clip into the new generation.",
            "Reference videos JSON": "Array form of the videos passthrough — Wan 2.7 accepts up to five reference videos in one call to lock multiple distinct characters' appearance and voice across the same scene.",
            "Reference images JSON": "Array form of the images passthrough — feeds Wan 2.7's 9-image structured grid that anchors subject identity, wardrobe, props, and environment without you having to describe them in text.",
            "Prompt extend": "Selects whether Wan's prompt rewriter expands your text (on), leaves it untouched (off), or uses the model's default; turn it off when you've already written a precise multi-shot storyboard.",
            "Ratio": "String passthrough that sends ratio directly to OpenRouter; use this when you need a non-standard aspect string the dropdown doesn't expose, otherwise prefer the Aspect ratio valve.",
        },
    },
    "bytedance/seedance-2.0-fast": {
        "display_name": "ByteDance: Seedance 2.0 Fast",
        "best_known_for": (
            "ByteDance's speed-and-cost-optimised variant of the Seedance 2.0 family, "
            "built on the same unified multimodal architecture but using distillation and "
            "accelerated sampling to cut generation time at roughly 30–33% lower cost than "
            "standard Seedance 2.0. Best known for cinematic 480p/720p output with native "
            "audio synchronised in a single pass, support for text-to-video, image-to-"
            "video with first/last frame control, and multimodal reference-to-video, plus "
            "very wide aspect-ratio coverage including 21:9 cinematic and 9:21. OpenRouter "
            "bills it via video tokens, so cost scales with pixels × seconds."
        ),
        "tips_and_pitfalls": [
            "Use Fast for drafting, prompt iteration, and bulk pipelines; switch to standard Seedance 2.0 for hero shots — Fast trades a small amount of motion refinement and detail for ~33% lower cost.",
            "Token math means doubling resolution or duration roughly multiplies cost; a 720p 10s clip costs far more than a 480p 5s draft, so iterate small first.",
            "This OpenRouter listing does not expose negative_prompt and caps at 720p — for 1080p or text-prompted negatives you need the standard 2.0 model or another provider.",
            "watermark toggles the visible provider/ByteDance branding overlay on the returned MP4, and req_key is ByteDance/Volcengine ModelArk's internal model-routing identifier — leave both at defaults unless your provider explicitly tells you otherwise.",
        ],
        "knob_descriptions": {
            "Duration": "Length of the generated clip in whole seconds; Seedance 2.0 Fast accepts any integer from 4 to 15s and longer durations linearly increase token cost.",
            "Aspect ratio": "Picks the frame shape from seven options including ultrawide cinematic 21:9, vertical 9:16/9:21, square 1:1, and standard 16:9/4:3/3:4 — wider than most peers in this tier.",
            "Resolution": "Selects 480p (cheapest, good for drafts) or 720p (final-quality tier on Fast); 1080p is not offered on the Fast variant on OpenRouter.",
            "Size": "Locks an exact pixel resolution (e.g. 1280×720, 854×480, 720×1680) overriding aspect/resolution when you need a specific output canvas.",
            "Frames": "Lets you pin the first and/or last frame of the video to a supplied image for image-to-video or precise start/end control of motion.",
            "Audio": "When on (model default true), Seedance generates synchronised dialogue, ambient sound, and music in the same pass as the video — no second audio model required.",
            "Seed": "Integer that makes generations reproducible — same prompt + seed yields the same clip, useful for A/B testing prompt edits.",
            "Provider options JSON": "Free-form passthrough block forwarded to OpenRouter's provider field for routing/fallback control; not for model parameters.",
            "Watermark": "Controls whether the upstream provider stamps a visible watermark on the returned MP4 — model_default keeps the provider's policy, on forces the watermark, off requests an unwatermarked clip (subject to provider permission/billing).",
            "Req key": "A ByteDance/Volcengine ModelArk routing string identifying which Seedance SKU/endpoint variant to dispatch to upstream; leave blank to use OpenRouter's default mapping.",
        },
    },
    "bytedance/seedance-2.0": {
        "display_name": "ByteDance: Seedance 2.0",
        "best_known_for": (
            "ByteDance's flagship multimodal video model, best known for \"locked\" "
            "character consistency — preserving faces, clothing, accessories, and small "
            "subject details across the duration of a clip and across multi-shot "
            "generations. It stands out for its \"Universal Reference\" system that "
            "accepts text plus up to 9 images and 3 video/audio clips in a single "
            "generation, letting you direct composition, camera movement, and character "
            "actions from reference assets at once. Unlike 2.0 Fast (speed over quality) "
            "and 1.5 Pro (limited to text + first/last frame), the full 2.0 variant is "
            "the production-quality choice with native audio (dialogue, ambience, SFX) "
            "and multi-shot story coherence."
        ),
        "tips_and_pitfalls": [
            "Reach for full Seedance 2.0 (not Fast) when you need production drafts where identity preservation matters — branded characters, story-led scenes, or repeatable creative formats — and accept the longer render in exchange for tighter facial/clothing fidelity.",
            "Long durations (12–15s) still drift more than short ones; for the most stable identity, anchor with a reference image AND a clear text description of the subject, and prefer 6–10s clips for hero shots.",
            "The multimodal reference workflow is the headline feature — use images for style/identity, video clips for motion/camera language, audio for pacing — but keep references coherent; conflicting references degrade consistency more than helping it.",
            "ByteDance gates real-person reference features and identity verification due to IP/likeness concerns; expect occasional refusals on celebrity or copyrighted-character prompts.",
        ],
        "knob_descriptions": {
            "Duration": "Sets clip length from 4 to 15 seconds; longer durations cost more tokens and increase the chance of subtle character or scene drift, so pick the shortest length that tells the shot.",
            "Aspect ratio": "Chooses the framing from seven options (1:1, 3:4, 9:16, 4:3, 16:9, 21:9, 9:21), useful for matching social, cinematic, or vertical-mobile delivery before sizing.",
            "Resolution": "Selects 480p, 720p, or 1080p output — 1080p is available on the full 2.0 variant and is the recommended choice for client-facing drafts where character detail matters.",
            "Size": "Locks the exact pixel dimensions from the supported list (e.g. 1920×1080, 1080×1920, 2520×1080) when you need a specific frame size rather than just an aspect ratio.",
            "Frames": "Lets you supply a first_frame and/or last_frame image to anchor the clip's start and end, the most reliable way to enforce character/scene continuity on this model.",
            "Audio": "Toggles native audio generation (dialogue, ambient sound, SFX) — Seedance 2.0 has phoneme-level lip-sync, so leave on for finished drafts and off only for silent B-roll.",
            "Seed": "Locks the random seed for reproducible output, letting you re-run the same prompt and references to get a near-identical clip for iteration or A/B comparison.",
            "Provider options JSON": "Forwards advanced ByteDance/provider parameters not surfaced as dedicated knobs (e.g. reference-mode flags, multi-reference weighting), for power users following provider docs.",
            "Watermark": "Per-model passthrough that toggles ByteDance's visible video watermark on the output; turn off only if your provider account permits unwatermarked delivery.",
            "Req key": "Per-model passthrough for an optional request/idempotency key forwarded to ByteDance, useful for tracing or de-duplicating long-running video jobs.",
        },
    },
    "alibaba/wan-2.6": {
        "display_name": "Alibaba: Wan 2.6",
        "best_known_for": (
            "Alibaba's most feature-rich video generation model (Dec 2025), supporting "
            "10+ unified visual creation capabilities (text-to-video, image-to-video, "
            "reference-to-video, voiceover, action generation, role-play, editing) on a "
            "14B-parameter MoE architecture. Best known for affordable multi-shot 1080p "
            "@ 24fps generation with synchronised native audio (multi-speaker dialogue, "
            "lip-sync, voice/music conditioning) and intelligent multi-shot narrative "
            "storyboarding that holds character and lighting consistency across cuts. "
            "Pick 2.6 over 2.7 when you want the cheaper, well-tuned generation pipeline; "
            "pick 2.7 only if you specifically need last-frame control, 9-grid input, "
            "instruction-based editing, or stronger physics."
        ),
        "tips_and_pitfalls": [
            "First-frame ONLY: Wan 2.6 supports first_frame image conditioning but has no last_frame. To define both endpoints of a clip, you must upgrade to Wan 2.7 — don't try to fake it through prompts.",
            "shot_type controls camera framing/composition (e.g., values like \"medium_to_closeup\"), used for cinematic shot intent; this knob was removed in 2.7. For multi-shot scripts, write scene-timed segments in the prompt itself.",
            "Use enable_prompt_expansion (LLM-based prompt rewriter) for short or terse prompts — it adds cinematographic detail \"for free\" without consuming your budget; turn it OFF when you've already crafted a long, precise prompt.",
            "Audio reference files must be 3–30s, WAV/MP3, max 15 MB; clips longer than the video get truncated and shorter clips leave a silent tail. Two-speaker dialogue tends to collapse to one dominant voice — generate single-speaker clips and composite.",
        ],
        "knob_descriptions": {
            "Duration": "Selects 5s or 10s of video (Wan 2.6 OpenRouter SKU caps at 10s; the 15s tier from Alibaba Cloud is not exposed here).",
            "Aspect ratio": "Picks 16:9 (landscape) or 9:16 (portrait/vertical) framing for the output clip.",
            "Resolution": "Chooses 720p or 1080p; 1080p costs roughly 50% more per second and is the model's native high-fidelity tier.",
            "Size": "Direct pixel dimensions (1280×720, 1920×1080, 720×1280, 1080×1920) — overrides aspect/resolution if you need an exact frame size.",
            "Frames": "Attaches a first-frame reference image to anchor the opening shot; Wan 2.6 has no last-frame slot.",
            "Negative prompt": "Free-text list of things to avoid (artifacts, styles, objects, motion) — passed through to suppress unwanted features in the render.",
            "Audio": "Toggles Wan 2.6's native A/V synthesis so the output clip ships with synchronised sound effects, ambience, dialogue, or voiceover instead of a silent video.",
            "Seed": "Integer that fixes the random initialisation for reproducible/iterative generations from the same prompt.",
            "Provider options JSON": "Escape hatch for any extra OpenRouter/Alibaba passthrough field not surfaced as a dedicated valve.",
            "Audio reference URL": "Public WAV/MP3 link (3–30s, ≤15 MB) that Wan 2.6 will lip-sync or musically conform the video to instead of generating audio from scratch.",
            "Enable prompt expansion": "Tri-state (model_default / on / off) for the LLM prompt-rewriter that auto-enriches short prompts with cinematographic detail; turn off for deterministic, fully-authored prompts.",
            "Shot type": "String hint controlling camera framing/composition (e.g., wide, medium, close-up, \"medium_to_closeup\") so Wan 2.6 picks the intended cinematographic shot — a 2.6-only knob.",
        },
    },
    "bytedance/seedance-1-5-pro": {
        "display_name": "ByteDance: Seedance 1.5 Pro",
        "best_known_for": (
            "ByteDance's first foundation model to natively generate video and audio in a "
            "single unified pass, using a 4.5B-parameter Dual-Branch Diffusion Transformer "
            "with a cross-modal joint module that locks phonemes to visemes and physics "
            "events to audio spikes at millisecond precision. Pick 1.5 Pro over Seedance "
            "2.0 when you want the older, production-validated audio-visual workflow at "
            "materially lower cost; it offers 1080p output, the wider 4–12s duration "
            "window, and reliable multilingual lip-sync (Mandarin, English, Japanese, "
            "Korean, Spanish, plus dialects)."
        ),
        "tips_and_pitfalls": [
            "Audio doubles the bill: video_tokens with audio is ~2× without audio, so toggle Audio off for silent B-roll, layout passes, or anything you'll dub later.",
            "Use 1.5 Pro for short, repeatable clips with simple camera work and known-good prompts; switch to 2.0 only when you need richer multimodal references, 2K output, or longer 15s shots — 1.5 Pro caps at 1080p and 12s.",
            "Long durations drift: 4–6s clips stay on-model, but 10–12s shots show face drift, color shift, and continuity errors — chain shorter shots with last_frame anchors and consistent character descriptions.",
            "last_frame is a directional guide, not a pixel-perfect target — pick an end frame with framing and lighting close to the start frame, or you'll get jumpy transitions in the final second.",
        ],
        "knob_descriptions": {
            "Duration": "Sets clip length from 4–12 seconds; cost scales linearly and quality/continuity degrade past ~8s, so iterate short and only extend after motion looks right.",
            "Aspect ratio": "Picks one of seven framings (1:1, 3:4, 9:16, 9:21, 4:3, 16:9, 21:9) and should match your input image orientation to avoid awkward crops or stretched motion.",
            "Resolution": "Chooses 480p (fast previews), 720p (balanced), or native 1080p (final delivery); 1.5 Pro does not offer 2K, unlike Seedance 2.0.",
            "Size": "Selects from 21 exact pixel dimensions — the widest size matrix of any video model on OpenRouter — so you can hit platform-specific targets without post-crop.",
            "Frames": "Accepts a first_frame to lock identity/lighting and an optional last_frame to steer the ending, enabling match cuts and multi-shot continuity when you chain clips.",
            "Audio": "Turns on the dual-branch joint generation so lip-sync and physics SFX are produced in the same pass; doubles the per-token price, so disable when you don't need sound.",
            "Seed": "Fixes the random initialisation for reproducible outputs — essential when iterating on prompt wording without re-rolling the whole scene.",
            "Provider options JSON": "Free-form passthrough for any extra ByteDance fields not mapped to a dedicated valve; useful for experimental flags surfaced in OpenRouter's video API.",
            "Watermark": "Per-model passthrough that toggles the visible ByteDance watermark on the rendered output.",
            "Req key": "Per-model passthrough idempotency/request token forwarded to ByteDance — set a stable value to dedupe retries on the provider side.",
        },
    },
    "openai/sora-2-pro": {
        "display_name": "OpenAI: Sora 2 Pro",
        "best_known_for": (
            "OpenAI's flagship video model, best known for physics-accurate motion "
            "(gravity, momentum, fluid dynamics, object permanence — e.g. a missed "
            "basketball realistically rebounds off the backboard) paired with natively "
            "synchronised audio: dialogue, sound effects, and ambient audio are predicted "
            "alongside the frames rather than dubbed in, so footsteps land on the correct "
            "frame and lip-sync stays tight. Its standout differentiator is world-state "
            "persistence across multi-shot sequences — characters, props, and spatial "
            "relationships stay consistent across cuts, enabling cohesive short-form "
            "storytelling. In this catalog it also offers the longest clips of any video "
            "model, up to 20 seconds at full 1080p."
        ),
        "tips_and_pitfalls": [
            "Text-to-video only here: this catalog entry has no frame_image support, so you can't seed it with a start/end image — drive the result entirely from prompt language.",
            "20-second durations and 1080p are unique strengths but render slow — community tests report 2–5 minutes for a 20s clip and much longer at peak, so prefer 4–8s 720p for iteration and reserve 16–20s 1080p for finals.",
            "Plays to its strengths on physics, motion weight, lighting, and ambient/dialogue audio; struggles with on-screen text, brand logos, fine hand details, and highly choreographed multi-character action — don't ship as-is for client deliverables that depend on legible text.",
            "Quality and Style are passthrough hints OpenRouter forwards; the OpenAI Videos API itself doesn't expose a discrete quality enum (resolution drives the tier), so treat them as soft hints rather than guaranteed switches.",
        ],
        "knob_descriptions": {
            "Duration": "Pick clip length in seconds from 4/8/12/16/20 — Sora 2 Pro is the only model in this catalog that reaches 20s, but render time and cost scale roughly linearly with duration.",
            "Aspect ratio": "Choose 16:9 for landscape/cinematic framing or 9:16 for vertical/social; this model does not support 1:1 or other ratios.",
            "Resolution": "720p is the cheap iteration tier ($0.30/s) while 1080p is the cinematic finishing tier ($0.50/s) with sharper textures and richer color depth at the cost of longer renders.",
            "Size": "Picks the exact pixel dimensions (1280×720, 1920×1080, 720×1280, 1080×1920) — use this when your downstream pipeline needs a specific frame size rather than just a ratio.",
            "Audio": "Sora 2 Pro generates synchronised audio natively (dialogue, SFX, ambience) from the same scene representation as the video — leave it on for realistic results.",
            "Provider options JSON": "Free-form JSON forwarded to OpenRouter for advanced/experimental fields not covered by the dedicated valves; leave empty unless you're following specific OpenRouter or OpenAI Videos API docs.",
            "Quality": "Passthrough hint (\"standard\" or \"hd\") that maps loosely to OpenAI's quality tier; in practice the visible quality tier is governed mainly by the chosen resolution, so this is a secondary nudge.",
            "Style": "Free-text passthrough that lets you bias the look (e.g. \"cinematic\", \"anamorphic\", \"documentary handheld\"); since the OpenAI API has no formal style enum, prompt language remains the primary stylistic lever.",
        },
    },
}


VIDEO_HELP_BY_MODEL = _PER_MODEL_HELP_DATA


_KNOB_GATE: dict[str, str | None] = {
    "Duration": None,
    "Aspect ratio": None,
    "Resolution": None,
    "Size": None,
    "Frames": None,
    "Negative prompt": "negative_prompt_or_camelcase",
    "Audio": "generate_audio_top_level",
    "Seed": "seed_top_level",
    "Provider options JSON": None,
    "Audio reference URL": "audio",
    "Last image URL": "last_image",
    "Reference video URL": "video",
    "Reference videos JSON": "videos",
    "Reference images JSON": "images",
    "Person generation": "personGeneration",
    "Conditioning scale": "conditioningScale",
    "Enhance prompt": "enhancePrompt",
    "Prompt optimizer": "prompt_optimizer",
    "Fast pretreatment": "fast_pretreatment",
    "Prompt extend": "prompt_extend",
    "Ratio": "ratio",
    "Enable prompt expansion": "enable_prompt_expansion",
    "Shot type": "shot_type",
    "Watermark": "watermark",
    "Req key": "req_key",
    "Quality": "quality",
    "Style": "style",
}


def _knob_is_active(knob: str, model: dict[str, Any]) -> bool:
    gate = _KNOB_GATE.get(knob)
    allowed = model.get("allowed_passthrough_parameters") or []
    if gate is None:
        return True
    if gate == "negative_prompt_or_camelcase":
        return any(p in allowed for p in ("negative_prompt", "negativePrompt"))
    if gate == "generate_audio_top_level":
        return bool(model.get("generate_audio"))
    if gate == "seed_top_level":
        return model.get("seed") is True
    return gate in allowed


_SKU_MODIFIERS: tuple[tuple[str, str], ...] = (
    ("with_audio_4k", "with audio, 4K"),
    ("without_audio_4k", "without audio, 4K"),
    ("with_audio_720p", "with audio, 720p"),
    ("without_audio_720p", "without audio, 720p"),
    ("with_audio", "with audio"),
    ("without_audio", "without audio"),
    ("text_to_video", "text-to-video"),
    ("image_to_video", "image-to-video"),
)

_SKU_BASE_LABELS: dict[str, str] = {
    "duration_seconds": "per second",
    "video_tokens": "per video token",
}

_SKU_RESOLUTION_TAGS: tuple[str, ...] = ("480p", "720p", "1024p", "1080p", "2k", "4k")


def _format_pricing_skus(pricing_skus: dict[str, str] | None) -> str:
    if not isinstance(pricing_skus, dict) or not pricing_skus:
        return ""
    bullets: list[tuple[str, str]] = []
    for raw_key in sorted(pricing_skus):
        raw_value = pricing_skus.get(raw_key)
        if not isinstance(raw_value, (int, float, str)) or str(raw_value).strip() == "":
            continue
        unit = _format_sku_unit(raw_key)
        value = str(raw_value).strip()
        bullets.append((raw_key, f"- {unit}: ${value}"))
    if not bullets:
        return ""
    bullets.sort(key=lambda pair: pair[0])
    return "\n".join(line for _, line in bullets)


def _format_sku_unit(raw_key: str) -> str:
    key = (raw_key or "").strip().lower()
    if not key:
        return "per unit"
    remainder = key
    modifiers: list[str] = []
    for token, label in _SKU_MODIFIERS:
        if token in remainder:
            modifiers.append(label)
            remainder = remainder.replace(token, "_").strip("_")
            while "__" in remainder:
                remainder = remainder.replace("__", "_")
    for tag in _SKU_RESOLUTION_TAGS:
        if remainder.endswith("_" + tag) or remainder == tag:
            modifiers.append(tag.upper() if tag.endswith("k") else tag)
            if remainder == tag:
                remainder = ""
            else:
                remainder = remainder[: -(len(tag) + 1)].rstrip("_")
            break
    base_label = ""
    for base_token, label in _SKU_BASE_LABELS.items():
        if remainder == base_token or remainder.startswith(base_token):
            base_label = label
            remainder = remainder[len(base_token):].lstrip("_")
            break
    if not base_label and remainder:
        base_label = "per " + remainder.replace("_", " ")
        remainder = ""
    elif remainder:
        modifiers.append(remainder.replace("_", " "))
    if modifiers:
        return f"{base_label} ({', '.join(modifiers)})" if base_label else "per " + ", ".join(modifiers)
    return base_label or "per unit"


def _format_csv(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    items = [str(item).strip() for item in value if str(item).strip()]
    return ", ".join(items)


def _format_frames_capability(supported_frames: Any) -> str:
    csv = _format_csv(supported_frames)
    if not csv:
        return "none (text-only)"
    if "first_frame" in csv and "last_frame" in csv:
        return "first_frame and last_frame"
    return csv


def _yes_no(value: Any) -> str:
    if value is True:
        return "yes"
    return "no"


def _render_template(model_id: str, model: dict[str, Any], data: dict[str, Any]) -> str:
    display_name = data.get("display_name") or model_id
    durations = _format_csv(model.get("supported_durations")) or "model default"
    aspects = _format_csv(model.get("supported_aspect_ratios")) or "model default"
    resolutions = _format_csv(model.get("supported_resolutions")) or "model default"
    frames = _format_frames_capability(model.get("supported_frame_images"))
    audio = _yes_no(model.get("generate_audio"))
    seed = _yes_no(model.get("seed") is True)

    knob_lines: list[str] = []
    for knob, description in data.get("knob_descriptions", {}).items():
        if not _knob_is_active(knob, model):
            continue
        knob_lines.append(f"- `{knob}`: {description}")

    pricing_block = _format_pricing_skus(model.get("pricing_skus") or {})
    pricing_section = ""
    if pricing_block:
        pricing_section = (
            "\n\n**Cost** (live from OpenRouter catalog `pricing_skus`)\n"
            f"{pricing_block}\n\n"
            "The final status line shows the actual usage cost from the "
            "OpenRouter poll response when generation completes."
        )

    tips = data.get("tips_and_pitfalls") or []
    tip_lines = "\n".join(f"- {bullet}" for bullet in tips)

    knobs_section = ""
    if knob_lines:
        knobs_section = "\n\n**Knobs in this filter**\n" + "\n".join(knob_lines)

    tips_section = ""
    if tip_lines:
        tips_section = "\n\n**Tips & pitfalls**\n" + tip_lines

    return (
        f"### {display_name}\n\n"
        f"{data['best_known_for']}\n\n"
        "**Output capabilities**\n"
        f"- Durations: {durations}\n"
        f"- Aspect ratios: {aspects}\n"
        f"- Resolutions: {resolutions}\n"
        f"- Frame controls: {frames}\n"
        f"- Generated audio: {audio}\n"
        f"- Deterministic seed: {seed}"
        f"{knobs_section}"
        f"{tips_section}"
        f"{pricing_section}"
    )


def render_video_help(model_id: str, video_model: dict[str, Any] | None = None) -> str:
    model = video_model if isinstance(video_model, dict) else {}
    canonical_id = _canonical_model_id(model_id, model)
    data = _PER_MODEL_HELP_DATA.get(canonical_id)
    if data:
        return _render_template(canonical_id, model, data)
    return _render_catalog_fallback(canonical_id, model)


def _canonical_model_id(model_id: str, model: dict[str, Any]) -> str:
    raw = model.get("id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(model_id, str) and model_id.strip():
        stripped = model_id.strip()
        if "/" in stripped:
            return stripped
        if "." in stripped:
            provider, name = stripped.split(".", 1)
            return f"{provider}/{name}"
        return stripped
    return ""


def _render_catalog_fallback(model_id: str, model: dict[str, Any]) -> str:
    raw_name = model.get("name")
    display = raw_name if isinstance(raw_name, str) else model_id
    raw_description = model.get("description")
    description = raw_description if isinstance(raw_description, str) else ""
    frames = _format_frames_capability(model.get("supported_frame_images"))
    params = _format_csv(model.get("allowed_passthrough_parameters")) or "none listed"
    ratios = _format_csv(model.get("supported_aspect_ratios")) or "model default"
    durations = _format_csv(model.get("supported_durations")) or "model default"
    resolutions = _format_csv(model.get("supported_resolutions")) or "model default"
    pricing_block = _format_pricing_skus(model.get("pricing_skus") or {})
    pricing_section = ""
    if pricing_block:
        pricing_section = (
            "\n\n**Cost** (live from OpenRouter catalog `pricing_skus`)\n"
            f"{pricing_block}"
        )
    return (
        f"### {display}\n\n"
        f"Capability: {description.strip() or 'OpenRouter video generation model.'}\n\n"
        f"Accepted inputs: {frames}.\n\n"
        "Useful prompt patterns: Describe subject, action, setting, camera "
        "movement, visual style, and constraints in one clear shot.\n\n"
        "Known limitations: This model is not ZDR-capable and exact continuity "
        "can vary by generation.\n\n"
        f"Supported knobs: durations {durations}; aspect ratios {ratios}; "
        f"resolutions {resolutions}; provider parameters {params}."
        f"{pricing_section}"
    )
