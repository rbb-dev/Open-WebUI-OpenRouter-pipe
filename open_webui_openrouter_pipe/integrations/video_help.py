from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from .image_help import OPENROUTER_PRICING
from .image_types import PROVIDER_OPTIONS_DESCRIPTION
from .video_types import VIDEO_REQ_KEY_DESCRIPTION

if TYPE_CHECKING:
    from ..filters.video_filter_renderer import VideoFilterSpec

TRUE_WAN_REFERENCES = (
    "lock subject identity, props and visual style across new scenes by feeding a grid of reference images. It also adds last-frame anchoring. Alibaba describes reference clips and voice conditioning for this model, but OpenRouter reports it as taking only text and pictures, so neither is offered here"
)

_PER_MODEL_HELP_DATA: dict[str, dict[str, Any]] = {
    "black-forest-labs/flux-3-video": {
        "display_name": "Black Forest Labs: FLUX.3 Video",
        "best_known_for": (
            "Black Forest Labs' FLUX.3 Video, a text- and image-to-video model built around "
            "controlled, keyframe-driven shots. You can hand it an opening still, a closing still, "
            "or both, and it fills in the motion between them. Clips run 5 to 20 seconds at 720p or "
            "1080p, across six framings from ultrawide 21:9 through to vertical 9:16, and it "
            "generates audio with the picture. There is no seed, so two runs of the same prompt "
            "will differ."
        ),
        "tips_and_pitfalls": [
            "Anchor both ends when you know where the shot should finish — a closing frame is what separates this from a model you can only point at a starting still.",
            "Build a long sequence a segment at a time: generate a shot, then use its closing still as the next shot's opening frame. Each segment stays sharper than one very long request, and you can stop and redirect between them.",
            "There is no seed here, so an idea you like cannot be re-rolled exactly — save the clip you want before iterating on the prompt.",
            "Continuing a clip you attach needs your administrator to have turned on sending media to a file host — without that the clip is left out and you get a fresh generation from the prompt alone.",
            "Audio is generated alongside the picture, so it is worth describing the sound you want rather than leaving it to chance.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, 5 to 20.",
            "Aspect ratio": "Framing, from ultrawide 21:9 through 16:9, 4:3, 1:1 and 3:4 to vertical 9:16.",
            "Resolution": "720p or 1080p — the detail tier the clip is rendered at.",
            "Frames": "Which supplied stills anchor the shot: none for pure text-to-video, first_only to animate from an opening still, or first_last to fix both ends and let the model fill the middle.",
            "Audio": "Whether a soundtrack is generated with the picture.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "bytedance/seedance-2.5": {
        "display_name": "ByteDance: Seedance 2.5",
        "best_known_for": (
            "ByteDance's Seedance 2.5, the long-form member of the Seedance family. It runs to 30 "
            "seconds in a single clip — twice the length most video models will give you — and is "
            "aimed at storytelling that has to hold together across that span: reference-driven "
            "generation, editing an existing clip, and extending one that already exists — the "
            "last two need a clip attached, which only reaches the model if your administrator has turned on sending media to a file host; otherwise it is left out with a note. It takes "
            "an opening still, a closing still, or both, offers six framings including ultrawide "
            "21:9, generates audio, and honours a seed, so a take you like can be reproduced and "
            "then adjusted a line at a time. Output is 480p or 720p, with twelve exact canvas sizes "
            "if you need to pin dimensions rather than pick a ratio."
        ),
        "tips_and_pitfalls": [
            "The 30-second ceiling is the reason to pick this model; if your shot is under 10 seconds another model will usually cover it.",
            "Lock a seed before you start refining — over a half-minute clip, an unseeded re-roll changes far more than the line you edited.",
            "Long clips reward one continuous action described plainly over a list of cuts; ask for a scene, not a sequence of shots.",
            "Use a closing still when the clip has to land somewhere specific, such as a product in frame or a logo settled in place.",
            OPENROUTER_PRICING,
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, 4 to 30 — the longest single take in the catalogue.",
            "Aspect ratio": "16:9, 4:3, 1:1, 3:4, 9:16, or ultrawide 21:9.",
            "Resolution": "480p or 720p.",
            "Size": "Pins exact pixel dimensions from the twelve this model publishes, instead of letting ratio and resolution decide.",
            "Frames": "Which supplied stills anchor the clip: none, first_only, or first_last to fix both ends.",
            "Audio": "Whether a soundtrack is generated with the picture.",
            "Seed": "Fixes the random draw so the same prompt and seed reproduce the same clip — worth setting before you iterate.",
            "Watermark": "Whether the provider's visible branding overlay is burned into the output.",
            "Request key": VIDEO_REQ_KEY_DESCRIPTION,
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "minimax/hailuo-3": {
        "display_name": "MiniMax: H3",
        "best_known_for": (
            "MiniMax's H3, a lightweight open-weights model aimed at precise, instruction-guided "
            "work rather than free-running scenes. It is the one to reach for when the clip has to "
            "carry legible text or a brand mark correctly, or when you want an edit applied to "
            "footage you supply instead of a scene invented from scratch. Attached footage only "
            "reaches it if your administrator has turned on sending media to a file host, and is otherwise left "
            "out with a note in the chat. Everything it makes is 2K — "
            "there is no lower tier to trade down to — across six framings from ultrawide 21:9 to "
            "vertical 9:16, in clips of 5 to 15 seconds, with audio. There is no seed, so runs vary."
        ),
        "tips_and_pitfalls": [
            "Write the instruction, not the scene: this model responds to being told what to change or render, and rewards precise wording over atmosphere.",
            "Put any text you need rendered in quotes exactly as it should appear, including capitalisation — this is one of the few models that will hold it.",
            "Every clip is 2K, so there is no lower resolution to draft at; keep drafts short instead and lengthen only once the prompt is right.",
            "Trim the reference set to the images that are doing work; each extra one is another thing the model has to reconcile.",
            "There is no seed, so an exact re-run is not available — keep the take you like rather than expecting to reproduce it.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, 5 to 15.",
            "Aspect ratio": "21:9, 16:9, 4:3, 1:1, 3:4, or 9:16.",
            "Resolution": "2K, the only tier this model publishes.",
            "Frames": "Which supplied stills anchor the clip: none, first_only, or first_last to fix both ends.",
            "Audio": "Whether a soundtrack is generated with the picture.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "runway/aleph-2": {
        "display_name": "Runway: Aleph 2.0",
        "best_known_for": (
            "Runway's Aleph 2.0, which edits video you already have rather than generating a scene "
            "from a description. You give it footage and an instruction — change the weather, "
            "replace what is on the wall, take the parked cars out of the street — and it applies "
            "that across the clip while leaving everything you did not ask about alone. Keyframes "
            "let you show it what a moment should look like instead of describing it. Because the "
            "work is done on your footage, the length and the dimensions of the result come from "
            "the clip you supply, not from a setting here; the eight framings it publishes cover "
            "everything from 21:9 down to 9:16. It honours a seed, and it does not add audio. "
            "Your footage only reaches it if your administrator has turned on sending media to a file host; with that "
            "off the clip is left out and the chat tells you so."
        ),
        "tips_and_pitfalls": [
            "Attach the clip you want edited to your message. It is only sent if your administrator has turned on sending media to a file host — with that off the clip is left out, the chat says so, and there is nothing here to edit.",
            "Name the change and nothing else. \"Make it raining\" preserves the shot; re-describing the whole scene invites the model to redo parts you wanted kept.",
            "Once it arrives, the clip you attached sets the length and the size of the result — there is no duration or resolution control here, so trim the footage to what you actually want before sending it.",
            "One instruction per pass holds up far better than a list; run a second pass for the second change and you keep the ability to reject either one.",
            "Use keyframes when a change is easier to show than to write — a frame of the intended look steers it harder than another sentence will.",
            "Fix a seed before iterating so the untouched parts of the shot stay untouched between runs.",
            "There is no generated audio — the soundtrack is whatever your source clip carried.",
            f"Batch small corrections into a single pass where you can, rather than sending a string of one-second fixes. {OPENROUTER_PRICING}",
        ],
        "knob_descriptions": {
            "Aspect ratio": "The framing to work in — 16:9, 4:3, 3:2, 1:1, 2:3, 3:4, 9:16, or 21:9.",
            "Seed": "Fixes the random draw so the same footage and instruction reproduce the same edit — set it before iterating.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "runway/gen-4.5": {
        "display_name": "Runway: Gen-4.5",
        "best_known_for": (
            "Runway's Gen-4.5, a text- and image-to-video model tuned for cinematic shots: strong "
            "motion, high visual fidelity, and close adherence to what the prompt actually asked "
            "for. It is deliberately narrow — 720p, landscape 16:9 or portrait 9:16 only, clips of "
            "2 to 10 seconds, animated from a single opening still when you supply one — and that "
            "narrowness is the point, because it means the one thing it does it does very well. It "
            "honours a seed, and it does not generate audio. Where Aleph 2.0 edits footage you "
            "already have, this is the Runway model that creates the shot in the first place."
        ),
        "tips_and_pitfalls": [
            "Write it like a shot list: subject, action, camera move, lens feel, lighting. Prompt adherence is this model's strength and it rewards being specific.",
            "Two seconds is a real option — stringing several short beats together often beats asking for one ten-second take.",
            "Only a first frame is accepted; there is no closing still, so describe where the shot should end up rather than expecting to pin it.",
            "Landscape and portrait are the only framings — if you need square or ultrawide, this is not the model.",
            "No audio is generated, so put the whole prompt into what is seen and add sound afterwards.",
            "Fix a seed to keep identity and staging stable while you refine the wording.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, 2 to 10.",
            "Aspect ratio": "Landscape 16:9 or portrait 9:16.",
            "Resolution": "720p, the only tier this model publishes.",
            "Size": "Pins exact pixel dimensions — 1280x720 or 720x1280 — instead of letting the ratio decide.",
            "Frames": "Whether a supplied still opens the shot: none for pure text-to-video, or first_only to animate from it.",
            "Seed": "Fixes the random draw so the same prompt and seed reproduce the same clip.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "x-ai/grok-imagine-video-1.5": {
        "display_name": "SpaceXAI: Grok Imagine Video 1.5",
        "best_known_for": (
            "Grok Imagine Video 1.5, built for fast iteration. Duration is any whole number of "
            "seconds from 1 to 15, so you can draft an idea as a two-second clip and only commit to a "
            "full-length take once the prompt is right. It offers seven framings — more than most "
            "models — including the 3:2 and 2:3 photographic shapes its neighbours skip, and three "
            "resolutions from 480p up to 1080p, so drafting rough and finishing sharp is a single "
            "change of one control. It works from a text prompt alone or from a supplied opening "
            "still, and it accepts no other provider parameters, which makes it one of the simplest "
            "models here to drive."
        ),
        "tips_and_pitfalls": [
            "Draft at 480p and one or two seconds — the composition reads clearly enough at the low tier to judge.",
            "Whole-second durations mean you can ask for exactly the beat you need rather than rounding up to the next preset.",
            "3:2 and 2:3 are worth remembering when a clip has to sit alongside photography — few other models offer them.",
            "There are no provider parameters to fall back on, so everything you want has to be in the prompt and the controls.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, any whole number from 1 to 15.",
            "Aspect ratio": "Seven framings: 16:9, 9:16, 1:1, 4:3, 3:4, and the photographic 3:2 and 2:3.",
            "Resolution": "480p, 720p, or 1080p — the detail tier the clip is rendered at.",
            "Frames": "Whether a supplied still opens the shot: none for pure text-to-video, or first_only to animate from it.",
            "Audio": "Asks for a soundtrack with the picture. Nothing is published about whether this model obliges, so leaving it alone keeps the model's own behaviour.",
            "Seed": "Asks for a fixed random draw so a prompt can be re-run. Nothing is published about whether this model honours one, so treat a repeat as likely rather than guaranteed.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "alibaba/happyhorse-1.1": {
        "display_name": "Alibaba: HappyHorse 1.1",
        "best_known_for": (
            "Alibaba's HappyHorse 1.1 text- and image-to-video model. It generates 3-to-15-second "
            "clips at 720p or 1080p and stands out for an unusually wide aspect-ratio range — the usual "
            "16:9 / 9:16 / 1:1 / 4:3 / 3:4 plus ultrawide 21:9 and tall 9:21 — which makes it a fit for "
            "cinematic letterbox shots and full-bleed vertical formats that the 8-second-capped models "
            "can't cover in one clip. It supports first-frame image conditioning and a seed for "
            "reproducible runs. OpenRouter publishes nothing either way about audio for this model, "
            "so the audio control is offered and whatever the model does by default is what you get. "
            "1.1 refines 1.0 with the same controls."
        ),
        "tips_and_pitfalls": [
            "Front-load one clear shot — subject, action, setting, camera move, and style in plain prose; one idea per clip holds together far better than crowded multi-subject scenes.",
            "Use a first-frame image to lock the opening composition and identity, then describe only the motion that follows, not the still itself.",
            "Longer durations (10-15s) tax motion and identity consistency harder — reuse a seed when iterating prompt tweaks so the clip doesn't drift between runs.",
            "Whether audio comes back is not something OpenRouter states for this model — try one short clip with the audio control on before planning a soundtrack around it.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds (3-15); longer clips are harder to keep consistent.",
            "Aspect ratio": "Framing from a wide set — 16:9 / 9:16 / 1:1 / 4:3 / 3:4 plus ultrawide 21:9 and tall 9:21.",
            "Resolution": "720p or 1080p — the detail tier the clip is rendered at.",
            "Size": "Pins exact pixel dimensions (e.g. 1920x1080, 1080x1920, 2520x1080) instead of letting aspect ratio + resolution decide.",
            "Frames": "First-frame image conditioning — auto/none for pure text-to-video, or first_only to animate from a supplied starting still.",
            "Audio": "Asks for a soundtrack with the picture. Nothing is published about whether this model obliges, so leaving it alone keeps the model's own behaviour.",
            "Seed": "Integer for reproducible regeneration — same prompt + seed yields a near-identical clip when iterating.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "alibaba/happyhorse-1.0": {
        "display_name": "Alibaba: HappyHorse 1.0",
        "best_known_for": (
            "Alibaba's first HappyHorse video model — text- and image-to-video with the same wide "
            "aspect-ratio range (including ultrawide 21:9 and tall 9:21), 3-to-15-second clips at 720p or "
            "1080p, first-frame conditioning, and seed control. Nothing is published either way about "
            "audio, so that control is offered and the model's own behaviour decides. Largely "
            "superseded by HappyHorse 1.1, which offers the same controls; reach for "
            "1.0 only when you need to pin the exact 1.0 generation behaviour."
        ),
        "tips_and_pitfalls": [
            "Prefer HappyHorse 1.1 for new work — it matches 1.0's controls and resolutions.",
            "Front-load a single clear shot in plain prose and keep to one idea per clip; multi-subject action remains a weak spot.",
            "Anchor the opening with a first-frame image and reuse a seed across iterations to keep identity stable.",
            "Whether audio comes back is not stated for this model — test one short clip with the audio control on rather than assuming either way.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds (3-15); longer clips strain consistency.",
            "Aspect ratio": "16:9 / 9:16 / 1:1 / 4:3 / 3:4 plus ultrawide 21:9 and tall 9:21.",
            "Resolution": "720p or 1080p — the detail tier the clip is rendered at.",
            "Size": "Exact pixel dimensions when you need a specific canvas rather than a ratio+resolution pair.",
            "Frames": "First-frame conditioning — none for text-to-video or first_only to animate from a starting still.",
            "Audio": "Asks for a soundtrack with the picture. Nothing is published about whether this model obliges, so leaving it alone keeps the model's own behaviour.",
            "Seed": "Integer seed for reproducible regeneration across prompt iterations.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "google/veo-3.1-fast": {
        "display_name": "Google: Veo 3.1 Fast",
        "best_known_for": (
            "Google DeepMind's speed-optimised tier of Veo 3.1, generating 4-, 6-, "
            "or 8-second clips up to 4K with native synchronised audio, rendering faster "
            "than full Veo 3.1. Editor blind tests "
            "put its quality close to the full tier's, making it "
            "the workhorse choice for drafting, A/B-testing creative concepts, batch ad "
            "and social content, and image-to-video work where dialogue and SFX must land "
            "in sync."
        ),
        "tips_and_pitfalls": [
            "Front-load one clear shot: cinematography + subject + action + context + style/audio in plain prose, one idea per clip — multi-subject scenes and crowded actions remain a weak spot.",
            "Use first/last frame for controlled transitions: describe the transformation between the two stills (e.g. \"camera arcs 180°, lighting shifts cool→warm\"), not the stills themselves; mismatched aspect/lighting causes identity pops.",
            "Audio is generated from prompt cues — describe ambient sound, dialogue, and SFX explicitly, otherwise you get generic ambience; turning audio off avoids out-of-sync lip movement on talking heads.",
            "At 8s, faces and logos can drift partway through — lock with reference text, reuse a seed when iterating, and use the negative prompt to exclude common failures (\"no text overlays, no extra fingers, no warped logos\").",
        ],
        "knob_descriptions": {
            "Duration": "Picks clip length in seconds — 4 (quick beat), 6 (mid-shot), or 8 (full scene with audio arc); longer durations tax motion consistency harder.",
            "Aspect ratio": "Chooses 16:9 for landscape (YouTube/TV) or 9:16 for vertical (Reels/TikTok/Shorts); Veo composes natively for the chosen ratio rather than cropping.",
            "Resolution": "Selects 720p, 1080p, or 4K, which also sets how long the render takes.",
            "Size": "Pins exact pixel dimensions (e.g., 1920×1080, 2160×3840) when you need a specific canvas instead of letting aspect_ratio + resolution decide.",
            "Frames": "Controls image conditioning — auto/none for pure text-to-video, first_only to animate from a starting still, or first_last to interpolate a controlled transition between two stills.",
            "Negative prompt": "Free-text list of things to exclude (e.g., \"no text, no extra limbs, no logos\") — Veo 3.1 honours negation explicitly per the DeepMind prompt guide.",
            "Audio": "Toggles native synchronised audio generation; off returns a silent clip with no dialogue or effects, model_default lets the model decide.",
            "Seed": "Integer for deterministic regeneration — same prompt + same seed yields a near-identical clip, useful for iterating prompt tweaks without identity drift.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Person generation": "Safety gate for human/face content — \"allow_all\" (broadest), \"allow_adult\" (default per Vertex AI docs, adults only), \"dont_allow\" (no people, Gemini API spelling), \"disallow\" (the same refusal, Vertex AI spelling); blank uses model default.",
            "Conditioning scale": "Biases how strongly reference/frame images steer the output versus the text prompt; 0 leaves the model at its default balance. Google publishes no range for it and no meaning for any value, so the 0–1 slider is this pipe's own caution — send a larger number under Provider options JSON if you need one.",
            "Enhance prompt": "Asks the provider to auto-rewrite/expand the prompt before generation — on for richer cinematic detail, off to send your prompt verbatim, model_default to defer.",
        },
    },
    "google/veo-3.1-lite": {
        "display_name": "Google: Veo 3.1 Lite",
        "best_known_for": (
            "Google's lightest Veo 3.1 tier, positioned for high-volume video "
            "applications and rapid iteration. "
            "It matches Veo 3.1 Fast's latency — making it the "
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
            "Duration": "Picks the clip length in seconds from the model's supported set (4, 6, or 8); longer clips tax motion consistency harder.",
            "Aspect ratio": "Selects landscape 16:9 or portrait 9:16 framing — the only two orientations Lite supports.",
            "Resolution": "Chooses 720p or 1080p — the detail tier the clip is rendered at; 4K is not available on this tier.",
            "Size": "Pins exact pixel dimensions (1280×720, 720×1280, 1920×1080, or 1080×1920) when you need a specific output size rather than just a resolution+ratio pair.",
            "Frames": "Controls image-to-video conditioning — auto/none for pure text-to-video, first_only to anchor the opening frame, or first_last to interpolate between a starting and ending image.",
            "Negative prompt": "Free-text list of things to keep out of the clip (e.g. \"blurry, watermark, distorted hands\"); Google applies it as an explicit exclusion rather than a hint.",
            "Audio": "Toggles native synchronised audio generation (ambient sound, SFX, dialogue, music); disabling it returns a silent clip at whichever resolution you picked.",
            "Seed": "Sets an integer seed for reproducibility — Google notes it improves determinism but does not strictly guarantee identical outputs across runs.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Person generation": "Controls whether humans may appear in output — allow_all, allow_adult, dont_allow (Gemini API) or disallow (Vertex AI); in EU/UK/CH/MENA only allow_adult is permitted for Veo 3.1.",
            "Conditioning scale": "Biases how strongly the model adheres to your input image(s) versus the text prompt when using first/last frame conditioning. Google publishes no range for it and no meaning for any value, so the 0–1 slider is this pipe's own caution — send a larger number under Provider options JSON if you need one.",
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
            "texture/lighting detail, and access to 4K output. "
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
            "Duration": "Length of the generated clip in seconds; Veo 3.1 supports 4, 6, or 8 seconds.",
            "Aspect ratio": "Sets framing as 16:9 (landscape) or 9:16 (vertical); pick 9:16 for mobile/social and 16:9 for cinematic or hero content.",
            "Resolution": "Chooses the output detail tier — 720p, 1080p, or 4K — where 4K is the sharpest this model publishes and is not offered on the Lite tier.",
            "Size": "Locks the exact pixel dimensions (e.g. 1920×1080, 2160×3840) when you need a specific frame size rather than just an aspect/resolution pair.",
            "Frames": "Lets you anchor generation with a first_frame and/or last_frame image, ideal for image-to-video starts and for stitching shots into longer continuous scenes.",
            "Negative prompt": "Free-text list of things to suppress (e.g., \"motion blur, warped hands, on-screen text\") — the primary lever for cleaning up Veo's known artifacts.",
            "Audio": "Toggles native synchronised audio generation; turning it off returns a silent clip, so you lose Veo 3.1's signature joint-diffusion soundtrack.",
            "Seed": "A 32-bit integer that makes generation reproducible — reuse the same seed plus prompt to get consistent results when iterating on small prompt changes.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Person generation": "Safety control for human subjects; \"allow_adult\" (default) permits adult faces and bodies, while \"dont_allow\" (Gemini API) and \"disallow\" (Vertex AI) refuse any people/faces.",
            "Conditioning scale": "Balances how hard the stills you supply steer the clip against your written prompt. Google publishes no range for it and no meaning for any value, so the 0–1 slider is this pipe's own caution — send a larger number under Provider options JSON if you need one.",
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
            "Duration": "Picks clip length in seconds — Kling O1 only accepts 5s or 10s.",
            "Aspect ratio": "Chooses the frame shape (16:9 landscape, 9:16 vertical, 1:1 square) to match the platform you're delivering to.",
            "Resolution": "Sets output quality tier; Kling O1 currently outputs only 720p, so this is effectively fixed.",
            "Size": "Selects the exact pixel dimensions tied to your aspect (1280×720, 720×1280, or 720×720); usually leave on auto so it follows the aspect ratio.",
            "Frames": "Lets you pin a first_frame and/or last_frame image to anchor the opening or closing pose, useful for continuity across shots or for image-to-video starts.",
            "Negative prompt": "Free-text list of things to avoid (artifacts, distorted faces, text, unwanted styles); Kling treats this as hard guardrails and it's the main quality lever here.",
            "Audio": "Toggles Kling's native audio generation (ambient sound / effects) along with the video — turn off if you plan to score the clip externally.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "kwaivgi/kling-v3.0-pro": {
        "display_name": "Kling: Video v3.0 Pro",
        "best_known_for": (
            "Kuaishou's top tier of Kling v3.0 — the highest-quality Kling "
            "listing OpenRouter carries, with sharper detail, stronger character consistency, "
            "and richer motion fidelity than the Standard tier. Best suited for hero "
            "shots, marketing deliverables, and pre-vis where quality is the "
            "priority. Same capability matrix as Kling v3.0 Standard (granular "
            "3–15s durations, first/last-frame anchoring, native audio, 720p, three "
            "aspects)."
        ),
        "tips_and_pitfalls": [
            "Use Pro for finals and hero shots; iterate on Standard first to lock prompt and references — the visual delta is meaningful but rarely what decides a draft.",
            "Kling responds to cinematic intent — describe camera move (slow dolly-in / tracking), motion physics, and end state explicitly rather than listing objects.",
            "No seed is exposed (catalog confirms seed=false), so re-running the same prompt does NOT produce identical output — lock look via first_frame / last_frame and the negative prompt instead.",
            "The CFG scale control is new in v3.0 — Kling O1 does not have it. Leave it at 0 to take Kling's own balance, or nudge it up (~0.5+) when the prompt must be followed strictly at the expense of creative variation.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in whole seconds; Kling v3.0 Pro accepts any integer from 3 to 15s.",
            "Aspect ratio": "Frame shape (16:9 landscape, 9:16 vertical, 1:1 square) — pick to match your delivery surface; the model fills the chosen aspect with a fixed 720p tier.",
            "Resolution": "Kling v3.0 currently outputs only 720p, so there is nothing else to pick here.",
            "Size": "Exact pixel dimensions tied to your aspect choice (1280×720, 720×1280, 720×720); usually leave on auto so it follows the aspect ratio.",
            "Frames": "Optional first_frame and/or last_frame reference images that anchor the opening and/or closing pose — essential for multi-shot continuity and for image-to-video starts.",
            "Negative prompt": "Free-text guardrails (e.g. \"blurry text, extra fingers, warped face, on-screen text\"); Kling honours negatives well, treat as hard constraints.",
            "CFG scale": "Classifier-free guidance strength (0–1); 0 uses the provider default, higher values force stricter prompt adherence and narrow creative range — new in Kling v3.0.",
            "Audio": "Toggles native synchronised ambient/effects audio along with the video; switch off only if you plan to score the clip externally.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "kwaivgi/kling-v3.0-std": {
        "display_name": "Kling: Video v3.0 Standard",
        "best_known_for": (
            "Kuaishou's standard tier of Kling v3.0, with the same capabilities as "
            "Kling v3.0 Pro (granular 3–15s durations, first/last-frame anchoring, "
            "native audio, 720p, three aspects). "
            "Best suited for prompt iteration, drafts, and bulk pipelines where "
            "throughput matters more than the last few percent of polish."
        ),
        "tips_and_pitfalls": [
            "Use Standard for drafting, prompt and reference iteration, and bulk runs; switch to Pro for finals when the quality delta is worth it.",
            "Kling responds to cinematic intent — describe camera move, motion physics, and end state explicitly rather than listing objects.",
            "No seed is exposed (catalog confirms seed=false), so re-running the same prompt does NOT produce identical output — lock look via first_frame / last_frame and the negative prompt instead.",
            "The CFG scale control is new in v3.0 — Kling O1 does not have it. Leave it at 0 to take Kling's own balance, or nudge it up (~0.5+) when the prompt must be followed strictly at the expense of creative variation.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in whole seconds; Kling v3.0 Standard accepts any integer from 3 to 15s.",
            "Aspect ratio": "Frame shape (16:9 landscape, 9:16 vertical, 1:1 square) — pick to match your delivery surface; the model fills the chosen aspect with a fixed 720p tier.",
            "Resolution": "Kling v3.0 currently outputs only 720p, so there is nothing else to pick here.",
            "Size": "Exact pixel dimensions tied to your aspect choice (1280×720, 720×1280, 720×720); usually leave on auto so it follows the aspect ratio.",
            "Frames": "Optional first_frame and/or last_frame reference images that anchor the opening and/or closing pose — essential for multi-shot continuity and for image-to-video starts.",
            "Negative prompt": "Free-text guardrails (e.g. \"blurry text, extra fingers, warped face, on-screen text\"); Kling honours negatives well, treat as hard constraints.",
            "CFG scale": "Classifier-free guidance strength (0–1); 0 uses the provider default, higher values force stricter prompt adherence and narrow creative range — new in Kling v3.0.",
            "Audio": "Toggles native synchronised ambient/effects audio along with the video; switch off only if you plan to score the clip externally.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
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
            "No audio: Hailuo 2.3 is silent — OpenRouter publishes it as generating none, so dialogue, effects and music all have to be added afterwards.",
            "First-frame only: 2.3 dropped last-frame conditioning that 2.0 had, so you can anchor the opening still but cannot pin the ending — plan motion to flow forward from the first frame.",
            "Hailuo rewards specific physical and emotional direction (e.g. \"tight smile turning to laughter,\" \"cloth catches the wind, then settles\") far more than other models — vague prompts under-use its physics strengths.",
            "Prompt optimizer rewrites and expands what you wrote so the model follows it more closely; turn it off only when your prompt is already deliberately precise. Fast pretreatment runs that same step more quickly with a small loss of quality — handy for batch runs, otherwise leave it alone.",
        ],
        "knob_descriptions": {
            "Duration": "Length of the generated clip in seconds; Hailuo 2.3 supports either 6s or 10s.",
            "Aspect ratio": "Frame shape of the output; this model is locked to 16:9 widescreen.",
            "Resolution": "Vertical pixel count of the render; this model only outputs 1080p (full HD).",
            "Size": "Exact pixel dimensions of the output frame; fixed at 1920×1080.",
            "Frames": "Optional reference images; Hailuo 2.3 accepts only a first_frame image to anchor the opening shot and does not support a last frame.",
            "Seed": "Asks for a fixed random draw so a prompt can be re-run. Nothing is published about whether this model honours one, so treat a repeat as likely rather than guaranteed.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Prompt optimizer": "Enables MiniMax's server-side prompt rewriter that expands and refines your prompt for better motion and adherence; leave on for short or casual prompts, set off for verbatim.",
            "Fast pretreatment": "Only meaningful when the prompt optimiser is active — runs a quicker, lighter optimisation pass to cut latency (handy for batch generation) at a small loss of fine-tuning quality.",
        },
    },
    "alibaba/wan-2.7": {
        "display_name": "Alibaba: Wan 2.7",
        "best_known_for": (
            "Alibaba Tongyi Lab's flagship multimodal video model, unifying text, image, "
            "audio, and video conditioning in a single 27B-parameter Diffusion Transformer "
            "with Flow Matching. Its standout capability is multi-reference control: "
            f"{TRUE_WAN_REFERENCES}. A \"Thinking Mode\" planner improves coherence on "
            "dialogue- and character-led shots — with weaker fast-motion physics "
            "than Seedance 2.0."
        ),
        "tips_and_pitfalls": [
            "Reference images JSON is the reference control you have here: it locks appearance, wardrobe and props, and reads like a 9-image storyboard grid in 2.7. Alibaba also describes clip and voice references for this model, but OpenRouter reports it as taking only text and pictures, so there is no control for either — describe the motion and the camera in the prompt instead.",
            "For talking-head and dialogue clips, the lip-sync you get is the one Wan generates from your words: turn Audio on and write the line you want spoken, in the language you want it spoken in. Matching a supplied voice would need an audio reference, which is not offered here.",
            "Wan 2.7 is tuned for character-led, narrative content; for fast sports/action shots its physics still trails Seedance 2.0 and Runway Gen-4, so add explicit motion verbs and a negative prompt against blur/morphing.",
            "Wan 2.7's instruction-following changed vs 2.6, so prompts calibrated on 2.6 may drift; turn Prompt extend on when prompts are short, and off when you've already written a precise multi-shot storyboard.",
        ],
        "knob_descriptions": {
            "Duration": "Sets clip length in seconds (2–10 here); longer durations let Wan 2.7's full-attention DiT carry character identity further.",
            "Aspect ratio": "Picks the canvas shape (16:9, 9:16, 1:1, 4:3, 3:4); 9:16 is the right choice for the talking-head / lip-sync workflows Wan 2.7 is tuned for.",
            "Resolution": "Selects 720p or 1080p output; 1080p is the model's native ceiling — there is no 4K, so upscale in post if you need it.",
            "Size": "Forces an explicit pixel size (e.g. 1920×1080, 1440×1080); use this when you need a specific frame size that the aspect-ratio preset doesn't expose.",
            "Frames": "Lets you pin a first_frame and/or last_frame image so Wan 2.7 interpolates the motion between your two keyframes — this is the FLF2V control added in 2.7.",
            "Negative prompt": "Free-text list of things to suppress (e.g. \"blurry, extra fingers, morphing\"); useful on Wan 2.7 to push back on residual fast-motion artefacts.",
            "Audio": "Toggles native audio generation — Wan 2.7 bakes synchronised speech, ambience, and effects into the clip rather than dubbing them in post.",
            "Seed": "Fixes the RNG so the same prompt + references reproduce the same clip; essential when iterating on multi-shot sequences that need to match.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Audio reference URL": "A public link to a sound file whose voice and timing Wan 2.7 should match — this is what its multi-language lip-sync works from.",
            "Last image URL": "A public link to the still the clip should finish on; pair it with an opening frame to fix both ends and let the model fill the motion between them.",
            "Reference video URL": "A public link to one clip whose motion style, camera moves or voice Wan 2.7 should carry into the new scene.",
            "Reference videos JSON": "Up to five clips at once, written as a JSON list of links, so several different characters keep their look and voice across the same scene.",
            "Reference images JSON": "Up to nine pictures, written as a JSON list of links, fixing who is in shot, what they wear, the props and the setting so you do not have to describe them.",
            "Prompt extend": "Selects whether Wan's prompt rewriter expands your text (on), leaves it untouched (off), or uses the model's default; turn it off when you've already written a precise multi-shot storyboard.",
            "Ratio": "A frame shape written the way Wan names it, for a shape the Aspect ratio list does not offer. Otherwise use Aspect ratio, which is checked against what this model accepts.",
        },
    },
    "bytedance/seedance-2.0-fast": {
        "display_name": "ByteDance: Seedance 2.0 Fast",
        "best_known_for": (
            "ByteDance's speed-optimised variant of the Seedance 2.0 family, "
            "built on the same unified multimodal architecture but using distillation and "
            "accelerated sampling to cut generation time relative to standard "
            "Seedance 2.0. Best known for cinematic 480p/720p output with native "
            "audio synchronised in a single pass, support for text-to-video, image-to-"
            "video with first/last frame control, and multimodal reference-to-video, plus "
            "very wide aspect-ratio coverage including 21:9 cinematic and 9:21."
        ),
        "tips_and_pitfalls": [
            "Use Fast for drafting, prompt iteration, and bulk pipelines; switch to standard Seedance 2.0 for hero shots — Fast trades a small amount of motion refinement and detail for speed.",
            "Iterate small first: settle framing and motion on a short 480p draft before committing to a longer 720p take.",
            "This listing caps at 720p — for 1080p or 4K, switch to the standard Seedance 2.0 model.",
            "Neither this model nor standard Seedance 2.0 offers a box for saying what to keep out, so anything you want excluded has to be worded into the prompt itself.",
            "Watermark turns ByteDance's visible branding on the finished clip on or off. Request key takes a value ByteDance accepts whose meaning OpenRouter does not publish. Leave both alone unless your provider has told you otherwise.",
        ],
        "knob_descriptions": {
            "Duration": "Length of the generated clip in whole seconds; Seedance 2.0 Fast accepts any integer from 4 to 15s.",
            "Aspect ratio": "Picks the frame shape from seven options including ultrawide cinematic 21:9, vertical 9:16/9:21, square 1:1, and standard 16:9/4:3/3:4 — wider than most peers in this tier.",
            "Resolution": "Selects 480p (good for drafts) or 720p (final-quality tier on Fast); 1080p is not offered on the Fast variant on OpenRouter.",
            "Size": "Locks an exact pixel resolution (e.g. 1280×720, 854×480, 720×1680) overriding aspect/resolution when you need a specific output canvas.",
            "Frames": "Lets you pin the first and/or last frame of the video to a supplied image for image-to-video or precise start/end control of motion.",
            "Audio": "When on (model default true), Seedance generates synchronised dialogue, ambient sound, and music in the same pass as the video — no second audio model required.",
            "Seed": "Integer that makes generations reproducible — same prompt + seed yields the same clip, useful for A/B testing prompt edits.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Watermark": "Whether ByteDance stamps its visible branding on the finished clip — model_default keeps ByteDance's own policy, on forces it, off asks for a clean clip, which your account has to be allowed to receive.",
            "Request key": VIDEO_REQ_KEY_DESCRIPTION,
        },
    },
    "bytedance/seedance-2.0": {
        "display_name": "ByteDance: Seedance 2.0",
        "best_known_for": (
            "ByteDance's flagship multimodal video model, best known for \"locked\" "
            "character consistency — preserving faces, clothing, accessories, and small "
            "subject details across the duration of a clip and across multi-shot "
            "generations. It takes references as well as a prompt: the first picture you "
            "attach anchors the opening of the shot and the last one the ending, and any "
            "further pictures ride along as references that fix a character, a prop or a "
            "setting without you having to describe it. Clips and sound files you attach "
            "ride along the same way, but only if your administrator has turned on sending "
            "media to a file host; otherwise they are left out with a note in the chat. A "
            "sound file on its own is left out either way, because it is only accepted "
            "alongside a picture or a clip. Against 2.0 Fast, which stops at 720p, and 1.5 "
            "Pro, which stops at 1080p and 12 seconds, the full 2.0 variant reaches 4K and "
            "15 seconds with native audio (dialogue, ambience, SFX) and multi-shot story "
            "coherence."
        ),
        "tips_and_pitfalls": [
            "Reach for full Seedance 2.0 (not Fast) when you need production drafts where identity preservation matters — branded characters, story-led scenes, or repeatable creative formats — and accept the longer render in exchange for tighter facial/clothing fidelity.",
            "Long durations (12–15s) still drift more than short ones; for the most stable identity, anchor with a reference image AND a clear text description of the subject, and prefer 6–10s clips for hero shots.",
            "The multimodal reference workflow is the headline feature — use images for style/identity, video clips for motion/camera language, audio for pacing — but keep references coherent; conflicting references degrade consistency more than helping it.",
            "ByteDance gates real-person reference features and identity verification due to IP/likeness concerns; expect occasional refusals on celebrity or copyrighted-character prompts.",
        ],
        "knob_descriptions": {
            "Duration": "Sets clip length from 4 to 15 seconds; longer durations increase the chance of subtle character or scene drift, so pick the shortest length that tells the shot.",
            "Aspect ratio": "Chooses the framing from seven options (1:1, 3:4, 9:16, 4:3, 16:9, 21:9, 9:21), useful for matching social, cinematic, or vertical-mobile delivery before sizing.",
            "Resolution": "Selects 480p, 720p, 1080p, or 4K output — 1080p is the usual pick for client-facing drafts where character detail matters, and 4K is the highest tier this model publishes.",
            "Size": "Locks the exact pixel dimensions from the supported list (e.g. 1920×1080, 1080×1920, 2520×1080) when you need a specific frame size rather than just an aspect ratio.",
            "Frames": "Lets you supply a first_frame and/or last_frame image to anchor the clip's start and end, the most reliable way to enforce character/scene continuity on this model.",
            "Audio": "Toggles native audio generation (dialogue, ambient sound, SFX) — Seedance 2.0 has phoneme-level lip-sync, so leave on for finished drafts and off only for silent B-roll.",
            "Seed": "Locks the random seed for reproducible output, letting you re-run the same prompt and references to get a near-identical clip for iteration or A/B comparison.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Watermark": "Whether ByteDance's visible branding is burned into the clip; turn it off only if your account is allowed to receive clean clips.",
            "Request key": VIDEO_REQ_KEY_DESCRIPTION,
        },
    },
    "alibaba/wan-2.6": {
        "display_name": "Alibaba: Wan 2.6",
        "best_known_for": (
            "Alibaba's most feature-rich video generation model (Dec 2025), supporting "
            "10+ unified visual creation capabilities (text-to-video, image-to-video, "
            "reference-to-video, voiceover, action generation, role-play, editing) on a "
            "14B-parameter MoE architecture. Best known for multi-shot 1080p "
            "@ 24fps generation with synchronised native audio (multi-speaker dialogue, "
            "lip-sync, voice/music conditioning) and intelligent multi-shot narrative "
            "storyboarding that holds character and lighting consistency across cuts. "
            "Pick 2.6 over 2.7 when you want the well-tuned generation pipeline; "
            "pick 2.7 only if you specifically need last-frame control, 9-grid input, "
            "instruction-based editing, or stronger physics."
        ),
        "tips_and_pitfalls": [
            "First-frame ONLY: Wan 2.6 supports first_frame image conditioning but has no last_frame. To define both endpoints of a clip, you must upgrade to Wan 2.7 — don't try to fake it through prompts.",
            "Shot type sets how close the camera sits (values written as Wan names them, such as \"medium_to_closeup\"); Wan 2.7 dropped it. For multi-shot scripts, write scene-timed segments into the prompt itself.",
            "Turn Enable prompt expansion on for short or terse prompts — it adds camera and lighting detail for you; turn it off when you have already written a long, precise prompt.",
            "Alibaba documents conditioning this model on a supplied voice or music track, but OpenRouter reports it as taking only text and pictures, so there is no control for it here — the Audio toggle generates the soundtrack from your prompt instead. Two-speaker dialogue tends to collapse to one dominant voice either way, so generate single-speaker clips and composite.",
        ],
        "knob_descriptions": {
            "Duration": "Selects 5s or 10s of video; the OpenRouter listing caps at 10s, so the 15s length Alibaba Cloud sells directly is not available here.",
            "Aspect ratio": "Picks 16:9 (landscape) or 9:16 (portrait/vertical) framing for the output clip.",
            "Resolution": "Chooses 720p or 1080p; 1080p is the model's native high-fidelity tier.",
            "Size": "Direct pixel dimensions (1280×720, 1920×1080, 720×1280, 1080×1920) — overrides aspect/resolution if you need an exact frame size.",
            "Frames": "Attaches a first-frame reference image to anchor the opening shot; Wan 2.6 has no last-frame slot.",
            "Negative prompt": "Free-text list of things to avoid (artifacts, styles, objects, motion) — passed through to suppress unwanted features in the render.",
            "Audio": "Toggles Wan 2.6's native A/V synthesis so the output clip ships with synchronised sound effects, ambience, dialogue, or voiceover instead of a silent video.",
            "Seed": "Integer that fixes the random initialisation for reproducible/iterative generations from the same prompt.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Audio reference URL": "Public WAV/MP3 link (3–30s, ≤15 MB) that Wan 2.6 will lip-sync or musically conform the video to instead of generating audio from scratch.",
            "Enable prompt expansion": "Tri-state (model_default / on / off) for the LLM prompt-rewriter that auto-enriches short prompts with cinematographic detail; turn off for deterministic, fully-authored prompts.",
            "Shot type": "How close the camera sits — wide, medium, close-up, \"medium_to_closeup\" — written the way Wan names it, so 2.6 frames the shot you intended. Wan 2.7 dropped this control.",
        },
    },
    "bytedance/seedance-1-5-pro": {
        "display_name": "ByteDance: Seedance 1.5 Pro",
        "best_known_for": (
            "ByteDance's first foundation model to natively generate video and audio in a "
            "single unified pass, using a 4.5B-parameter Dual-Branch Diffusion Transformer "
            "with a cross-modal joint module that locks phonemes to visemes and physics "
            "events to audio spikes at millisecond precision. Pick 1.5 Pro over Seedance "
            "2.0 when you want the older, production-validated audio-visual workflow; "
            "it offers 1080p output, the wider 4–12s duration "
            "window, and reliable multilingual lip-sync (Mandarin, English, Japanese, "
            "Korean, Spanish, plus dialects)."
        ),
        "tips_and_pitfalls": [
            "Toggle Audio off for silent B-roll, layout passes, or anything you'll dub later.",
            "Use 1.5 Pro for short, repeatable clips with simple camera work and known-good prompts; switch to 2.0 only when you need richer multimodal references, 4K output, or longer 15s shots — 1.5 Pro caps at 1080p and 12s.",
            "Long durations drift: 4–6s clips stay on-model, but 10–12s shots show face drift, color shift, and continuity errors — chain shorter shots with last_frame anchors and consistent character descriptions.",
            "last_frame is a directional guide, not a pixel-perfect target — pick an end frame with framing and lighting close to the start frame, or you'll get jumpy transitions in the final second.",
        ],
        "knob_descriptions": {
            "Duration": "Sets clip length from 4–12 seconds; quality and continuity degrade past ~8s, so iterate short and only extend after motion looks right.",
            "Aspect ratio": "Picks one of seven framings (1:1, 3:4, 9:16, 9:21, 4:3, 16:9, 21:9) and should match your input image orientation to avoid awkward crops or stretched motion.",
            "Resolution": "Chooses 480p (fast previews), 720p (balanced), or native 1080p (final delivery); 1080p is the top tier here, while Seedance 2.0 carries on to 4K.",
            "Size": "Selects from 21 exact pixel dimensions, so you can hit platform-specific targets without post-crop.",
            "Frames": "Accepts a first_frame to lock identity/lighting and an optional last_frame to steer the ending, enabling match cuts and multi-shot continuity when you chain clips.",
            "Audio": "Turns on the dual-branch joint generation so lip-sync and physics SFX are produced in the same pass; disable it when you don't need sound.",
            "Seed": "Fixes the random initialisation for reproducible outputs — essential when iterating on prompt wording without re-rolling the whole scene.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Watermark": "Whether ByteDance's visible branding is burned into the finished clip.",
            "Request key": VIDEO_REQ_KEY_DESCRIPTION,
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
            "storytelling. It runs to 20 seconds at full 1080p."
        ),
        "tips_and_pitfalls": [
            "Text-to-video only here: this model takes no opening or closing still, so you cannot seed it with a picture — drive the result entirely from prompt language.",
            "Long takes at 1080p render slowly — community tests report 2–5 minutes for a 20s clip and much longer at peak, so prefer 4–8s 720p for iteration and reserve 16–20s 1080p for finals.",
            "Plays to its strengths on physics, motion weight, lighting, and ambient/dialogue audio; struggles with on-screen text, brand logos, fine hand details, and highly choreographed multi-character action — don't ship as-is for client deliverables that depend on legible text.",
            "Quality and Style are hints OpenRouter forwards exactly as you type them; OpenAI's video API documents neither, so treat both as a suggestion rather than a switch.",
        ],
        "knob_descriptions": {
            "Duration": "Pick clip length in seconds from 4, 8, 12, 16 or 20 — nothing in between is accepted, and render time scales roughly linearly with the length you pick.",
            "Aspect ratio": "Choose 16:9 for landscape/cinematic framing or 9:16 for vertical/social; this model does not support 1:1 or other ratios.",
            "Resolution": "720p is the quick iteration tier while 1080p is the cinematic finishing tier, with sharper textures and richer color depth but longer renders.",
            "Size": "Picks the exact pixel dimensions (1280×720, 1920×1080, 720×1280, 1080×1920) — use this when your downstream pipeline needs a specific frame size rather than just a ratio.",
            "Audio": "Sora 2 Pro generates synchronised audio natively (dialogue, SFX, ambience) from the same scene representation as the video — leave it on for realistic results.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
            "Quality": "A free-text hint. OpenAI publishes no quality values for video — \"standard\" and \"hd\" belong to its image models — so send only a value your provider accepts; in practice what you see is governed mainly by the resolution you pick.",
            "Style": "Free text that biases the look (e.g. \"cinematic\", \"anamorphic\", \"documentary handheld\"); OpenAI publishes no list of styles, so the prompt itself stays the main stylistic lever.",
        },
    },
    "x-ai/grok-imagine-video": {
        "display_name": "SpaceXAI: Grok Imagine Video",
        "best_known_for": (
            "SpaceXAI's fast text-, image-, and reference-conditioned video generator, "
            "producing short clips (1-15 seconds, 24 fps) at 480p or 720p across "
            "seven aspect ratios. Best for rapid iteration where you want tight "
            "control over duration in whole seconds and can start as short as one "
            "second — no other model in this catalog goes below two — with the "
            "option to anchor on a first frame for image-to-video continuity."
        ),
        "tips_and_pitfalls": [
            "Duration is any whole number from 1 to 15 seconds — pick the exact length you need rather than rounding up to a preset.",
            "480p is the iteration tier, 720p the finishing tier; there is no 1080p/4K on this model.",
            "Single-frame conditioning only — `first_frame` is supported but `last_frame` is not. Use Veo 3.1 or Kling if you need both endpoints locked.",
            "Seven aspect ratios cover landscape, vertical, square, and 4:3 / 3:2 photo formats; pick by destination platform.",
        ],
        "knob_descriptions": {
            "Duration": "Pick clip length in seconds, any whole number from 1 through 15. One second is the shortest start any model in this catalog offers — the next shortest begin at 2.",
            "Aspect ratio": "Choose from 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, or 2:3 — broader landscape/portrait/square/photo coverage than Sora or Veo Lite.",
            "Resolution": "Picks 480p (iteration) or 720p (finishing) — the detail tier the clip is rendered at.",
            "Size": "Pin exact pixel dimensions when you need a specific canvas (e.g. 854×480 for legacy SD, 1280×720 for HD).",
            "Frames": "Image conditioning — `first_frame` for image-to-video continuity, none for pure text-to-video. `last_frame` is not supported on this model.",
            "Audio": "Asks for a soundtrack with the picture. Nothing is published about whether this model obliges, so leaving it alone keeps the model's own behaviour.",
            "Seed": "Asks for a fixed random draw so a prompt can be re-run. Nothing is published about whether this model honours one, so treat a repeat as likely rather than guaranteed.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "alibaba/wan-3.0": {
        "display_name": "Alibaba: Wan 3.0",
        "best_known_for": (
            "Wan 3.0, the generalist of this catalogue and the one to reach for when you do not "
            "yet know what you need. Clips run any whole number of seconds from 2 to 30 — the "
            "longest range published here — across three resolutions from 480p to 1080p and five "
            "framings, so the same model drafts a two-second test at 480p and finishes a "
            "half-minute take at 1080p with one control changed. It generates its own audio, "
            "honours a seed, and animates from an opening still when you supply one. It accepts "
            "no provider parameters at all, which makes it one of the simplest models here to "
            "drive: what you see in these settings is the whole surface."
        ),
        "tips_and_pitfalls": [
            "Draft at 480p and move up once the prompt is right — finding the shot at the lowest tier and stepping up for the keeper is the whole discipline for this model.",
            "Thirty seconds is available but rarely the right first ask — a long take commits you to every second of it, and a mistake at second three spoils the whole clip.",
            "Only a first frame is accepted, so describe where the shot should finish rather than expecting to pin the closing image.",
            "Audio is generated with the video. If you plan to score it yourself, say so in the prompt rather than expecting a silent track.",
            "Fix a seed before you iterate, or every re-run changes the staging as well as the wording you meant to test.",
            "The five framings cover landscape through portrait but not ultrawide — if you need 21:9, this is not the model.",
            f"A long take at 1080p is the heaviest render this model offers, so be sure of the prompt before you commit to one. {OPENROUTER_PRICING}",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, any whole number from 2 to 30.",
            "Aspect ratio": "The framing to work in — 16:9, 4:3, 1:1, 3:4, or 9:16.",
            "Resolution": "480p, 720p or 1080p.",
            "Frames": "Whether a supplied still opens the shot: none for pure text-to-video, or first_only to animate from it.",
            "Audio": "Whether the model generates a soundtrack alongside the picture.",
            "Seed": "Fixes the random draw so the same prompt and seed reproduce the same clip.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "alibaba/wan-3.0-prime": {
        "display_name": "Alibaba: Wan 3.0 Prime",
        "best_known_for": (
            "Wan 3.0 Prime, the higher-fidelity tier of Wan 3.0. Everything about how you drive it "
            "is identical to its sibling — 2 to 30 seconds, 480p through 1080p, the same five "
            "framings, its own generated audio, a seed, and an optional opening still — so a "
            "prompt developed on Wan 3.0 moves here unchanged. What differs is the render: Prime "
            "is the higher-fidelity tier of the two. The sensible pattern is to "
            "find the shot on Wan 3.0 and render the keeper here. Like its sibling it accepts "
            "no provider parameters."
        ),
        "tips_and_pitfalls": [
            "Draft on plain Wan 3.0, finish here. The two models take the same settings, so nothing has to be re-tuned when you switch.",
            "Treat it as a finishing choice rather than a default — Wan 3.0 is the one to explore on.",
            "Clips run 2 to 30 seconds, but a long take at 1080p is the heaviest render here — be sure of the prompt first.",
            "Only a first frame is accepted, so describe the ending rather than trying to pin it with a closing still.",
            "Audio is generated with the video; say so in the prompt if you want it sparse.",
            "Fix a seed before iterating so the staging holds still while you change the wording.",
            "Ultrawide is not offered — the five framings run 16:9 to 9:16.",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, any whole number from 2 to 30.",
            "Aspect ratio": "The framing to work in — 16:9, 4:3, 1:1, 3:4, or 9:16.",
            "Resolution": "480p, 720p or 1080p.",
            "Frames": "Whether a supplied still opens the shot: none for pure text-to-video, or first_only to animate from it.",
            "Audio": "Whether the model generates a soundtrack alongside the picture.",
            "Seed": "Fixes the random draw so the same prompt and seed reproduce the same clip.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "minimax/hailuo-3-max": {
        "display_name": "MiniMax: H3 Max",
        "best_known_for": (
            "H3 Max, the widest-framing model in MiniMax's line and the one with the most "
            "flexible clip length in its family: any whole number of seconds from 5 to 15. It "
            "offers six framings including the 21:9 ultrawide its siblings skip, animates from a "
            "supplied first frame, a last frame, or both — so you can pin where a shot starts and "
            "where it ends and let the model find the motion between them — and renders at 480p "
            "or 768p. It does not generate audio and does not honour a seed, so identical prompts "
            "will not reproduce identical clips; plan to pick from several takes rather than to "
            "refine one deterministically."
        ),
        "tips_and_pitfalls": [
            "Supply both a first and a last frame when you know the beginning and the end — bookending the shot steers it far harder than describing the motion in words.",
            "There is no seed, so the same prompt twice gives two different clips. Iterate by generating a few and choosing, not by locking a draw.",
            "Draft at 480p and step up to 768p once the framing and motion are right.",
            "21:9 is available here and on few other models, so this is the one to use when you need a true ultrawide.",
            "No audio is generated; the clip arrives silent and the whole prompt should go into what is seen.",
            "Five seconds is the shortest it will make — for a shorter beat, generate five and trim.",
            f"Ask for the length you actually need rather than the longest available. {OPENROUTER_PRICING}",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, any whole number from 5 to 15.",
            "Aspect ratio": "The framing to work in — 21:9, 16:9, 4:3, 1:1, 3:4, or 9:16.",
            "Resolution": "480p or 768p.",
            "Frames": "Which supplied stills anchor the clip: a first frame, a last frame, or both.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "bytedance/seedance-2.0-mini": {
        "display_name": "ByteDance: Seedance 2.0 Mini",
        "best_known_for": (
            "Seedance 2.0 Mini, the lightest tier of the Seedance family and the only model here "
            "that takes all four input kinds — text, an image, a video clip and an audio track — "
            "in the same request. Clips run 4 to 15 seconds at 480p or 720p, across seven "
            "framings from 21:9 down to 9:21, and you can either pick a framing or pin exact pixel "
            "dimensions from the thirteen sizes it publishes. It generates its own audio, honours "
            "a seed, and anchors on a first frame, a last frame, or both. Supplying a reference "
        ),
        "tips_and_pitfalls": [
            "Building on a clip you already have is the more controllable path — the model works from your footage instead of inventing the whole shot.",
            "Your media only reaches the model if your administrator has turned on sending media to a file host — with that off the reference is left out and the chat tells you so.",
            "Pin a size rather than an aspect ratio when the output has to drop into a fixed frame; the thirteen sizes are exact pixel dimensions.",
            "Bookend with a first and last frame when you know both ends of the shot — it steers motion better than any amount of prose.",
            "Fix a seed before iterating so the staging holds while you change the wording.",
            "Audio is generated with the video, so say in the prompt what you want from it rather than discarding a track afterwards.",
            f"Resolution and length both drive how heavy a render is, so raise them only when the shot needs it. {OPENROUTER_PRICING}",
        ],
        "knob_descriptions": {
            "Duration": "Clip length in seconds, any whole number from 4 to 15.",
            "Aspect ratio": "The framing to work in — 1:1, 3:4, 9:16, 4:3, 16:9, 21:9, or 9:21.",
            "Resolution": "480p or 720p.",
            "Size": "Pins exact pixel dimensions instead of letting the ratio decide — thirteen are published, from 480x480 to 1680x720.",
            "Frames": "Which supplied stills anchor the clip: a first frame, a last frame, or both.",
            "Audio": "Whether the model generates a soundtrack alongside the picture.",
            "Seed": "Fixes the random draw so the same prompt and seed reproduce the same clip.",
            "Reference video URL": "A clip the model works from instead of inventing the whole shot.",
            "Audio reference URL": "A sound track the model works to.",
            "Watermark": "Whether ByteDance stamps its visible branding on the finished clip — model_default keeps ByteDance's own policy, on forces it, off asks for a clean clip, which your account has to be allowed to receive.",
            "Request key": VIDEO_REQ_KEY_DESCRIPTION,
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "heygen/avatar-iv": {
        "display_name": "HeyGen: Avatar IV",
        "best_known_for": (
            "Avatar IV, which animates a single photograph into a lip-synced talking head. It is "
            "unlike everything else in this catalogue: there is no duration control, because the "
            "length is however long the speech takes, and no prompt-driven scene, because the "
            "picture is the scene. You give it one still and either a script to voice or an audio "
            "track to lip-sync to. Rather than only matching mouth shapes, it reads tone and "
            "rhythm and drives head motion and expression from them. It renders at 720p or 1080p "
            "in 16:9, 9:16 or 1:1, and almost everything worth setting — the voice, its speed and "
            "pitch, how expressive the avatar is, the background, captions — lives in the provider "
            "parameters rather than the ordinary controls."
        ),
        "tips_and_pitfalls": [
            "Length is set by the speech, not by a control — a longer script is a longer clip, so trim the script to trim the video.",
            "Supply either a script or an audio track. An audio track lip-syncs directly; a script is voiced by HeyGen text-to-speech and then needs a voice_id.",
            "voice_id is an opaque identifier from HeyGen's own voice list, not a name you can invent — look it up in your HeyGen account first.",
            "expressiveness takes high, medium or low and defaults to low, so if the delivery looks flat it is probably doing exactly what it was told.",
            "motion_prompt is free text describing body motion and gestures, and applies to photo avatars — it is the closest thing here to a scene prompt.",
            "remove_background and background are separate: the first strips what was behind the person, the second supplies a flat colour or an image to replace it.",
            "Your photograph only reaches the model if your administrator has turned on sending media to a file host — with that off there is nothing to animate and the chat says so.",
            f"Trim the script rather than the resolution when you need a shorter result. {OPENROUTER_PRICING}",
        ],
        "knob_descriptions": {
            "Aspect ratio": "The framing to work in — 16:9, 9:16 or 1:1.",
            "Resolution": "720p or 1080p.",
            "Audio reference URL": "A voice track the photograph is lip-synced to, instead of a written script.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "black-forest-labs/flux-video-edit": {
        "display_name": "Black Forest Labs: FLUX Video Edit",
        "best_known_for": (
            "FLUX Video Edit, which changes footage you already have rather than generating a "
            "scene. You supply a clip and an instruction, and it applies that change across the "
            "video. Because the work is done on your footage, the length, the framing and the "
            "resolution of the result all come from the clip you send — which is why this model "
            "publishes no duration, aspect ratio or resolution control at all. It does not honour "
            "a seed and does not generate audio; the soundtrack is whatever your source carried. "
            "Its one provider parameter is a content-moderation threshold. Your footage only "
            "reaches it if your administrator has turned on sending media to a file host."
        ),
        "tips_and_pitfalls": [
            "Attach the clip you want changed to your message. With sending media to a file host turned off, the clip is left out, the chat says so, and there is nothing to edit.",
            "Name the change and nothing else. Re-describing the whole scene invites it to redo parts you wanted kept.",
            "The result takes its length, framing and resolution from your source clip — trim and crop before sending, because there is no control here to do it afterwards.",
            "One instruction per pass holds up better than a list; run a second pass for the second change and you keep the ability to reject either one.",
            "There is no seed, so two runs of the same instruction will differ. Generate a couple and choose.",
            "No audio is generated — whatever your source clip carried is what you get back.",
            f"The length of the clip you supply sets the length of the result, so trim it first. {OPENROUTER_PRICING}",
        ],
        "knob_descriptions": {
            "Reference video URL": "The clip to be edited. Everything about the output's shape comes from it.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
    "black-forest-labs/flux-video-upscale": {
        "display_name": "Black Forest Labs: FLUX Video Upscale",
        "best_known_for": (
            "FLUX Video Upscale, which enlarges footage you already have. It is the most narrowly "
            "scoped model in this catalogue: it publishes no duration, framing or resolution "
            "control, because all three are decided by the clip you supply and by how far you ask "
            "it to enlarge. Its two settings are how much bigger to make the video, between 1.5 "
            "and 3 times, and which of two modes it works in: one preserves the source exactly "
            "and sharpens it, the other restores and invents fine detail that was not there. "
            "It neither honours a seed nor "
            "generates audio, and your footage only reaches it if your administrator has turned "
            "on sending media to a file host."
        ),
        "tips_and_pitfalls": [
            "Attach the clip you want enlarged. With sending media to a file host turned off, there is nothing to upscale and the chat says so.",
            "A 3x pass on a long clip is the heaviest thing you can ask of it — enlarge only as far as the result actually needs.",
            "Choose the mode deliberately rather than leaving the default: faithful suits faces, products and brand assets; inventive suits textures, crowds and scenery.",
            "Enlarge once, not twice — a second pass compounds whatever the first one invented.",
            "Trim the clip before sending it; there is no duration control here to cut it afterwards, and the source is capped at 20 seconds and 2K.",
            "There is no seed, so two passes over the same footage will not invent the same detail.",
            f"No audio is generated; the soundtrack is whatever your source carried. {OPENROUTER_PRICING}",
        ],
        "knob_descriptions": {
            "Reference video URL": "The clip to be enlarged. Its length and dimensions decide the output.",
            "Provider options JSON": PROVIDER_OPTIONS_DESCRIPTION,
        },
    },
}


VIDEO_HELP_BY_MODEL = _PER_MODEL_HELP_DATA


_INTENT_ADMIN_GATE = "intent_classifier_admin"

_INTENT_KNOB_DESCRIPTIONS: dict[str, str] = {
    "Reuse previous videos": (
        "Whether a follow-up such as \"make it black\" edits the video you just got, or "
        "starts a new one from that message alone. On unless an admin says otherwise."
    ),
    "Clarifying question limit": (
        "How many short questions the chat may ask in a row when it cannot tell which "
        "earlier video you mean, before it picks one and gets on with it. 0 asks none."
    ),
    "Which frame to use from previous video": (
        "Which still is taken from the earlier clip when it is reused as a starting "
        "point: last continues from where it ended, first restarts from how it began."
    ),
    "Show what was reused": (
        "When to show the thumbnail naming what was reused. It appears while the clip "
        "is being made, so a wrong pick can be stopped before the generation finishes."
    ),
}


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
    "CFG scale": "cfg_scale",
    "Enhance prompt": "enhancePrompt",
    "Prompt optimizer": "prompt_optimizer",
    "Fast pretreatment": "fast_pretreatment",
    "Prompt extend": "prompt_extend",
    "Ratio": "ratio",
    "Enable prompt expansion": "enable_prompt_expansion",
    "Shot type": "shot_type",
    "Watermark": "watermark",
    "Request key": "req_key",
    "Quality": "quality",
    "Style": "style",
    "Reuse previous videos": _INTENT_ADMIN_GATE,
    "Clarifying question limit": _INTENT_ADMIN_GATE,
    "Which frame to use from previous video": _INTENT_ADMIN_GATE,
    "Show what was reused": _INTENT_ADMIN_GATE,
}


def _knob_is_active(knob: str, spec: VideoFilterSpec) -> bool:
    gate = _KNOB_GATE.get(knob)
    if gate is None:
        return True
    if gate == "negative_prompt_or_camelcase":
        return spec.supports_negative_prompt
    if gate == "generate_audio_top_level":
        return spec.supports_generate_audio_toggle
    if gate == "seed_top_level":
        return spec.supports_seed
    if gate == _INTENT_ADMIN_GATE:
        return spec.intent_classifier_admin_enabled
    return gate in spec.allowed_params


def _published_amount(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        amount = Decimal(str(value).strip())
    except (ArithmeticError, ValueError):
        return None
    return amount if amount.is_finite() else None


def _numeric_order(items: list[str]) -> list[str]:
    pairs: list[tuple[Decimal, str]] = []
    for item in items:
        amount = _published_amount(item)
        if amount is None:
            return items
        pairs.append((amount, item))
    pairs.sort(key=lambda pair: pair[0])
    return [item for _amount, item in pairs]


def _format_csv(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    items = [str(item).strip() for item in value if str(item).strip()]
    return ", ".join(_numeric_order(items))


def _format_frames_capability(supported_frames: Any) -> str:
    csv = _format_csv(supported_frames)
    if not csv:
        return "none"
    if "first_frame" in csv and "last_frame" in csv:
        return "first_frame and last_frame"
    return csv


_INPUT_KIND_WORDS = {
    "text": "a written prompt",
    "image": "an image",
    "audio": "an audio track",
    "video": "a video clip",
}


def _help_input_kinds(model: dict[str, Any]) -> list[str]:
    declared = model.get("input_modalities")
    if not isinstance(declared, list):
        arch = model.get("architecture")
        declared = arch.get("input_modalities") if isinstance(arch, dict) else None
    if not isinstance(declared, list):
        return []
    return [item for item in declared if isinstance(item, str) and item.strip()]


def _format_accepted_inputs(model: dict[str, Any]) -> str:
    kinds = _help_input_kinds(model)
    if not kinds:
        return "not published"
    known = [_INPUT_KIND_WORDS[kind] for kind in _INPUT_KIND_WORDS if kind in kinds]
    extra = sorted(kind for kind in kinds if kind not in _INPUT_KIND_WORDS)
    words = known + extra
    if not words:
        return "not published"
    if len(words) == 1:
        return words[0]
    return f"{', '.join(words[:-1])} and {words[-1]}"


_UNDECLARED_CAPABILITY = "not published; the control is offered and the model's own default applies"


def _declared_capability(declared: Any, offered: bool) -> str:
    if declared is True:
        return "yes"
    if offered:
        return _UNDECLARED_CAPABILITY
    return "no"


def _panel_knob_descriptions(curated: Any) -> dict[str, str]:
    merged = dict(curated) if isinstance(curated, dict) else {}
    for knob, description in _INTENT_KNOB_DESCRIPTIONS.items():
        merged.setdefault(knob, description)
    return merged


def _render_template(
    model_id: str,
    model: dict[str, Any],
    data: dict[str, Any],
    admin_valves: Any = None,
) -> str:
    from ..filters.video_filter_renderer import (
        _unhandled_params,
        build_video_filter_spec,
    )
    from .image_types import PASSTHROUGH_DESCRIPTION

    spec = build_video_filter_spec(model_id, model, admin_valves=admin_valves)
    display_name = str(model.get("name") or "").strip() or data.get("display_name") or model_id
    durations = _format_csv(model.get("supported_durations")) or "model default"
    aspects = _format_csv(model.get("supported_aspect_ratios")) or "model default"
    resolutions = _format_csv(model.get("supported_resolutions")) or "model default"
    frames = _format_frames_capability(model.get("supported_frame_images"))
    accepted = _format_accepted_inputs(model)
    audio = _declared_capability(model.get("generate_audio"), spec.supports_generate_audio_toggle)
    seed = _declared_capability(model.get("seed"), spec.supports_seed)

    knob_lines: list[str] = []
    for knob, description in _panel_knob_descriptions(data.get("knob_descriptions")).items():
        if not _knob_is_active(knob, spec):
            continue
        knob_lines.append(f"- `{knob}`: {description}")

    # Settings the model publishes that have no purpose-built control are still drawn,
    # as free text. Reading them from the same place the renderer does means help cannot
    # omit a control the chat UI shows -- which is the failure the curated table above
    # can produce on its own.
    for name in _unhandled_params(spec):
        knob_lines.append(f"- `{name}`: {PASSTHROUGH_DESCRIPTION}")

    tips = data.get("tips_and_pitfalls") or []
    tip_lines = "\n".join(f"- {bullet}" for bullet in tips)

    knobs_section = ""
    if knob_lines:
        knobs_section = "\n\n**Controls**\n" + "\n".join(knob_lines)

    tips_section = ""
    if tip_lines:
        tips_section = "\n\n**Tips & pitfalls**\n" + tip_lines

    return (
        f"### {display_name}\n\n"
        f"{data['best_known_for']}\n\n"
        "**Output capabilities**\n"
        f"- Accepted inputs: {accepted}\n"
        f"- Durations: {durations}\n"
        f"- Aspect ratios: {aspects}\n"
        f"- Resolutions: {resolutions}\n"
        f"- Frame controls: {frames}\n"
        f"- Generated audio: {audio}\n"
        f"- Deterministic seed: {seed}"
        f"{knobs_section}"
        f"{tips_section}"
    )


def render_video_help(
    model_id: str,
    video_model: dict[str, Any] | None = None,
    *,
    admin_valves: Any = None,
) -> str:
    model = video_model if isinstance(video_model, dict) else {}
    canonical_id = _canonical_model_id(model_id, model)
    data = _PER_MODEL_HELP_DATA.get(canonical_id)
    if data:
        return _render_template(canonical_id, model, data, admin_valves)
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
    accepted = _format_accepted_inputs(model)
    params = _format_csv(model.get("allowed_passthrough_parameters")) or "none listed"
    ratios = _format_csv(model.get("supported_aspect_ratios")) or "model default"
    durations = _format_csv(model.get("supported_durations")) or "model default"
    resolutions = _format_csv(model.get("supported_resolutions")) or "model default"
    return (
        f"### {display}\n\n"
        f"Capability: {description.strip() or 'OpenRouter video generation model.'}\n\n"
        f"Accepted inputs: {accepted}.\n\n"
        f"Frame controls: {frames}.\n\n"
        "Useful prompt patterns: Describe subject, action, setting, camera "
        "movement, visual style, and constraints in one clear shot.\n\n"
        "Known limitations: This model is not ZDR-capable and exact continuity "
        "can vary by generation.\n\n"
        f"Supported knobs: durations {durations}; aspect ratios {ratios}; "
        f"resolutions {resolutions}; provider parameters {params}."
    )
