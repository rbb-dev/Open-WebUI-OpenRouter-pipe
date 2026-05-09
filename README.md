# Open WebUI → OpenRouter Pipe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-2.5.3-blue.svg)](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe)
[![Open WebUI Compatible](https://img.shields.io/badge/Open%20WebUI-0.9.1%2B-green.svg)](https://openwebui.com/)

**390+ AI models. Chat, image, and video — all from your Open WebUI.**

GPT-5.5, Gemini 3, Claude Opus, Llama 4, FLUX.2, Sora 2, Veo 3.1, Kling, Wan, Riverflow — text, images, and video generation through OpenRouter's unified API. One key, one bill, every model that matters.

<p align="center">
  <img width="49%" alt="chat" src="https://github.com/user-attachments/assets/c937443b-f1be-4091-9555-b49789f16a97" />
  <img width="49%" alt="generation" src="https://github.com/user-attachments/assets/27681655-c494-408d-bc1a-b09e3d09f4c7" />
</p>

---

## What this is (in one minute)

* **OpenRouter Integration Subsystem for Open WebUI** — not a standalone service. Open WebUI loads it as a Function / Pipe.
* **Multimodal-aware routing adapters** — inspects the payload (text + images/files/audio/video) and picks the endpoint and format the target model supports.
* **Responses-first endpoint routing** — builds canonical requests and routes between `/responses` and `/chat/completions` based on model rules, fallbacks, or attachments.
* **Native image and video generation** — exposed as regular chat models with per-model knobs.
* **OpenRouter server tools** — `web_search`, `web_fetch`, and `datetime` behind one OWUI filter.
* **Operator controls via valves** — routing, limits, storage, security, telemetry, and templates.

---

## What You Get

🎯 **Every Model, One Place**
371 chat models, 13 video models, 16 image-output models. All variants (`:nitro`, `:thinking`, `:exacto`, `:free`) and OpenRouter presets (`@preset/...`).

🎨 **Image Generation, Inline**
16 image models — Sourceful Riverflow, Black Forest Labs FLUX.2, ByteDance Seedream, Gemini Image, GPT-5 Image. Type a prompt, get an image. Custom fonts, super-resolution, and Gemini's ultrawide aspect ratios all exposed as one-click filters.

🎬 **Video Generation**
13 video models — Veo 3.1, Sora 2 Pro, Kling, Wan, Hailuo, Seedance. Type a prompt, get a video that plays inline. Per-model knobs (duration, aspect ratio, resolution, audio, frames, negative prompt) all exposed as one-click filters.

🖼️ **Multimodal That Actually Works**
Drop in images, PDFs, audio, video. The pipe figures out what each model supports — `/responses` vs `/chat/completions`, file vs RAG, streaming vs not.

🔧 **OpenRouter Server Tools**
Web Search, Web Fetch, and Datetime — OpenRouter's server-side tools (run on their infrastructure, not yours, no client-side code). Any model can call them. Bundled into one toggleable filter; calls render as styled cards with citations.

🛡️ **Zero Data Retention (ZDR) Controls**
Filter to ZDR-only models. Enforce ZDR routing when privacy demands it. Video models always treated as not-ZDR.

🎨 **Complete Integration**
Model icons + descriptions + capabilities sync automatically. Per-chat cost display. Per-user cost attribution. Feels native because it is.

---

## What's New

- **Native image generation** — 16 image-output models (Sourceful, FLUX, Seedream, Gemini Image, GPT-5 Image) with 3 per-family filters (generic, Gemini Options, Sourceful Options).
- **Video generation** — 13 OpenRouter video models with per-model filters and inline `<video>` rendering.
- **OpenRouter Web Tools** — Web Search + Web Fetch + Datetime as one toggleable filter; tool execution cards with citations.
- **Open WebUI 0.9.x compatibility** — fully migrated to the async DB stack.
- **Provider routing filters** — admin + user-controlled routing, fallbacks, ZDR, sort order.
- **Direct Uploads filter** — bypass OWUI RAG; forward chat attachments as `input_file` to OpenRouter.

---

## For IT & Operations

⚡ **Production Hardened**
Rate limiting, circuit breakers, request admission, graceful degradation. 3500+ pytest tests, both readable and compressed bundle variants.

🔐 **Security First**
Encrypted credential storage. SSRF protection with HTTPS-only remote fetches by default. No secrets in logs. Capability-gated filter attach (image and video models cannot accidentally enable tools they don't support).

📊 **Cost & Attribution**
Track spending per user, per session, per model. Optional Redis export for billing integration. Per-user concurrency caps for video and image generation.

📝 **Audit Trail**
Optional encrypted session logs for incident response. Request identifiers flow through to OpenRouter for end-to-end attribution.

🏢 **Enterprise Controls**
Provider routing policies, ZDR enforcement, retention controls, per-model access (admin curated), tool/feature kill switches via valves.

---

## Quick Start

**1. Install**

In Open WebUI: **Admin Panel** → **Functions** → **+** → **Import from Link**

Pick one:

**Readable bundle (easy to audit/edit):**
```
https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/latest/download/open_webui_openrouter_pipe_bundled.py
```

**Compressed bundle (routine installs; smaller payload, faster upload, same runtime behavior):**
```
https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/latest/download/open_webui_openrouter_pipe_bundled_compressed.py
```

Both are automatically generated from the same modular source code on every release.

<details>
<summary>Alternative: bleeding-edge from dev branch</summary>

For the latest development commits (may be unstable):

**Readable bundle:**
```
https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/download/dev/open_webui_openrouter_pipe_bundled.py
```

**Compressed bundle:**
```
https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/releases/download/dev/open_webui_openrouter_pipe_bundled_compressed.py
```

</details>

**2. Enable**

Toggle the pipe **ON** (the switch next to the function name).

**3. Add Your API Key**

Click the **⚙️ gear icon** on the pipe → paste your [OpenRouter API key](https://openrouter.ai/keys) → **Save**.

**4. Select a Model**

Back in the chat, click the model dropdown — you'll see all OpenRouter chat, image, and video models. Pick one.

**5. Chat!**

For image generation, just describe what you want. For video, type a prompt and the pipe submits, polls, downloads, and renders inline. For chat models with reasoning, the `<think>` tokens stream live.

That's it.

### Try the per-model help

Type `help` (literally just that word, nothing else) in a chat against any image or video model — the pipe responds with curated, model-specific guidance: what it's best for, every knob the filter exposes, and tips/pitfalls for that specific model. Different answer for every one of the 30+ generation models.

---

## Requirements

- Open WebUI 0.9.1+
- An [OpenRouter](https://openrouter.ai/) account
- `WEBUI_SECRET_KEY` configured (required for encrypted credential storage)

---

## Documentation

Every document in [`docs/`](docs/README.md):

**Get started**
- [Valves & Configuration Atlas](docs/valves_and_configuration_atlas.md) — every valve, verified defaults
- [Image Generation](docs/openrouter_image_generation.md) — models, filters, per-model knobs
- [Video Generation](docs/openrouter_video_generation.md) — models, async lifecycle, resume behaviour
- [Server Tools](docs/openrouter_server_tools.md) — Web Search, Web Fetch, Datetime, legacy Image Gen
- [Direct Uploads](docs/openrouter_direct_uploads.md) — bypass OWUI RAG, forward as `input_file`
- [Provider Routing](docs/openrouter_provider_routing.md) — admin + user routing filters
- [Variants & Presets](docs/model_variants_and_presets.md) — `:nitro`, `:exacto`, `@preset/...`
- [Telemetry & Cost Attribution](docs/openrouter_integrations_and_telemetry.md) — identifiers, headers, exports
- [Error Handling & UX](docs/error_handling_and_user_experience.md) — what users see, troubleshooting
- [Web Search: OWUI vs OpenRouter](docs/web_search_owui_vs_openrouter_search.md) — when to use which

**Security & compliance**
- [Security & Encryption](docs/security_and_encryption.md) — credential storage, SSRF, hardening
- [Zero Data Retention](docs/openrouter_zdr.md) — ZDR filtering and enforcement
- [Persistence, Encryption & Storage](docs/persistence_encryption_and_storage.md) — what's stored, retention, ops
- [Session Log Storage](docs/session_log_storage.md) — encrypted incident-response archives
- [Request Identifiers & Abuse Attribution](docs/request_identifiers_and_abuse_attribution.md) — multi-user, privacy

**Operations & performance**
- [Concurrency Controls & Resilience](docs/concurrency_controls_and_resilience.md) — admission, breaker, tuning
- [Streaming Pipeline & Emitters](docs/streaming_pipeline_and_emitters.md) — streaming lifecycle, perf tradeoffs
- [Testing, Bootstrap & Operations](docs/testing_bootstrap_and_operations.md) — test harness, dev runbooks
- [Production Readiness Report](docs/production_readiness_report.md) — assessment-style doc

**Engineering deep dives**
- [Developer Guide & Architecture](docs/developer_guide_and_architecture.md) — systems map, contributor reference
- [Model Catalog & Routing Intelligence](docs/model_catalog_and_routing_intelligence.md) — catalog refresh, routing logic
- [History Reconstruction & Context](docs/history_reconstruction_and_context.md) — context restoration on resume
- [Multimodal Ingestion Pipeline](docs/multimodal_ingestion_pipeline.md) — image/audio/video processing
- [Task Models & Housekeeping](docs/task_models_and_housekeeping.md) — task model wiring, sweepers
- [Tooling & Integrations](docs/tooling_and_integrations.md) — plugin/tool integration patterns

Plus [`CHANGELOG.md`](CHANGELOG.md) — audit trail of changes.

---

## Contributing & forking

If you fork this, run the same checks CI does before pushing: `ruff check`, `pyright`, and `pytest`. CI validates the source plus both generated bundles (readable + compressed) on every push.

For code review, start with the pytest suite in `tests/` — broad coverage across chat, image, video, tools, filters, persistence, and the streaming pipeline.

---

## License

MIT — use it, fork it, ship it.
