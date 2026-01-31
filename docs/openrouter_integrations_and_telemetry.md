# OpenRouter Integrations & Telemetry

This document covers behaviors that are specific to the OpenRouter Responses API integration: request shaping, model catalog behavior, OpenRouter-specific parameters, and optional telemetry exports.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Identifiers](request_identifiers_and_abuse_attribution.md) · [Errors](error_handling_and_user_experience.md)

---

## 1. Endpoint and OpenRouter headers

- The pipe targets the OpenRouter base URL configured by `BASE_URL` (default `https://openrouter.ai/api/v1`), using the `/responses` endpoint.
- Requests include OpenRouter-identifying headers:
  - `X-Title` (pipe title)
  - `HTTP-Referer` (default project URL; can be overridden via `HTTP_REFERER_OVERRIDE` — must be a full URL including scheme)
- Optional provider beta headers:
  - For `anthropic/...` models, when `ENABLE_ANTHROPIC_INTERLEAVED_THINKING=True`, the pipe sends `x-anthropic-beta: interleaved-thinking-2025-05-14` to opt into Claude “interleaved thinking” streaming (reasoning may appear in multiple blocks during a single answer).

---

## 2. Request shaping and schema enforcement

### 2.1 Allowed request fields (OpenRouter Responses allowlist)
Before sending requests to OpenRouter, the pipe filters request bodies to the allowlist below (`ALLOWED_OPENROUTER_FIELDS`). Any keys not in this list are dropped. Explicit `null` values are also dropped because OpenRouter rejects `null` for optional fields.

| Field | Purpose / notes |
| --- | --- |
| `model` | Primary model for the request (selected in Open WebUI). |
| `models` | Fallback model list (OpenRouter will try these if the primary `model` fails). This pipe supports `model_fallback` as an OWUI convenience mapping to this field. |
| `input` | Responses input payload (constructed from Open WebUI messages and content blocks). |
| `instructions` | Additional instructions passed through to OpenRouter when present. |
| `metadata` | OpenRouter metadata map; sanitized to string→string with length/pair constraints (invalid entries dropped). |
| `stream` | Enables streaming mode. |
| `max_output_tokens` | Output token cap. The pipe may set/omit this depending on `USE_MODEL_MAX_OUTPUT_TOKENS` and routing decisions. |
| `temperature` | Sampling parameter (passed through when present). |
| `top_k` | Sampling parameter; numeric strings are coerced to numbers. When the pipe routes via `/chat/completions` (forced or fallback), `top_k` is rounded before sending upstream. |
| `top_p` | Sampling parameter (passed through when present). |
| `reasoning` | Reasoning configuration; only recognized subfields are forwarded (unknown keys dropped). |
| `include_reasoning` | Legacy reasoning flag; may be used as a fallback depending on model/provider behavior. |
| `tools` | Tool definitions (merged from Open WebUI registry tools plus Open WebUI Direct Tool Servers when present). |
| `tool_choice` | Tool selection directive. |
| `plugins` | Plugin configuration (for example OpenRouter web search attachment when enabled and supported). |
| `preset` | OpenRouter preset slug for pre-configured LLM settings (system prompts, provider routing, parameters). See [Model Variants & Presets](model_variants_and_presets.md#presets). |
| `text` | Response text configuration (`text.format` for structured outputs / JSON mode; `text.verbosity` when supported). |
| `parallel_tool_calls` | Tool parallelism hint (when supported). |
| `user` | OpenRouter user identifier (optional; controlled by identifier valves). |
| `session_id` | OpenRouter session identifier (optional; controlled by identifier valves). |
| `transforms` | OpenRouter transforms list (for example automatic middle-out trimming when enabled). |

Operational note:
- The pipe always constructs a canonical "Responses-style" request first, then converts it to a Chat Completions payload only when needed (forced endpoint selection or automatic fallback).
- Some parameters are Chat-only (for example `stop`, `seed`, `logprobs`, `top_logprobs`, `preset`). These are ignored when calling `/responses`, but are preserved so they can be used if the request is sent via `/chat/completions`.
- When a `preset` parameter is present in the request body, the pipe automatically forces `/chat/completions` because presets only work on that endpoint. For presets that work with `/responses`, use the VARIANT_MODELS approach with `@preset/slug` syntax instead. See [Model Variants & Presets](model_variants_and_presets.md#presets).

### 2.2 Advanced Model Parameters (per-model overrides)
Open WebUI allows per-model “Advanced Model Parameters” (stored on the model in `model.params`, typically under `model.params.custom_params`). This pipe supports a small set of per-model parameters that:

- affect **how requests are shaped** before they are sent to OpenRouter, and/or
- affect what the pipe will **auto-sync into Open WebUI model metadata** (icons, capability checkboxes, integrations toggles, descriptions).

The pipe accepts the following Advanced Model Parameters:

| Advanced param | Type | Applies to | What it does |
| --- | --- | --- | --- |
| `model_fallback` | `str` (CSV) | Requests | Convenience mapping for OpenRouter fallbacks: converts a CSV list into the OpenRouter `models` array (order-preserving, de-duplicated). |
| `disable_native_websearch` | `bool-ish` | Requests | Prevents OpenRouter native web search from being used for this model by stripping the OpenRouter web-search plugin and related request fields. |
| `disable_model_metadata_sync` | `bool-ish` | Model metadata sync | Master switch: the pipe will not “manage” this model’s settings at all (capabilities, icon, description, auto-attached integrations, default-on integrations). |
| `disable_capability_updates` | `bool-ish` | Model metadata sync | Prevents overwriting Open WebUI capability checkboxes (`meta.capabilities`). |
| `disable_image_updates` | `bool-ish` | Model metadata sync | Prevents overwriting the model icon/profile image (`meta.profile_image_url`). |
| `disable_description_updates` | `bool-ish` | Model metadata sync | Prevents overwriting the model description (`meta.description`). |
| `disable_openrouter_search_auto_attach` | `bool-ish` | Model metadata sync | Prevents auto-attaching the **OpenRouter Search** integration toggle (filter id) for this model. |
| `disable_openrouter_search_default_on` | `bool-ish` | Model metadata sync | Prevents auto-enabling **OpenRouter Search** by default for this model (prevents seeding `meta.defaultFilterIds`). |
| `disable_direct_uploads_auto_attach` | `bool-ish` | Model metadata sync | Prevents auto-attaching the **Direct Uploads** integration toggle (filter id) for this model. |

Notes:
- “bool-ish” accepts JSON booleans (`true/false`) and common string/int forms (`"true"`, `"1"`, `"on"`, etc.).
- `disable_native_websearch` has an alias key: `disable_native_web_search`.

### 2.3 `model_fallback` → OpenRouter `models`
OpenRouter supports a primary `model` plus a fallback list `models` (array). Open WebUI does not expose a first-class UI for OpenRouter’s `models` field, so this pipe supports a convenience parameter:

- Custom param: `model_fallback` (CSV string)
- Pipe behavior:
  - Parses the CSV into a de-duplicated list (order-preserving).
  - Merges with any existing `models` list in the request (existing entries first).
  - Writes the final list to `models` (fallback list only).
  - Removes `model_fallback` from the outgoing OpenRouter payload.

Example Open WebUI custom parameter value:

```text
openai/gpt-5,openai/gpt-5.1,anthropic/claude-sonnet-4.5
```

### 2.4 `disable_native_websearch` → disable OpenRouter web search plugin
Open WebUI can attach per-model custom parameters (model settings → `custom_params`). This pipe supports a boolean custom parameter to **disable OpenRouter’s built-in web search** for specific models.

- Custom param: `disable_native_websearch` (bool-ish; accepts `true/false`, `1/0`, etc.)
  - Alias: `disable_native_web_search`
- Pipe behavior (when truthy):
  - Removes `plugins` entries with `{"id": "web"}`.
  - Removes `web_search_options` when present (OpenRouter `/chat/completions`).

This is useful when OpenRouter Search is enabled by default (via the model’s Default Filters / `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER`) but you want to block provider-native web search on selected models.

Note: Open WebUI’s built-in **Web Search** is separate (OWUI-native) and is not controlled by this parameter.
See: [Web Search (Open WebUI) vs OpenRouter Search](web_search_owui_vs_openrouter_search.md).

### 2.5 `disable_model_metadata_sync` → opt out of all model metadata sync for a model
This is a per-model “master kill switch” for the pipe’s Open WebUI model metadata sync behavior.

- Custom param: `disable_model_metadata_sync` (bool-ish)
- Pipe behavior (when truthy):
  - Skips all metadata writes for that model, even if sync valves are enabled (`UPDATE_MODEL_*`, `AUTO_ATTACH_*`, `AUTO_DEFAULT_*`).
  - This preserves operator-edited settings (icons, descriptions, capabilities, and integration toggle defaults).

### 2.6 `disable_capability_updates` → preserve capability checkboxes
- Custom param: `disable_capability_updates` (bool-ish)
- Pipe behavior (when truthy):
  - Leaves `meta.capabilities` as-is for that model (no checkbox overwrites), even when `UPDATE_MODEL_CAPABILITIES=True`.

### 2.7 `disable_image_updates` → preserve the model icon
- Custom param: `disable_image_updates` (bool-ish)
- Pipe behavior (when truthy):
  - Leaves `meta.profile_image_url` as-is for that model, even when `UPDATE_MODEL_IMAGES=True`.

### 2.8 `disable_description_updates` → preserve the model description
- Custom param: `disable_description_updates` (bool-ish)
- Pipe behavior (when truthy):
  - Leaves `meta.description` as-is for that model, even when `UPDATE_MODEL_DESCRIPTIONS=True`.

### 2.9 `disable_openrouter_search_auto_attach` → preserve the OpenRouter Search toggle wiring
- Custom param: `disable_openrouter_search_auto_attach` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not add/remove the OpenRouter Search filter id in `meta.filterIds` for that model, even when `AUTO_ATTACH_ORS_FILTER=True`.
  - As a consequence, default-on seeding is also avoided (the pipe will not mark a filter as default unless it is attached).

### 2.10 `disable_openrouter_search_default_on` → preserve default-on behavior
- Custom param: `disable_openrouter_search_default_on` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not seed OpenRouter Search into `meta.defaultFilterIds` for that model, even when `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER=True`.
  - The OpenRouter Search toggle may still be auto-attached if `AUTO_ATTACH_ORS_FILTER=True` and the model supports it.

### 2.11 `disable_direct_uploads_auto_attach` → preserve the Direct Uploads toggle wiring
- Custom param: `disable_direct_uploads_auto_attach` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not add/remove the Direct Uploads filter id in `meta.filterIds` for that model, even when `AUTO_ATTACH_DIRECT_UPLOADS_FILTER=True`.

---

## 3. Model catalog and capability-aware routing

The pipe loads OpenRouter’s `/models` catalog and caches it to drive capability-aware behavior (for example: vision inputs, web search eligibility, reasoning toggles, token caps).

Key valves:
- `MODEL_ID` (default `auto`) controls whether the pipe exposes the full catalog or a comma-separated allowlist.
- `MODEL_CATALOG_REFRESH_SECONDS` controls refresh cadence.
- `USE_MODEL_MAX_OUTPUT_TOKENS` controls whether the pipe forwards provider-advertised output token caps.

### 3.1 Open WebUI model metadata sync (icons + capabilities)

Open WebUI stores additional per-model UI metadata (capabilities checkboxes and profile images) in its own Models table. This pipe can **automatically sync that metadata** for the OpenRouter models it exposes.

What it syncs (best-effort):
- `meta.profile_image_url`: downloads the model icon, converts it to **PNG**, and stores it as a `data:image/png;base64,...` data URL (Open WebUI does not process remote image URLs here).
  - SVG icons are rasterized to PNG (requires `cairosvg`).
  - Other images are converted to PNG (requires `Pillow`).
- `meta.description`: writes the model’s user-facing description from OpenRouter’s `/models` catalog when present.
- `meta.capabilities`: writes the Open WebUI capability checkboxes (for example `vision`, `file_upload`, `web_search`, `image_generation`).
  - Web search is **augmented** using OpenRouter’s public frontend catalog signals (in addition to `/models` pricing), so models like `x-ai/grok-4` and `openai/gpt-4o` can correctly show `web_search` even when `pricing.web_search` is `0`.

Data sources / egress:
- Fetches `https://openrouter.ai/api/frontend/models` (no auth) to discover icons and frontend-only capability signals.
- Downloads each icon URL (absolute or relative to `https://openrouter.ai`) and may fall back to a maker page OpenGraph image (`https://openrouter.ai/<maker>`).

Controls:
- `UPDATE_MODEL_IMAGES` (default `True`): enable/disable profile image sync.
- `UPDATE_MODEL_DESCRIPTIONS` (default `True`): enable/disable model description sync.
- `UPDATE_MODEL_CAPABILITIES` (default `True`): enable/disable capability checkbox sync.
- `NEW_MODEL_ACCESS_CONTROL` (default `admins`): sets the access control applied when the pipe **inserts** a new OpenRouter model overlay into Open WebUI (existing `access_control` values are preserved). Use `admins` to set `access_control={}` (private), which relies on Open WebUI’s `BYPASS_ADMIN_ACCESS_CONTROL` for admin access.
Per-model opt-outs:
- `disable_model_metadata_sync`: disables all metadata sync for the model.
- `disable_image_updates`, `disable_description_updates`, `disable_capability_updates`: disable specific metadata fields for the model.

Operational note:
- This sync updates Open WebUI’s Models table using Open WebUI’s own helper APIs (not raw SQL), but it is still a **write** to Open WebUI’s model metadata. Disable the valves if you want to manage model icons/capabilities manually.

---

## 4. Auto context trimming (OpenRouter transforms)

When `AUTO_CONTEXT_TRIMMING=True`, the pipe may attach OpenRouter’s `middle-out` transform by setting `transforms=["middle-out"]` **only when the request has no `transforms` list already**.

Operational guidance:
- Leave this enabled if you want long prompts to degrade gracefully instead of failing due to context limits.
- Disable it if you manage `transforms` explicitly in your deployment.

---

## 5. Tooling and plugins

- Web search:
  - When the **OpenRouter Search** toggle is enabled for the request (per chat, or enabled by default via Default Filters), and the selected model/provider supports OpenRouter web search, the pipe attaches the OpenRouter web-search plugin.
  - `WEB_SEARCH_MAX_RESULTS` caps result count.
- Response healing:
  - The OpenRouter response-healing plugin is intentionally **not** exposed by this pipe.
  - We prefer fail-fast behavior for malformed/invalid outputs so errors remain visible.
- Tools:
  - Tool schemas are built from Open WebUI’s `__tools__` registry plus any selected Open WebUI **Direct Tool Servers**.
  - Direct Tool Servers are executed client-side via Open WebUI (the pipe emits `execute:tool` via Socket.IO).
- Tool schema strictness:
  - When `ENABLE_STRICT_TOOL_CALLING=True`, the pipe strictifies tool schemas for more predictable function calling.

See also: [Tooling & Integrations](tooling_and_integrations.md).
And: [Web Search (Open WebUI) vs OpenRouter Search](web_search_owui_vs_openrouter_search.md).

---

## 6. User-visible telemetry (status and usage)

### Final usage status banner
When `SHOW_FINAL_USAGE_STATUS=True`, the pipe emits a final status message that can include timing, token counts, and OpenRouter cost/usage information when present in the upstream usage payload.

This is intended as user-visible telemetry and operator troubleshooting signal (not as an authoritative billing record).

### Status UI formatting
When `ENABLE_STATUS_CSS_PATCH=True`, the pipe applies a small UI formatting patch so multi-line statuses are easier to read in Open WebUI.

---

## 7. Optional telemetry export: cost snapshots to Redis

When enabled, the pipe can write per-request usage snapshots into Redis for downstream analytics and chargeback workflows.

Valves:
- `COSTS_REDIS_DUMP` (default `False`) enables/disables the feature.
- `COSTS_REDIS_TTL_SECONDS` (default `900`) controls retention in Redis.

Behavior (as implemented):
- Writes occur only when Redis caching is already enabled and available (`_redis_enabled=True`).
- Snapshots are written only when all required fields are present:
  - Open WebUI user ID (`guid`)
  - user `email`
  - user `name`
  - model ID
  - OpenRouter usage payload
- Keys are namespaced per pipe identifier:

```text
costs:{pipe_namespace}:{user_id}:{uuid}:{epoch_seconds}
```

Payload fields include:
- `guid`, `email`, `name`, `model`, `usage`, `ts`

Privacy guidance:
- These snapshots include user identity fields (email/name) from Open WebUI. Treat Redis access as sensitive, apply TTLs, and avoid using this feature if you do not need per-user cost attribution.

---

## 8. Persistence and encryption defaults (OpenRouter workloads)

OpenRouter reasoning outputs can be large, so persistence controls matter for operational cost and storage growth.

Relevant valves:
- `PERSIST_REASONING_TOKENS` (system default `conversation`)
- `ARTIFACT_ENCRYPTION_KEY` (enables encryption when set)
- `ENCRYPT_ALL` (default `True`; when encryption is enabled, encrypts all artifacts vs reasoning-only)
- `ENABLE_LZ4_COMPRESSION` (default `True`, when `lz4` is available)

See [Persistence, Encryption & Storage](persistence_encryption_and_storage.md) for the full behavior description.

---

## Related topics

- [Valves & Configuration Atlas](valves_and_configuration_atlas.md)
- [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md)
- [Model Catalog & Routing Intelligence](model_catalog_and_routing_intelligence.md)
- [Tooling & Integrations](tooling_and_integrations.md)
- [Error Handling & User Experience](error_handling_and_user_experience.md)
