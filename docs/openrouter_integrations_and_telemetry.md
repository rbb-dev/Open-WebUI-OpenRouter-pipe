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
| `top_k` | Sampling parameter; numeric strings are coerced to numbers and invalid values are dropped. |
| `top_p` | Sampling parameter (passed through when present). |
| `reasoning` | Reasoning configuration; only recognized subfields are forwarded (unknown keys dropped). |
| `include_reasoning` | Legacy reasoning flag; may be used as a fallback depending on model/provider behavior. |
| `tools` | Tool definitions (merged from Open WebUI registry tools plus Open WebUI Direct Tool Servers when present). |
| `tool_choice` | Tool selection directive. |
| `plugins` | Plugin configuration (for example OpenRouter web search attachment when enabled and supported). |
| `text` | Response text configuration (`text.format` for structured outputs / JSON mode; `text.verbosity` when supported). |
| `parallel_tool_calls` | Tool parallelism hint (when supported). |
| `user` | OpenRouter user identifier (optional; controlled by identifier valves). |
| `session_id` | OpenRouter session identifier (optional; controlled by identifier valves). |
| `transforms` | OpenRouter transforms list (for example automatic middle-out trimming when enabled). |

### 2.2 `model_fallback` → OpenRouter `models`
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

### 2.3 `disable_native_websearch` → disable OpenRouter web search plugin
Open WebUI can attach per-model custom parameters (model settings → `custom_params`). This pipe supports a boolean custom parameter to **disable OpenRouter’s built-in web search** for specific models.

- Custom param: `disable_native_websearch` (bool-ish; accepts `true/false`, `1/0`, etc.)
  - Alias: `disable_native_web_search`
- Pipe behavior (when truthy):
  - Removes `plugins` entries with `{"id": "web"}`.
  - Removes `web_search_options` when present (OpenRouter `/chat/completions`).

This is useful when `ENABLE_WEB_SEARCH_TOOL=True` globally but you want to block native web search on selected models.

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
- `meta.capabilities`: writes the Open WebUI capability checkboxes (for example `vision`, `file_upload`, `web_search`, `image_generation`).
  - Web search is **augmented** using OpenRouter’s public frontend catalog signals (in addition to `/models` pricing), so models like `x-ai/grok-4` and `openai/gpt-4o` can correctly show `web_search` even when `pricing.web_search` is `0`.

Data sources / egress:
- Fetches `https://openrouter.ai/api/frontend/models` (no auth) to discover icons and frontend-only capability signals.
- Downloads each icon URL (absolute or relative to `https://openrouter.ai`) and may fall back to a maker page OpenGraph image (`https://openrouter.ai/<maker>`).

Controls:
- `UPDATE_MODEL_IMAGES` (default `True`): enable/disable profile image sync.
- `UPDATE_MODEL_CAPABILITIES` (default `True`): enable/disable capability checkbox sync.

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
  - When `ENABLE_WEB_SEARCH_TOOL=True` and the selected model/provider supports web search, the pipe can attach OpenRouter web search tooling automatically.
  - `WEB_SEARCH_MAX_RESULTS` caps result count.
- Tools:
  - Tool schemas are built from Open WebUI’s `__tools__` registry plus any selected Open WebUI **Direct Tool Servers**.
  - Direct Tool Servers are executed client-side via Open WebUI (the pipe emits `execute:tool` via Socket.IO).
- Tool schema strictness:
  - When `ENABLE_STRICT_TOOL_CALLING=True`, the pipe strictifies tool schemas for more predictable function calling.

See also: [Tooling & Integrations](tooling_and_integrations.md).

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
