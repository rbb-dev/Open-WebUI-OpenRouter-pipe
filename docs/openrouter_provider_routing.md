# OpenRouter Provider Routing

OpenRouter supports **provider routing** — the ability to control which infrastructure providers serve your model requests and in what order. This is useful for:
- **Prioritizing specific providers** (e.g., always try OpenAI first, then Azure)
- **Excluding providers** (e.g., avoid providers that retain data)
- **Enforcing compliance** (e.g., only use Zero Data Retention endpoints)
- **Optimizing for cost/latency** (e.g., sort by price or latency)

The **Provider Routing Filters** integration generates dedicated Open WebUI filter functions for specific models, with dropdown menus populated from OpenRouter's live provider catalog.

> **Quick navigation:** [Docs Home](README.md) · [Valves Atlas](valves_and_configuration_atlas.md) · [Direct Uploads](openrouter_direct_uploads.md) · [Web Search](web_search_owui_vs_openrouter_search.md)

---

## How it works

When you configure provider routing for a model, the pipe:

1. **Fetches provider data** from OpenRouter's `/api/frontend/models` catalog
2. **Generates a dedicated filter** for each model with human-readable dropdown options
3. **Installs the filter** in Open WebUI's Functions database
4. **Attaches the filter** to the model so it appears in the Integrations menu

The generated filters expose OpenRouter's provider routing options as easy-to-use dropdowns and toggles — no need to guess provider slugs or remember API field names.

---

## Two-tier visibility control

Provider routing filters support **two visibility modes** that control who can configure and disable them:

| Mode | Who Configures | Can Users Disable? | Use Case |
|------|---------------|-------------------|----------|
| **Admin-only** | Admins only (Valves) | No (`toggle=False`) | Enforce provider policies |
| **User-configurable** | Users (UserValves) | Yes (`toggle=True`) | Let users choose preferences |
| **Both** | Both (user overrides admin) | Yes | Admin defaults + user choice |

When a model appears in **both** `ADMIN_PROVIDER_ROUTING_MODELS` and `USER_PROVIDER_ROUTING_MODELS`, the filter has both `Valves` (admin defaults) and `UserValves` (user overrides), with user settings taking precedence.

---

## Quick start

### Admin setup

1. In Open WebUI, go to **Admin → Functions → [Your OpenRouter Pipe] → Valves**

2. Configure one or both valves:
   ```
   ADMIN_PROVIDER_ROUTING_MODELS: openai/gpt-4o, anthropic/claude-3.5-sonnet
   USER_PROVIDER_ROUTING_MODELS: meta-llama/llama-3.2-3b-instruct, openai/gpt-4o
   ```

3. **Refresh models** (the pipe generates filters on model list refresh)

4. Verify filters appear in **Admin → Functions** (named `OpenRouter Provider: <model>`)

### User usage

1. Open a chat with a model that has a provider routing filter

2. In the **Integrations menu** (tools icon), enable the provider routing filter

3. Click the **filter settings** (gear icon) to configure:
   - **ORDER**: Provider priority (e.g., "OpenAI > Azure > Together")
   - **ALLOW_FALLBACKS**: Whether to use backup providers
   - **REQUIRE_PARAMETERS**: Only use providers supporting all request params
   - **ZDR**: Zero Data Retention mode

---

## Generated filter options

Each generated filter exposes OpenRouter's provider routing fields as user-friendly options:

### ORDER dropdown

The ORDER field is a dropdown showing all possible provider priority orderings. For a model with 3 providers (OpenAI, Azure, Together), you'll see:

```
(no preference)
OpenAI > Azure > Together
OpenAI > Together > Azure
Azure > OpenAI > Together
Azure > Together > OpenAI
Together > OpenAI > Azure
Together > Azure > OpenAI
```

The dropdown labels use provider display names (from OpenRouter's catalog), while the underlying API call uses provider slugs.

### Boolean toggles

| Field | Default | Purpose |
|-------|---------|---------|
| `ALLOW_FALLBACKS` | `True` | Use backup providers if preferred ones are unavailable |
| `REQUIRE_PARAMETERS` | `False` | Only use providers that support all request parameters |
| `ZDR` | `False` | Zero Data Retention — only use ZDR-compliant endpoints |

### Quantization (when available)

If the model has multiple quantization options (e.g., `fp16`, `bf16`, `int4`), a QUANTIZATIONS field appears allowing you to filter by quantization level.

---

## Valve reference

### Pipe valves (admin)

These are configured on the **pipe** function in Open WebUI (Admin → Functions → pipe → Valves).

| Valve | Type | Default | Purpose |
|-------|------|---------|---------|
| `ADMIN_PROVIDER_ROUTING_MODELS` | `str` | `""` | Comma-separated model slugs for admin-only provider routing filters. Users cannot override or disable these filters. |
| `USER_PROVIDER_ROUTING_MODELS` | `str` | `""` | Comma-separated model slugs for user-configurable provider routing filters. Users can toggle and configure these per-chat. |

### Generated filter valves (per filter)

Each generated filter has its own valves based on visibility:

#### Admin valves (`Valves`) — for admin-only or "both" visibility

| Valve | Type | Default | Maps to OpenRouter API |
|-------|------|---------|----------------------|
| `ORDER` | `Literal[...]` | `"(no preference)"` | `provider.order` |
| `ALLOW_FALLBACKS` | `bool` | `True` | `provider.allow_fallbacks` |
| `REQUIRE_PARAMETERS` | `bool` | `False` | `provider.require_parameters` |
| `ZDR` | `bool` | `False` | `provider.zdr` |
| `QUANTIZATIONS` | `Literal[...]` | `"(no preference)"` | `provider.quantizations` |

#### User valves (`UserValves`) — for user-configurable or "both" visibility

Same fields as admin valves, but configured by users per-chat.

---

## Data flow

### Filter inlet (before request)

When a provider routing filter runs:

1. Reads user valve settings from `__user__["valves"]` (OWUI-injected)
2. Builds a `provider` dict with non-default values
3. Injects into `__metadata__["openrouter_pipe"]["provider"]`
4. Returns the request body unchanged

### Pipe orchestrator (during request)

The pipe's orchestrator:

1. Reads `__metadata__["openrouter_pipe"]["provider"]`
2. Merges with any existing `provider` settings on the request
3. Adds to the outgoing OpenRouter API request

### Task model exclusion

Provider routing is **not applied** to task model requests (title generation, tags, follow-ups). These use different models that may not have the same providers available.

---

## Provider catalog source

Provider data comes from OpenRouter's `/api/frontend/models` endpoint, which returns one entry per model+provider combination:

```json
{
  "data": [
    {"slug": "openai/gpt-4o", "endpoint": {"provider_info": {"slug": "openai", "displayName": "OpenAI"}}},
    {"slug": "openai/gpt-4o", "endpoint": {"provider_info": {"slug": "azure", "displayName": "Azure"}}},
    ...
  ]
}
```

The pipe groups these by model slug to build the provider list for each model.

### Variant endpoint filtering

Endpoints with `model_variant_slug` set (e.g., Venice serving only `:free` variants) are **excluded** from the provider list for base models. This prevents showing providers that aren't actually available for the base model.

---

## Filter lifecycle

### Creation

Filters are generated when:
- Model list refreshes (via `pipes()`)
- Provider routing valves are configured
- Provider catalog data is available

### Updates

Filters are regenerated when:
- Provider catalog refreshes (new providers added/removed)
- Valve lists change (models added/removed from routing)

The pipe uses a **state hash** to avoid unnecessary database writes when nothing has changed.

### Removal

When a model is removed from routing valves:
- The filter is **detached** from the model
- The filter is **disabled** (`is_active=False`)
- The filter is **not deleted** — this preserves user settings if the model is re-added later

---

## Troubleshooting

### "No allowed providers are available for the selected model"

This OpenRouter error (HTTP 404) means your provider routing settings are too restrictive:

- **ORDER only**: If all providers in your ORDER list are unavailable, the request fails
- **ZDR mode**: Not all providers support Zero Data Retention
- **QUANTIZATIONS**: Not all providers offer all quantization levels

**Fix**: Enable `ALLOW_FALLBACKS` or use less restrictive settings.

### Filter not appearing for a model

Check:
1. The model slug is correctly spelled in the valve (e.g., `openai/gpt-4o` not `gpt-4o`)
2. The model is in the pipe's exposed models (`MODEL_ID` valve)
3. Provider catalog data is available (refresh models to trigger fetch)

### Filter settings not taking effect

User valves are read from `__user__["valves"]` (injected by OWUI), not from `self.user_valves`. If settings aren't working:

1. Ensure the filter is **enabled** in the Integrations menu
2. Verify settings are saved (click outside the input field)
3. Check that the model supports the requested provider/feature

### Provider slugs contain slashes

This is normal. OpenRouter uses slashes in provider slugs to indicate specific endpoints:
- `deepinfra/turbo` — DeepInfra's turbo endpoint
- `venice/beta` — Venice's beta endpoint (often `:free` variants only)

---

## OpenRouter Provider API reference

The generated filters expose a subset of OpenRouter's provider routing options. For full documentation, see [OpenRouter Provider Routing](https://openrouter.ai/docs/provider-routing).

| API Field | Type | Description |
|-----------|------|-------------|
| `order` | `string[]` | Provider slugs to try in order |
| `only` | `string[]` | Allow only specific providers |
| `ignore` | `string[]` | Ignore specific providers |
| `sort` | `string` | Sort by "price", "throughput", or "latency" |
| `quantizations` | `string[]` | Filter by quantization (int4, int8, fp8, bf16, etc.) |
| `allow_fallbacks` | `boolean` | Allow backup providers (default: true) |
| `require_parameters` | `boolean` | Only use providers supporting all params |
| `zdr` | `boolean` | Zero Data Retention enforcement |
| `data_collection` | `"allow" \| "deny"` | Control data storage policies |
| `max_price.prompt` | `number` | Max price for prompt ($/M tokens) |
| `max_price.completion` | `number` | Max price for completion ($/M tokens) |
| `max_price.image` | `number` | Max price per image ($/image) |
| `max_price.audio` | `number` | Max price for audio ($/unit) |
| `max_price.request` | `number` | Max price per request ($/request) |

---

## Security considerations

### External data

Provider names and slugs come from OpenRouter's external API. The pipe:
- Validates all strings before interpolation
- Uses `repr()` for safe Python literal escaping
- Validates generated filter source with `ast.parse()` before installation

### Code injection prevention

Generated filters are validated as syntactically-correct Python before being written to the database. Invalid or suspicious content is rejected.

### User valve isolation

User settings are scoped per-user and per-filter. One user's provider routing preferences don't affect other users.
