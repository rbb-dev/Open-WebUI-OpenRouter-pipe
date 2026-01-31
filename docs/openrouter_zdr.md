# OpenRouter Zero Data Retention (ZDR)

OpenRouter supports **Zero Data Retention (ZDR)** routing to ensure requests only hit endpoints that do not retain prompts or responses. This pipe exposes ZDR controls as admin valves and per-chat user valves so you can filter the model list and/or enforce ZDR on requests.

> **Quick navigation:** [Docs Home](README.md) · [Valves Atlas](valves_and_configuration_atlas.md) · [Provider Routing](openrouter_provider_routing.md)

---

## How it works

OpenRouter exposes ZDR-capable endpoints via the `/api/v1/endpoints/zdr` list. The pipe uses that list to:

- **Hide non‑ZDR models** when `ZDR_MODELS_ONLY` is enabled
- **Validate enforcement** before sending a request when ZDR is enforced
- **Attach `provider.zdr=true`** when ZDR is requested or enforced

> Note: The ZDR endpoint list is endpoint‑level; a model is treated as ZDR‑capable if at least one endpoint for that model appears in the ZDR list.

---

## Admin valves (pipe)

Configure these in **Open WebUI → Admin → Functions → [OpenRouter pipe] → Valves**:

- **`ZDR_MODELS_ONLY`**
  - Filters the model list to only ZDR‑capable models.
  - **Catalog filter only** — does not enforce ZDR on requests.

- **`ZDR_ENFORCE`**
  - Forces `provider.zdr=true` on every request.
  - Rejects requests for models without ZDR endpoints.

- **`ALLOW_USER_ZDR_OVERRIDE`**
  - Allows users to request ZDR per chat.
  - Ignored when `ZDR_ENFORCE` is enabled.

---

## User valve (per chat)

When `ALLOW_USER_ZDR_OVERRIDE` is enabled (and `ZDR_ENFORCE` is disabled), users can toggle:

- **`REQUEST_ZDR`**
  - Requests ZDR routing for that chat.

---

## Relationship to provider routing filters

Provider routing filters also expose a `ZDR` toggle that maps to `provider.zdr`. If you enable **both** the provider routing filter and pipe‑level ZDR enforcement, the pipe will force `provider.zdr=true` regardless of filter settings.

---

## OpenRouter docs

- ZDR overview: https://openrouter.ai/docs/guides/features/zdr
- ZDR endpoints list: https://openrouter.ai/api/v1/endpoints/zdr
- Provider routing reference: https://openrouter.ai/docs/guides/routing/provider-selection
