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
  - **Catalog filter and request admission** — a hidden model is also refused if requested directly. It never sends `provider.zdr=true`.

- **`ZDR_ENFORCE`**
  - Forces `provider.zdr=true` on every request.
  - Rejects requests for models without ZDR endpoints.
  - Routing suffixes the pipe synthesises (`:nitro`, `:floor`, `:online`) are checked against their base model: if the base has ZDR endpoints, the variant is admitted and `provider.zdr=true` guarantees only ZDR endpoints are used. A suffix OpenRouter lists as a model in its own right — `:free`, `:thinking` — is answered for **itself**, not for its base, so a listed `:free` with no ZDR endpoint is refused here rather than routed.
  - Video models are always rejected, with or without a variant suffix.
  - `ZDR_MODELS_ONLY` matches against the suffix-stripped base id, the same rule `ZDR_ENFORCE` uses, so routing variants (`:nitro`, `:floor`, `:online`) of a ZDR-capable base are shown and allowed. It stays a catalog and request-admission filter: it never sends `provider.zdr: true`, and it fails open when the ZDR endpoint list cannot be loaded, except for video models, which have no ZDR endpoints and stay hidden.

- **`ALLOW_USER_ZDR_OVERRIDE`**
  - Allows users to request ZDR per chat.
  - Ignored when `ZDR_ENFORCE` is enabled.
  - If a user's stored `REQUEST_ZDR` value cannot be parsed, the pipe cannot tell whether they opted in, so it enforces ZDR for that request rather than routing without it. A model that is not ZDR-capable is then refused with a `Restricted by` row naming the user's own `Request ZDR` preference rather than the `Enforce ZDR routing` valve, so an operator is not sent to a setting that is switched off. A failure to *read* the row is different: the preference Open WebUI supplied is used, so one unreadable settings row does not end that user's chat.

---

## User valve (per chat)

When `ALLOW_USER_ZDR_OVERRIDE` is enabled (and `ZDR_ENFORCE` is disabled), users can toggle:

- **`REQUEST_ZDR`**
  - Requests ZDR routing for that chat.

---

## Plugins and tools are outside ZDR

OpenRouter's ZDR enforcement applies to provider routing for inference only. Plugins and tools — including web search and the `:online` variant's search plugin — are operated by third-party services with their own data retention policies. Review those policies separately if you have strict retention requirements.

---

## Relationship to provider routing filters

Provider routing filters also expose a `ZDR` toggle that maps to `provider.zdr`. If you enable **both** the provider routing filter and pipe‑level ZDR enforcement, the pipe will force `provider.zdr=true` regardless of filter settings.

---

## OpenRouter docs

- ZDR overview: https://openrouter.ai/docs/guides/features/zdr
- ZDR endpoints list: https://openrouter.ai/api/v1/endpoints/zdr
- Provider routing reference: https://openrouter.ai/docs/guides/routing/provider-selection
