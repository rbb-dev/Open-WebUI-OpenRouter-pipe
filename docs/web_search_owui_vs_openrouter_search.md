# Web Search (Open WebUI) vs OpenRouter Search

Open WebUI exposes a built-in **Web Search** toggle in the Integrations menu. Separately, OpenRouter offers a provider-native web-search capability for some models (via the OpenRouter `plugins: [{ "id": "web" }]` mechanism).

This pipe supports **both**, and intentionally keeps them separate to avoid ambiguity.

---

## Two toggles, two different systems

### 1) Web Search — Open WebUI native

- This is Open WebUI’s own web-search pipeline.
- When enabled, OWUI runs a web search **before** calling the model, then attaches the results to the request as files/context.
- It does not require the model/provider to support any special “web tool” feature.

### 2) OpenRouter Search — provider-native plugin

- This is OpenRouter’s provider-native web-search plugin (`{ "id": "web" }`) on the Responses API.
- Only some OpenRouter models support it (the pipe detects support from OpenRouter’s catalog).
- Results appear through the provider’s response/tool artifacts.

---

## The rule: OpenRouter Search overrides Web Search (no double-search)

If both toggles are enabled, the pipe enforces **OpenRouter Search only** to prevent:

- running two searches,
- paying twice (OWUI search + OpenRouter search),
- confusing “which citations came from where?” outcomes.

Implementation detail:
- The OpenRouter Search companion filter disables OWUI’s `features.web_search` for the request (preventing OWUI’s native web-search handler), while still signaling the pipe to enable the OpenRouter plugin.

---

## How OpenRouter Search is surfaced in the Integrations menu

Open WebUI does not provide a “pipe can inject new toggles” frontend extension point. The only supported UI injection points are:

- Tools (tool registry / tool servers), and
- **toggleable filter functions** (shown as switches in Integrations).

So OpenRouter Search is implemented as a **toggleable filter function**:

- The pipe can **auto-install / auto-update** this filter into Open WebUI’s Functions DB when `AUTO_INSTALL_ORS_FILTER` is enabled.
- The pipe can **auto-enable** it in each compatible model’s settings when `AUTO_ATTACH_ORS_FILTER` is enabled.
- The pipe can **enable it by default** on compatible models when `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER` is enabled.

This keeps the User Interface clean: users don’t see an OpenRouter Search switch for models where it can’t work.

---

## Recommended operator settings

### OpenRouter Search enabled by default (current defaults)

- `AUTO_INSTALL_ORS_FILTER=true`
- `AUTO_ATTACH_ORS_FILTER=true`
- `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER=true`

Result:
- Users see **OpenRouter Search** only on models that support OpenRouter-native web search.
- OpenRouter Search starts enabled by default on those models, but can still be turned off per chat.
- Users can still use Open WebUI-native **Web Search** on any model.

These are the current defaults.

### Make OpenRouter Search opt-in (lower cost)

Result:
- Set `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER=false`.
- The OpenRouter Search switch remains available on compatible models, but will not be enabled by default.
- If you previously enabled it by default on a specific model (via the model’s **Default Filters**), that per-model setting is respected until you change it.

---

## Troubleshooting

### “I enabled AUTO_ATTACH_ORS_FILTER but don’t see the OpenRouter Search toggle”

- You likely don’t have the companion filter installed.
- Enable `AUTO_INSTALL_ORS_FILTER`, save valves, then trigger a models refresh.

### “I see OpenRouter Search toggle on models that don’t support it”

- By design, the pipe avoids this by auto-enabling the OpenRouter Search filter only on models that support OpenRouter web search.
- If you manually attached the filter to a model, remove it from the model’s Filters list (or let the next refresh remove it when `AUTO_ATTACH_ORS_FILTER` is enabled).
