# Web Search: Open WebUI vs OpenRouter

Open WebUI has a built-in **Web Search** feature. Separately, OpenRouter provides a **Web Search server tool** that the model can call during generation. This pipe supports both, and intentionally keeps them separate to avoid ambiguity.

> **Quick navigation:** [Docs Home](README.md) · [Server Tools](openrouter_server_tools.md) · [Valves Atlas](valves_and_configuration_atlas.md)

---

## Two different systems

### 1) Open WebUI Web Search (OWUI-native)

- Open WebUI's own web-search pipeline.
- When enabled, OWUI runs a web search **before** calling the model, then attaches the results to the request as context.
- Does not require the model or provider to support any special tool feature.
- Configured in Open WebUI's settings (search engine, API keys, etc.).

### 2) OpenRouter Web Search (server tool)

- An OpenRouter server tool passed in the `tools` array of the API request.
- The **model** decides when to search (tool calling), and OpenRouter executes the search server-side.
- Supports engine selection (`auto`, `native`, `exa`, `firecrawl`, `parallel`), result limits, domain restrictions, and location-aware results.
- Configured via the **OpenRouter Web Tools** companion filter (admin valves for engines/limits, user valves for per-chat toggles and preferences).

---

## The rule: OpenRouter Web Search suppresses OWUI Web Search

When OpenRouter Web Search is enabled for a request, the Web Tools filter sets `body["features"]["web_search"] = False`, which suppresses Open WebUI's native web-search handler. This prevents:

- running two searches (one OWUI, one OpenRouter),
- paying twice,
- ambiguous citation sources.

If OpenRouter Web Search is **disabled** (user turns off the `WEB_SEARCH` toggle in the filter), OWUI Web Search is left untouched and works normally.

---

## How OpenRouter Web Search is surfaced in the UI

Open WebUI does not provide a "pipe can inject new toggles" frontend extension point. The only supported UI injection points are tool registry entries and **toggleable filter functions**.

The pipe implements OpenRouter Web Search as part of the **OpenRouter Web Tools** toggleable filter:

- The pipe can **auto-install / auto-update** this filter when `AUTO_INSTALL_WEB_TOOLS_FILTER` is enabled.
- The pipe can **auto-attach** it to pipe models when `AUTO_ATTACH_WEB_TOOLS_FILTER` is enabled.
- The pipe can **enable it by default** on models when `AUTO_DEFAULT_WEB_TOOLS_FILTER` is enabled.

Users see an "OpenRouter Web Tools" switch in the Integrations menu. Individual tools (Web Search, Web Fetch, Datetime) are toggled via the filter's user valves.

---

## Per-model overrides

Two per-model custom parameters (set in Open WebUI model Advanced Parameters) control Web Tools filter attachment on a per-model basis:

| Parameter | Effect |
| --- | --- |
| `disable_openrouter_search_auto_attach` | Prevents auto-attaching the Web Tools filter to this model. The toggle will not appear in the Integrations menu for this model. As a consequence, default-on seeding is also skipped. |
| `disable_openrouter_search_default_on` | Prevents auto-enabling the Web Tools filter by default for this model. The toggle appears but starts off; users can still enable it per chat. |

These parameters are respected even when the global `AUTO_ATTACH_WEB_TOOLS_FILTER` and `AUTO_DEFAULT_WEB_TOOLS_FILTER` pipe valves are enabled.

---

## When to use which

### Use OpenRouter Web Search when:

- You want the **model** to decide when to search (tool calling).
- You want search results integrated naturally into the model's response.
- You want engine selection, domain restrictions, and location-aware results.
- The model supports tool calling (most modern models do).

### Use OWUI Web Search when:

- You want search results injected as context **before** the model sees the prompt.
- You want to use OWUI's configured search engine (Google, Bing, SearXNG, etc.).
- The model does not support tool calling.
- You prefer a deterministic "always search" behavior rather than model-decided searching.

### Use both (advanced):

Not recommended. When both are enabled on the same request, the Web Tools filter suppresses OWUI Web Search to avoid double-searching. If you need OWUI Web Search for specific models, use `disable_openrouter_search_auto_attach` on those models to prevent the Web Tools filter from being attached.

---

## Recommended operator settings

### OpenRouter Web Tools available but opt-in (current defaults)

- `AUTO_INSTALL_WEB_TOOLS_FILTER=True`
- `AUTO_ATTACH_WEB_TOOLS_FILTER=True`
- `AUTO_DEFAULT_WEB_TOOLS_FILTER=False`

Result: Users see **OpenRouter Web Tools** on all pipe models but must enable it per chat. Admin can set `AUTO_DEFAULT_WEB_TOOLS_FILTER=True` to pre-enable it for all models.

### Enable OpenRouter Web Tools by default

- Set `AUTO_DEFAULT_WEB_TOOLS_FILTER=True`.
- Web Search and Datetime start enabled by default. Users can disable per chat. OWUI Web Search is suppressed when OpenRouter Web Search is active.

### Prefer OWUI Web Search for specific models

- Set `disable_openrouter_search_auto_attach` in the model's Advanced Parameters.
- The Web Tools toggle will not appear for that model, and OWUI Web Search will work normally.

---

## Troubleshooting

### "I don't see the OpenRouter Web Tools toggle"

- Confirm `AUTO_INSTALL_WEB_TOOLS_FILTER=True` and `AUTO_ATTACH_WEB_TOOLS_FILTER=True` on the pipe valves.
- Trigger a model catalog refresh (the filter is installed/attached during refresh).
- Check that the model does not have `disable_openrouter_search_auto_attach` set in its Advanced Parameters.

### "OWUI Web Search doesn't work when OpenRouter Web Tools is enabled"

- This is by design. The Web Tools filter suppresses OWUI Web Search when OpenRouter Web Search is active, to prevent double-searching.
- To use OWUI Web Search instead, disable the `WEB_SEARCH` user valve in the filter's settings, or disable the Web Tools toggle entirely for that chat.

### "I see the old OpenRouter Search filter"

- The old filter is automatically disabled on pipe startup. If it persists, manually deactivate it in Admin > Functions.
- The old pipe valves (`AUTO_ATTACH_ORS_FILTER`, `AUTO_INSTALL_ORS_FILTER`, `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER`) no longer exist; use the new `AUTO_*_WEB_TOOLS_FILTER` valves instead.
