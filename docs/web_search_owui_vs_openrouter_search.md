# Web Search: Open WebUI vs OpenRouter

Open WebUI has a built-in **Web Search** feature. Separately, OpenRouter provides a **Web Search server tool** that the model can call during generation. This pipe supports both, and intentionally keeps them separate to avoid ambiguity.

> **Quick navigation:** [Docs Home](README.md) · [Server Tools](openrouter_server_tools.md) · [Valves Atlas](valves_and_configuration_atlas.md)

---

## Two different systems

### 1) Open WebUI Web Search (OWUI-native)

- Open WebUI's own web search, configured in Open WebUI's settings (search engine, API keys, etc.).
- By default (native function calling), Open WebUI hands the model its `search_web` and `fetch_url` tools, and the model decides when to search. In the pipe's default `Pipeline` tool mode, the pipe runs them. Search hits are not shown as sources (Open WebUI's own chats do not show them either), but a page the model fetches is — unless the model's `Citations` box is unticked, which withholds a fetched page's source as well.
- Only for a model set to legacy function calling does Open WebUI search **before** the model is called, attaching the results to the request as context. That form needs no tool support from the model.

### 2) OpenRouter Web Search (server tool)

- An OpenRouter server tool passed in the `tools` array of the API request.
- The **model** decides when to search (tool calling), and OpenRouter executes the search server-side.
- Supports engine selection (`auto`, `native`, `exa`, `firecrawl`, `parallel`, `perplexity`), result limits, per-result character caps, domain restrictions, and location-aware results.
- Configured via the **OpenRouter Web Tools** companion filter (admin valves for engines/limits, user valves for per-chat toggles and preferences).

---

## The rule: OpenRouter Web Search suppresses OWUI Web Search

When OpenRouter Web Search is enabled for a request, the Web Tools filter sets `body["features"]["web_search"] = False`, which turns off Open WebUI's own web search for that request in both forms (its tools and its legacy pre-search). This prevents:

- running two searches (one OWUI, one OpenRouter),
- paying twice,
- ambiguous citation sources.

If OpenRouter Web Search is **disabled** (the user turns off the `WEB_SEARCH` toggle in the filter, or an admin switches `ENABLE_WEB_SEARCH` off on the pipe), the filter no longer sets that flag, so OWUI Web Search is left untouched and works normally. The pipe rewrites every Web Tools filter it installed, and every Web Tools row that carries no install record, so the suppression stops everywhere it is this pipe's to stop and not only in the row you happened to look at.

---

## How OpenRouter Web Search is surfaced in the UI

Open WebUI does not provide a "pipe can inject new toggles" frontend extension point. The only supported UI injection points are tool registry entries and **toggleable filter functions**.

The pipe implements OpenRouter Web Search as part of the **OpenRouter Web Tools** toggleable filter:

- The pipe can **auto-install / auto-update** this filter when `AUTO_INSTALL_WEB_TOOLS_FILTER` is enabled. A switch-off reaches every copy this pipe installed, and every copy that carries no install record, whether it is on or off, and a copy this version switched off comes back on its own when a web tool is on again, including one installed by hand and with `AUTO_INSTALL_WEB_TOOLS_FILTER` off.
- The pipe can **auto-attach** it to pipe models when `AUTO_ATTACH_WEB_TOOLS_FILTER` is enabled.
- The pipe can **enable it by default** on models when `AUTO_DEFAULT_WEB_TOOLS_FILTER` is enabled.

Users see an "OpenRouter Web Tools" switch in the Integrations menu. Individual tools (Web Search, Web Fetch, Datetime, Advisor, Subagent, Search Models) are toggled via the filter's user valves.

---

## Per-model overrides

Two per-model custom parameters (set in Open WebUI model Advanced Parameters) control Web Tools filter attachment on a per-model basis:

| Parameter | Effect |
| --- | --- |
| `disable_web_tools_auto_attach` | Prevents auto-attaching the Web Tools filter to this model. The toggle will not appear in the Integrations menu for this model. As a consequence, default-on seeding is also skipped, and any default the pipe had already seeded for this model is left alone. |
| `disable_web_tools_default_on` | Prevents auto-enabling the Web Tools filter by default for this model. The toggle appears but starts off on the next sync — that is, the next catalogue or settings change, or a worker restart; users can still enable it per chat. A default the pipe had already seeded for this model is released on that same sync, and comes back on the next such sync if the parameter is later removed. |

These parameters are respected even when the global `AUTO_ATTACH_WEB_TOOLS_FILTER` and `AUTO_DEFAULT_WEB_TOOLS_FILTER` pipe valves are enabled: `disable_web_tools_default_on` releases a default the pipe seeded for that model on the next sync, and neither of them detaches a panel the pipe attached — the next sync being the next catalogue or settings change, or a worker restart. A default seeded under an id the panel no longer resolves is released on that same sync, including after the panel has been reinstalled under a new id, and turning the valve back on reclaims a default re-ticked in between so the next turn-off still removes it.

---

## When to use which

### Use OpenRouter Web Search when:

- You want the **model** to decide when to search (tool calling).
- You want search results integrated naturally into the model's response.
- You want engine selection, domain restrictions, and location-aware results.
- The model supports tool calling (most modern models do).

### Use OWUI Web Search when:

- You want to use OWUI's configured search engine (Google, Bing, SearXNG, etc.).
- The model does not support tool calling, or you prefer a deterministic "always search" behavior rather than model-decided searching: set the model to legacy function calling, so Open WebUI searches **before** the model sees the prompt and injects the results as context.

### Use both (advanced):

Not recommended. When both are enabled on the same request, the Web Tools filter suppresses OWUI Web Search to avoid double-searching. If you need OWUI Web Search for specific models, use `disable_web_tools_auto_attach` on those models to prevent the Web Tools filter from being attached.

---

## Recommended operator settings

### OpenRouter Web Tools available but opt-in (current defaults)

- `AUTO_INSTALL_WEB_TOOLS_FILTER=True`
- `AUTO_ATTACH_WEB_TOOLS_FILTER=True`
- `AUTO_DEFAULT_WEB_TOOLS_FILTER=False`

Result: Users see **OpenRouter Web Tools** on every pipe model that is not a picture-only model, a video-generation or a Fusion model (a model that answers with text as well as pictures is a chat model that can draw, and it keeps the switch) but must enable it per chat. Admin can set `AUTO_DEFAULT_WEB_TOOLS_FILTER=True` to pre-enable it for every model that gets the switch, and turning it off again removes that pre-enablement from the models it was applied to on the next sync — the next sync being the next catalogue or settings change, or a worker restart — in either direction: a default re-ticked by an operator is reclaimed when the valve goes back on, so the next turn-off removes it again. A model that is later excluded with `disable_web_tools_default_on`, or whose panel is detached by `AUTO_ATTACH_WEB_TOOLS_FILTER=False`, loses the default the pipe seeded for it on the next sync rather than keeping it default-on against a panel that no longer runs.

### Enable OpenRouter Web Tools by default

- Set `AUTO_DEFAULT_WEB_TOOLS_FILTER=True`.
- Web Search and Datetime start enabled by default. Users can disable per chat. OWUI Web Search is suppressed when OpenRouter Web Search is active.

### Prefer OWUI Web Search for specific models

- Set `disable_web_tools_auto_attach` in the model's Advanced Parameters.
- The Web Tools toggle will not appear for that model, and OWUI Web Search will work normally.

---

## Troubleshooting

### "I don't see the OpenRouter Web Tools toggle"

- Confirm `AUTO_INSTALL_WEB_TOOLS_FILTER=True` and `AUTO_ATTACH_WEB_TOOLS_FILTER=True` on the pipe valves.
- Trigger a model catalog refresh (the filter is installed/attached during refresh).
- Check that the model does not have `disable_web_tools_auto_attach` set in its Advanced Parameters.

### "OWUI Web Search doesn't work when OpenRouter Web Tools is enabled"

- This is by design. The Web Tools filter suppresses OWUI Web Search when OpenRouter Web Search is active, to prevent double-searching.
- To use OWUI Web Search instead, disable the `WEB_SEARCH` user valve in the filter's settings, or disable the Web Tools toggle entirely for that chat.

### "Open WebUI's search stopped after OpenRouter web search was switched off"
- Expected, and it clears itself. The Web Tools filter suppresses Open WebUI's search, so a filter that still offers
  web search keeps it off even after the switch.
- The pipe rewrites every Web Tools filter it installed, and every Web Tools row that carries no install
  record, without the switched-off tool at the next model-list refresh, or in the background after the first
  message that still asks for it. A message sent before the rewrite lands gets neither
  search; later ones get Open WebUI's.
- If a log warning says a filter's code could not be read, the pipe left that filter alone: update it or remove it.
- If you refreshed and the toggle is still there, the row you are looking at is often the **wrong row**. With several
  Web Tools copies installed, the pipe maintains one and leaves the others alone; the pipe's log line names the row it
  actually wrote. Look for that one.
- If the row you need is one **you** switched off, the pipe will not bring it back: it keeps that row's code current but
  never switches it on. Switch it on in Open WebUI's Functions list.

### "I see the old OpenRouter Search filter"

- The old filter is automatically disabled on pipe startup. If it persists, manually deactivate it in Admin > Functions.
- The old pipe valves (`AUTO_ATTACH_ORS_FILTER`, `AUTO_INSTALL_ORS_FILTER`, `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER`) no longer exist; use the new `AUTO_*_WEB_TOOLS_FILTER` valves instead.
