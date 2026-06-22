# OpenRouter Fusion

[OpenRouter Fusion](https://openrouter.ai/docs/guides/routing/routers/fusion-router) turns a prompt
into a small **multi-model deliberation**: a *panel* of up to 8 models answers in parallel (with web
search/fetch), a *judge* model produces a structured analysis (consensus, disagreements, gaps, blind
spots), and the outer model writes the final answer from it.

> **Cost:** Fusion bills the sum of every underlying call — roughly **4–5× a single completion**, and
> it scales with panel size. Treat it as an expensive feature.

## The "OpenRouter Fusion" filter

The pipe ships a dedicated Open WebUI filter, **OpenRouter Fusion** (`openrouter_fusion`), that exposes
Fusion's options as UI knobs. It injects a `{"id": "fusion", …}` entry into the request's `plugins`
array (the same mechanism as the Web Tools filter). It acts on the **`openrouter/fusion`** model and —
unless an admin opts in — no-ops on every other model.

### Per-user options (UserValves)

Each user sets these per chat under the filter's controls (the **UI title** is what the user sees;
the **valve** is the underlying `UserValves` field name).

| Valve | UI title | Maps to | Notes |
|-------|----------|---------|-------|
| `FUSION_PRESET` | Preset | `preset` | `general-high` (strongest) or `general-budget` (cheaper). Empty = Fusion default. Explicit panel/judge below override a preset. |
| `FUSION_ANALYSIS_MODELS` | Panel models (comma-separated) | `analysis_models` | 1–8 model IDs answering in parallel. More than 8 is rejected with a clear error. Empty = preset/default panel. |
| `FUSION_JUDGE_MODEL` | Judge model | `model` | Model that reviews the panel and writes the analysis. Empty = Fusion default. |
| `FUSION_MAX_TOOL_CALLS` | Max tool calls per model | `max_tool_calls` | 1–16 web-search/fetch steps per inner model. `0` = Fusion default (8). |
| `FUSION_FORCE_TOOL_CALL` | Always run Fusion | `tool_choice="required"` | **Off by default.** See below. |

### "Always run Fusion" (forcing)

By default Fusion is a *tool the model may decline to call* — for simple prompts it answers directly
and Fusion never runs. Turning **Always run Fusion** on sets `tool_choice="required"` on the request,
exactly as [OpenRouter documents](https://openrouter.ai/docs/guides/routing/routers/fusion-router#forcing-fusion-on-every-request).

On the `openrouter/fusion` model Fusion is the **only** injected tool, so requiring *some* tool call
forces Fusion. Per OpenRouter: *"If your request also includes other tools, the model may pick one of
those instead."* So if you also enable other tool integrations on the same chat (e.g. the OpenRouter
Web Tools filter, or OWUI-native tools), deliberation is no longer guaranteed — for predictable
forcing, run Fusion without other tools enabled.

The filter leaves a caller-supplied `tool_choice` / `function_call` untouched, and skips forcing when
the Fusion plugin is explicitly `enabled:false` (requiring a tool with no active Fusion would just
force some other tool).

## Enablement — pipe valves (admin)

These live on the **pipe** (the OpenRouter manifold's `Valves`) and control install/attach/default
wiring. They are documented alongside the other pipe valves in
[valves_and_configuration_atlas.md](valves_and_configuration_atlas.md).

| Valve | Default | Effect |
|-------|---------|--------|
| `ENABLE_OPENROUTER_FUSION` | `True` | Master switch; installs the filter and auto-wires it to `openrouter/fusion`. `False` deactivates the installed filter on the next `pipes()` call. |
| `AUTO_INSTALL_FUSION_FILTER` | `True` | Install/update the filter function in OWUI. |
| `AUTO_ATTACH_FUSION_FILTER` | `True` | Attach the filter to the `openrouter/fusion` model **only** (never other models). |
| `AUTO_DEFAULT_FUSION_FILTER` | `True` | Pre-enable the filter per chat on `openrouter/fusion` (does not force Fusion to run). |

The dedicated `openrouter/fusion` model is auto-wired because access to it is already governed by Open
WebUI's model ACLs. The filter is **never** auto-attached to any other model.

## Live deliberation panel

Every `openrouter/fusion` chat automatically renders the deliberation — preamble intent, per-model panels, the
judge's analysis, the final answer, and cost — as a live, theme-aware HTML panel. The panel iframe is emitted
**once** and then updated in place: each deliberation event is pushed over Open WebUI's own socket as a custom
`fusion:event` that the panel's same-origin socket connection consumes (no iframe reload → no flashing). The
final answer streams **into** the panel and is also written to the message as a **collapsed `<details>`** — so
multi-turn context, copy, and regenerate read the answer natively (the panel embed is UI-only and is never sent
back to the model), while the visible surface stays the panel.

- The live panel requires Open WebUI's **iframe same-origin** setting (Settings → Interface → "iframe sandbox
  allow same origin") — the panel reads the session token to open its socket. With it off, the panel still
  renders the complete deliberation **statically** on completion / page reload (from the persisted embed) — just
  not live.
- A **browser close** mid-run does not abort the deliberation: Open WebUI runs it as a detached task, so it
  finishes server-side and the full panel + answer are persisted; reopening the chat shows the finished result.
- A mid-stream **socket drop** has no live replay; reloading restores the complete panel from the persisted state.

- Forces the `/responses` endpoint (the only one that emits the granular Fusion events). A
  `FORCE_CHAT_COMPLETIONS_MODELS` match on the fusion model is overridden to `/responses` with a warning log.
- No effect on **Direct Connections** — Open WebUI does not deliver in-chat embeds on that path; the final answer is unaffected.
- Automatic on `openrouter/fusion` whenever Fusion is enabled — the master `ENABLE_OPENROUTER_FUSION` switch turns it off along with the rest of Fusion. Non-fusion models are never affected.

### Socket transport — network & CSP requirements (admin)

The live panel runs inside Open WebUI's sandboxed embed iframe and opens its **own** authenticated
Socket.IO connection back to this instance to receive `fusion:event` updates. It authenticates the
**same way Open WebUI's own client does**: it reads the signed-in user's session token from
`localStorage` and sends it as the Socket.IO handshake `auth: { token }`, which the server decodes to
join that user's event room. No separate credential is minted or embedded — which is also why the
**iframe same-origin** setting (above) is required: without it the iframe is a distinct origin, cannot
read the token, and the panel stays static.

Because the embed is a `srcdoc` document (origin `about:srcdoc`), it cannot reuse Open WebUI's bundled,
module-scoped `socket.io-client`. Instead the Socket.IO client is **inlined directly into the panel
HTML at build time** — there is **no runtime CDN fetch and no external script dependency**. The client
is **version-pinned** (socket.io-client 4.8.3) and its exact bytes are **verified against a pinned SHA-384 digest at build time**, so a
substituted or tampered client can never be inlined into the shipped template. (Inlining adds
no new outbound network requirement — at runtime the panel only needs the WebSocket back to this
instance.)

By default this needs no configuration — Open WebUI injects **no** CSP into embeds unless `IFRAME_CSP`
is set (an empty policy is returned unchanged). **If you set `IFRAME_CSP`**, the live panel requires:

| Capability | Directive | Minimum to allow |
|------------|-----------|------------------|
| Panel's own inline script (includes the inlined Socket.IO client) | `script-src` | `'unsafe-inline'` |
| WebSocket back to this instance | `connect-src` | this origin (`'self'`; also add `wss:`/`ws:` if your policy is scheme-explicit) |

Because the Socket.IO client is inlined rather than fetched, a single `script-src 'unsafe-inline'`
covers both the panel logic and the socket client — no external script host needs allow-listing.

Example scoped policy:

```
IFRAME_CSP="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:"
```

The failure mode is **non-fatal**: if `connect-src` omits this origin (or `script-src` omits
`'unsafe-inline'`) the socket never opens, and the panel degrades to the fully-rendered **static**
deliberation (from the persisted embed) with no live streaming and no error. This is the same fallback
as running with the **iframe same-origin** setting off (above), where the panel cannot read the session
token to authenticate the socket.

## Filter admin valves (on the filter itself)

These are **separate** from the pipe valves above. They live on the **filter's own** `Valves`, edited
at *Admin → Functions → OpenRouter Fusion → ⚙ valves*. They are not per-user — they apply to every
chat that uses the filter.

| Valve | Default | Effect |
|-------|---------|--------|
| `ALLOW_ON_NON_FUSION_MODELS` | `False` | **Off (default):** the filter acts only on `openrouter/fusion`. If you attach it to **any other model**, its inlet returns immediately and injects nothing — no Fusion plugin, no `tool_choice` forcing — so a user's *Preset / Panel / Judge / **Always run Fusion*** settings have **no effect** on that model. **On:** the filter adds the Fusion panel (and forcing, if the user enabled *Always run Fusion*) to **any** model it is attached to. This valve is the **only** way to use Fusion on a model other than `openrouter/fusion`: manually attach the filter to that model, then turn this on. (The model must also support tool calling, since forcing sets `tool_choice="required"`.) |
| `priority` | `0` | Filter execution order. OWUI runs a chat's attached filters sorted by `(priority, id)`, lowest first. |

> **Gotcha:** attaching the filter to another model and enabling *Always run Fusion* does **nothing**
> while `ALLOW_ON_NON_FUSION_MODELS` is `False` — the inlet bails out before it can inject the plugin
> or set `tool_choice`. Turn this valve on first.
