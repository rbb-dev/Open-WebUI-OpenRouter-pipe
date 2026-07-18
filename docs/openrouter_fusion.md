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

The filter has two distinct roles:

1. **On the fusion models (auto-attached): configuration only.** Preset, panel models, judge, and
   max tool calls shape *how* the deliberation runs. *Whether* it runs is not the filter's job —
   the pipe forces deliberation on every fusion-model chat request, so `Always run Fusion` is
   redundant there.
2. **On any other model (admin opt-in via `ALLOW_ON_NON_FUSION_MODELS`): fusion as an add-on
   tool.** The filter offers OpenRouter's fusion plugin to an ordinary model, which then decides
   per prompt whether to deliberate — unless the user turns `Always run Fusion` on, which is the
   only place that switch has an effect.

### Activation is guaranteed by the pipe

OpenRouter's fusion aliases answer as a plain single model on `/responses` unless the request
carries an explicit `{"id": "fusion"}` plugins entry — and even with it, invoking the deliberation
tool is left to the model's discretion, which in practice means it randomly answers plain. A
dedicated deliberation model that only sometimes deliberates is useless, so the pipe guarantees it:
every fusion-model chat request gets the plugins entry AND `tool_choice: "required"`. **Fusion
models always deliberate.** This holds even with the filter detached — the filter **configures**
Fusion (panel, judge); it does not gate it. The `openrouter:fusion` **server tool** on your own
model is a different surface and stays optional by design. Exceptions, in the pipe:

- housekeeping/task requests (title, tags, follow-up generation) get neither the entry nor the
  forcing — a chat title must not bill a full deliberation panel;
- with `ENABLE_OPENROUTER_FUSION` off, nothing is injected or forced — the master switch genuinely
  turns Fusion off;
- a caller-supplied Fusion entry (including `{"id": "fusion", "enabled": false}`) and a
  caller-supplied `tool_choice` are always left untouched, so an explicit opt-out disables
  deliberation for that request.

If a `/responses` request falls back to `/chat/completions`, the pipe strips the Fusion plugin entry:
Fusion on that endpoint returns a flattened text transcript with no structured events, so the fallback
answers as a normal completion instead of billing an unrenderable deliberation. A fusion request that
streams no deliberation events despite an active Fusion entry logs a warning naming the model — the
tripwire for the next time OpenRouter's beta behavior shifts; the fallback path suppresses it.

### Per-user options (UserValves)

Each user sets these per chat under the filter's controls (the **UI title** is what the user sees;
the **valve** is the underlying `UserValves` field name).

| Valve | UI title | Maps to | Notes |
|-------|----------|---------|-------|
| `FUSION_PRESET` | Preset | `preset` | `general-high` (strongest) or `general-budget` (cheaper). Empty = Fusion default. Explicit panel/judge below override a preset. |
| `FUSION_ANALYSIS_MODELS` | Panel models (comma-separated) | `analysis_models` | 1–8 model IDs answering in parallel. More than 8 is rejected with a clear error. Empty = preset/default panel. |
| `FUSION_JUDGE_MODEL` | Judge model | `model` | Model that reviews the panel and writes the analysis. Empty = Fusion default. |
| `FUSION_MAX_TOOL_CALLS` | Max tool calls per model | `max_tool_calls` | 1–16 web-search/fetch steps per inner model. `0` = Fusion default (8). |
| `FUSION_FORCE_TOOL_CALL` | Always run Fusion | `tool_choice="required"` | **No effect on the fusion models** — the pipe already forces deliberation there. Matters only on non-fusion models an admin attached the filter to (see below). |

### "Always run Fusion" (forcing)

On the fusion models the pipe already sets `tool_choice="required"` on every chat request, exactly
as [OpenRouter documents](https://openrouter.ai/docs/guides/routing/routers/fusion-router#forcing-fusion-on-every-request)
— this valve adds nothing there. It remains meaningful only when an admin attaches the filter to a
**non-fusion** model via `ALLOW_ON_NON_FUSION_MODELS`: there Fusion stays a tool the model may
decline, and turning **Always run Fusion** on sets `tool_choice="required"` for that chat.

Per OpenRouter: *"If your request also includes other tools, the model may pick one of those
instead."* — extra tools are escape hatches from `required`, and the `openrouter:*` server tools
(web search/fetch/datetime) are the worst offenders: a research prompt makes the model pick web
search over deliberation every time. The pipe closes that hole twice over: the Web Tools filter is
**never auto-attached to fusion models** (their panel and judge already run `openrouter:web_search`
and `openrouter:web_fetch` internally, so outer web tools add nothing), and any `openrouter:*`
server tools that still reach a fusion-model request — a manually attached filter, a leftover
per-chat toggle, a direct API caller — are **stripped before send**. The remaining caveat applies
only to OWUI-native function tools you attach yourself: with those present, the model may satisfy
`required` by calling one of them instead of deliberating.

Both the filter and the pipe leave a caller-supplied `tool_choice` / `function_call` untouched, and
skip forcing when the Fusion plugin is explicitly `enabled:false` (requiring a tool with no active
Fusion would just force some other tool).

## Enablement — pipe valves (admin)

These live on the **pipe** (the OpenRouter manifold's `Valves`) and control install/attach/default
wiring. They are documented alongside the other pipe valves in
[valves_and_configuration_atlas.md](valves_and_configuration_atlas.md).

| Valve | Default | Effect |
|-------|---------|--------|
| `ENABLE_OPENROUTER_FUSION` | `True` | Master switch; installs the filter, auto-wires it to `openrouter/fusion`, and gates the pipe's activation injection. `False` deactivates the installed filter on the next `pipes()` call and stops injecting the Fusion plugin entry — Fusion is then fully off. |
| `AUTO_INSTALL_FUSION_FILTER` | `True` | Install/update the filter function in OWUI. |
| `AUTO_ATTACH_FUSION_FILTER` | `True` | Attach the filter to the `openrouter/fusion` model **only** (never other models). |
| `AUTO_DEFAULT_FUSION_FILTER` | `True` | Pre-enable the filter per chat on `openrouter/fusion` (does not force Fusion to run). |

The dedicated `openrouter/fusion` model is auto-wired because access to it is already governed by Open
WebUI's model ACLs. The filter is **never** auto-attached to any other model.

### `openrouter/fusion-flash` (forward-compat)

OpenRouter documents a faster `openrouter/fusion-flash` alias (the `general-fast` preset pinned as its
own model), but it is not live on the API yet. The pipe already treats it as a full member of the
fusion model family — endpoint forcing, the live panel, filter auto-wiring, the activation injection,
and `tool_choice: "required"` all apply automatically once OpenRouter ships it and it appears in the
catalog. Filter updates reach installed deployments via `AUTO_INSTALL_FUSION_FILTER`; attach-only
deployments (auto-install off) keep their existing filter copy, which does not recognize flash until
it is reinstalled.

## Live deliberation panel

Every `openrouter/fusion` chat automatically renders the deliberation — preamble intent, per-model panels, the
judge's analysis, the final answer, and cost — as a live, theme-aware HTML panel. The panel iframe is emitted
**once** and then updated in place: each deliberation event is pushed over Open WebUI's own socket as a custom
`fusion:event` that the panel's same-origin socket connection consumes (no iframe reload → no flashing). The
final answer streams **into** the panel and is also written to the message as a **collapsed `<details>`** — so
multi-turn context, copy, and regenerate read the answer natively (the panel embed is UI-only and is never sent
back to the model), while the visible surface stays the panel.

While the panel deliberates, each model's card is **live**: its status line becomes a ticker showing
the tail of whatever that model is currently producing plus a running word count, streamed from
OpenRouter's per-token panel events (batched server-side to a few updates per second per model).
Models that expose reasoning gain a collapsible **Thinking** section on their card — hidden until
reasoning actually arrives, streaming live while expanded, rendered as Markdown on first open, and
kept in the persisted panel for every completed model (a panel still mid-answer at the moment of a
reload recovers its reasoning when it completes). The Thinking section has its own
copy button, and **Copy all** includes each model's thinking alongside its answer. The high-volume
token deltas themselves are never baked into the persisted embed — the full reasoning text is
reattached to each panel's completed event instead, keeping the snapshot small. If OpenRouter stops
streaming panel deltas, the cards simply fill in at completion as before.

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
