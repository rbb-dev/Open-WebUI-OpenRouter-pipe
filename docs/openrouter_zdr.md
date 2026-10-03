# OpenRouter Zero Data Retention (ZDR)

OpenRouter supports **Zero Data Retention (ZDR)** routing to ensure requests only hit endpoints that do not retain prompts or responses. This pipe exposes ZDR controls as admin valves and per-chat user valves so you can filter the model list and/or enforce ZDR on requests.

> **Quick navigation:** [Docs Home](README.md) · [Valves Atlas](valves_and_configuration_atlas.md) · [Provider Routing](openrouter_provider_routing.md)

---

## How it works

OpenRouter exposes ZDR-capable endpoints via the `/api/v1/endpoints/zdr` list. The pipe uses that list to:

- **Hide non‑ZDR models** when `ZDR_MODELS_ONLY` is enabled
- **Validate enforcement** before sending a request when ZDR is enforced
- **Attach `provider.zdr=true`** when ZDR is requested or enforced, on the transports whose schema defines that key

> Note: The ZDR endpoint list is endpoint‑level; a model is treated as ZDR‑capable if at least one endpoint for that model appears in the ZDR list.

The video help card a person gets from `help` reports that same verdict in the same
three states — ZDR-capable, not ZDR-capable, or not established because the list has
never been read. It is a *report*, not a gate: it changes no routing and refuses
nothing -- and it reports the roster's verdict only. Roster-capable is not the same
thing as enforceable: a model whose answer goes on the video or image transport is
refused under enforcement even when the roster names it. Only the valves above gate
anything.

---

## Admin valves (pipe)

Configure these in **Open WebUI → Admin → Functions → [OpenRouter pipe] → Valves**:

- **`ZDR_MODELS_ONLY`**
  - Filters the model list to only ZDR‑capable models.
  - **Catalog filter and request admission** — a hidden model is also refused if requested directly. It never sends `provider.zdr=true`.

- **`ZDR_ENFORCE`**
  - Forces `provider.zdr=true` on every request.
  - Rejects requests for models without ZDR endpoints.
  - On a Fusion run the refusal is reported per member, naming the privacy decision, so the judge and the synthesis model are told the pipe refused rather than that the model declined.
  - **What a failed read means.** A read that did not succeed carries the last read that did, so a `500`/`429` on `/models`, or an outage of `/endpoints/zdr`, no longer makes the pipe forget what it knew: a model that has been answering for an hour keeps its ZDR answer, and every enforced request still carries `provider.zdr=true`, which is what makes OpenRouter hold it to a no-retention endpoint. A request is refused outright only when **no ZDR list has ever been read** — that is what `zdr_list_available() is False` now means, and it is the cold-start path this behaviour deliberately does not weaken. The limit of the carry-forward is worth stating plainly: it can still prove a model is *not* ZDR-capable (and that stays refused), but a model that genuinely lost its ZDR endpoints is noticed only once a read succeeds.
  - Routing suffixes the pipe synthesises (`:nitro`, `:floor`, `:online`) are checked against their base model: if the base has ZDR endpoints, the variant is admitted and `provider.zdr=true` guarantees only ZDR endpoints are used. A suffix OpenRouter lists as a model in its own right — `:free`, `:thinking` — is answered for **itself**, not for its base, so a listed `:free` with no ZDR endpoint is refused here rather than routed. A `~`-prefixed `-latest` id is answered for the model its catalog `alias_target` names and for its own key, and only here: every other capability read still answers from the alias row.
  - Video models are answered from the same list as every other model, with or without a variant suffix, and a video model whose ZDR endpoints are on the list is admitted by the gate exactly like any other. It is then refused on the next line, because the transport cannot carry the control: OpenRouter's video schema defines one provider property, `options`, and no `zdr`, so a `zdr` sent there would be accepted and ignored -- the job would read as protected in OpenRouter's own logs while nothing enforced it. An **image-only** model (one that emits images and no text) is refused the same way, on the image API's six-key schema. A model that answers in text *and* pictures is a chat model that can draw and is not refused: it goes over `/chat/completions`, where `zdr` is a defined key.
  - The refusal is made **before anything is sent** -- before any acquire, any upload and any billed work -- and it uses the same machinery as every other pre-send refusal: the `MODEL_RESTRICTED_TEMPLATE` card, a `Restricted by` row naming the control that refused (`Enforce ZDR routing`, or `Request ZDR` when the user asked for it themselves), and the same per-member report on a Fusion run, so a refused panel member is a visible failed member with a card rather than a member that dies silently. The task leg gets the task adapter's own refusal shape. The reason is its own key rather than the never-read-list one, because the list was read fine.
  - `ZDR_MODELS_ONLY` matches against the suffix-stripped base id, the same rule `ZDR_ENFORCE` uses, so routing variants (`:nitro`, `:floor`, `:online`) of a ZDR-capable base are shown and allowed. A `~`-prefixed `-latest` id is answered for the model its catalog `alias_target` names and for its own key, and only here: every other capability read still answers from the alias row. It stays a catalog and request-admission filter: it never sends `provider.zdr: true`. It filters from the last ZDR list read successfully, so a later read that fails does not let non-ZDR models back into the picker; only a list that has *never* been read leaves filtering skipped. Video models are filtered like any other model -- and filtering is a statement about the **roster**, not about the transport, so a video or image-only model the roster names stays visible in the picker and is still refused at request time under `ZDR_ENFORCE`. The two answer different questions, and the picker is deliberately not narrowed to match.

- **`ALLOW_USER_ZDR_OVERRIDE`**
  - Allows users to request ZDR per chat.
  - Ignored when `ZDR_ENFORCE` is enabled.
  - If a user's stored `REQUEST_ZDR` value cannot be parsed, **or the stored row cannot be read at all**, the pipe cannot tell whether they opted in, so it enforces ZDR for that request rather than routing without it. A model that is not ZDR-capable is then refused with a `Restricted by` row naming the user's own `Request ZDR` preference rather than the `Enforce ZDR routing` valve, so an operator is not sent to a setting that is switched off. One unreadable settings row does not end that user's chat: the request is routed ZDR, or refused by that ordinary restriction card, and repairing the row restores the answer.
  - A failed read is grouped with the unparseable case because both are the same thing to the pipe — an answer it cannot evidence. A row that is read but holds a value that will not parse loses every field; a row that cannot be read loses every field **except** those Open WebUI's own instance demonstrably carries. Since `functions.py` substitutes a **default-constructed** instance when it cannot build one, and that instance carries nothing, an unreadable row on the host path loses every field — `REQUEST_ZDR` is reported unreadable and ZDR **is** enforced, for a user who may never have asked for it. One unreadable row does not end that user's chat: the request is routed ZDR, or refused by that ordinary restriction card, and repairing the row restores the answer. The user's own fields survive only on the narrow path where a **populated** instance did reach the pipe — Open WebUI having already parsed the row successfully on its way in — and there the pipe reports them as read rather than rejected.
  - The rule is fail-closed for **every** user valve, not only `REQUEST_ZDR`, and `REQUEST_ZDR` is merely the most consequential of them. A field the pipe could not read falls back to that user valve's own per-user default, never to the administrator's site-wide value (see [Valves & Configuration Atlas](valves_and_configuration_atlas.md)); `SHOW_TOOL_CARDS`, `THINKING_OUTPUT_MODE`, `REASONING_EFFORT` and `TOOL_EXECUTION_MODE` are decided the same way, and would be equally wrong to inherit the administrator's value.
  - This is a deliberate divergence from Open WebUI's own fail-open behaviour (`functions.py` substitutes a default-constructed instance with no diagnostic). That substitution is safe for Open WebUI's own valves, where it costs a preference; here the same substitution is read as a privacy decision, because the pipe routes provider retention from `REQUEST_ZDR` and decides what the model is handed from the tool and reasoning valves. The divergence is therefore scoped to the whole user-valve surface, not to `REQUEST_ZDR` alone.
  - The stored row is read **once per request**: every panel member, the judge and the synthesis are decided from that one snapshot, so a `REQUEST_ZDR` saved while a Fusion turn is already running cannot leave one stage routing with ZDR and another without it, inside that turn.

---

## User valve (per chat)

When `ALLOW_USER_ZDR_OVERRIDE` is enabled (and `ZDR_ENFORCE` is disabled), users can toggle:

- **`REQUEST_ZDR`**
  - Requests ZDR routing for that chat.

---

## When the ZDR list is stale, and what that means for enforcement

One rule governs both ways a read can fail: **a read that did not succeed carries the last read that did.** A `500`/`429` on `/models`, or an outage of `/endpoints/zdr`, leaves the previous endpoint list in force. That carry-forward is **per credential**: the pipe keeps one last-good list for each of the most recent 4 OpenRouter accounts, so a different account sees no list rather than another's, and is refused while `ZDR_ENFORCE` is on until its own read succeeds. An account older than those 4 has its list dropped rather than kept; if it comes back it is re-read rather than answered from cache, because the clock that would have suppressed the read belongs to whichever account was read most recently. A `/models` failure is the same case. During the failing credential's own backoff window — the one its second consecutive failure recorded — no read is attempted at all, and it is answered `None`: nothing read, nothing known, which is the same fail-closed state.

This **knowingly reverses** an earlier deliberate choice (commit `1f75dc1`, 2026-05-03) that wiped the list on failure and failed closed. The reason it was reversed: the wipe was invisible and much wider than it looked. `ensure_loaded` already serves the cached catalogue across a failed refresh, so nothing appeared broken — yet with `ZDR_ENFORCE` on, *every* chat was refused with a "restricted" card, including a model that had been answering for hours; a task silently returned its default; a Fusion run came back with every member failed for a reason no member mentioned. With `ZDR_MODELS_ONLY` on, filtering switched off and non-ZDR models became requestable again. Because `_refresh` returns normally on a ZDR-only outage, this recurred on **every** refresh, indefinitely, with no backoff.

The security argument does not rest on the list's freshness, and that is what makes the reversal safe:

- **Nothing unverified gets through.** Every enforced request still carries a per-request `provider.zdr=true`. That flag, not the catalogue, is what makes OpenRouter hold the request to a no-retention endpoint.
- **A negative answer still stands.** The carried list can prove a model is *not* ZDR-capable, and such a request is still refused. A stale list cannot *manufacture* a capability claim for a model it says is capable-of — it only repeats the last verified answer.
- **Never-read is still a hard stop.** If no list has ever been read, `zdr_list_available()` is `False`, `is_zdr_capable()` is `None`, and enforcement refuses every request. `ZDR_MODELS_ONLY` skips filtering in the same state. This is the cold-start path, and the change deliberately does not weaken it — an outage on the very first read must never read as "no model is zero-retention".

### The limits, stated plainly

- A model that **genuinely lost** its ZDR endpoints is noticed only once a read succeeds. The carry-forward can lag reality by at most one cache interval; the "bounded carry-over" and "staleness deadline" alternatives were both considered and rejected (the first is inert — a retired model cannot be requested; the second re-creates the very outage being removed).
- The image and video catalogue writers read the same kept list, so a media model can also carry a ZDR verdict from a list that is no longer current. This was a conscious choice for one rule over two. A spec that a refresh *preserves* (a media model no longer in `/models`) has its `zdr_capable` key re-stamped **from that carried roster** rather than carried over from the previous spec, so the key and the gate still answer alike; a refresh with no roster in force at all leaves the key absent for every spec, because a failed read does not mean "not zero-retention".
- `spec["zdr_capable"]` and `is_zdr_capable()` are reconciled from the same authority on every refresh, so on the sequential path the published row and the gate do not disagree. The spec key is a **fetch-time record** — stamped from the roster in force when the catalogue was read, and re-stamped by every credential that adopts a roster — while `is_zdr_capable()` is the **per-request authority** every gate uses, read from the credential in hand. The stamp comes from the gate's own candidate-key set, so a `~`-prefixed `-latest` alias row — the shape the default Fusion panels are made of — is stamped the way the gate answers it, not by plain membership. Enforcement is decided by `is_zdr_capable()` alone, which is why a setting that switches something off holds on every path; the row is the record of the last reconciliation, not the decision anything is enforced from.

### If you are reading a note that says the list is cleared on failure

That text is stale. It describes the behaviour before this change and should not be "fixed" again.

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
