# OpenRouter Zero Data Retention (ZDR)

OpenRouter supports **Zero Data Retention (ZDR)** routing to ensure requests only hit endpoints that do not retain prompts or responses. This pipe exposes ZDR controls as admin valves and per-chat user valves so you can filter the model list and/or enforce ZDR on requests.

> **Quick navigation:** [Docs Home](README.md) · [Valves Atlas](valves_and_configuration_atlas.md) · [Provider Routing](openrouter_provider_routing.md)

---

## How it works

The pipe asks OpenRouter for ZDR routing and leaves the judgement to it. OpenRouter also exposes ZDR-capable endpoints via the `/api/v1/endpoints/zdr` list, which the pipe reads for the model picker. The pipe:

- **Attaches `provider.zdr=true`** when ZDR is requested or enforced, on the transports whose schema defines that key. It does not check the model against the list first: a model with no ZDR endpoint is sent with the flag, and OpenRouter refuses it.
- **Shows that refusal as a short card.** For a flagged request that no endpoint can serve, OpenRouter answers HTTP 404 with `metadata.failed_routing_step` set to `Filter by Data Policy`. The pipe matches on that status and field, never on the message text, and shows the `MODEL_RESTRICTED_TEMPLATE` card with a `Restricted by` row naming the control that asked for ZDR. The same answer to a request the pipe did not flag (the account's own privacy settings) keeps the ordinary provider-error card.
- **Hides non‑ZDR models** from the picker, and refuses a hidden one if it is requested directly, when `ZDR_MODELS_ONLY` is enabled.

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
  - `ZDR_MODELS_ONLY` matches against the suffix-stripped base id, so routing variants (`:nitro`, `:floor`, `:online`) of a ZDR-capable base are shown and allowed, while a suffix OpenRouter lists as a model in its own right (`:free`, `:thinking`) is answered for **itself**, not for its base. A `~`-prefixed `-latest` id is answered for the model its catalog `alias_target` names and for its own key, and only here: every other capability read still answers from the alias row. It stays a catalog and request-admission filter: it never sends `provider.zdr: true`. It filters from the last ZDR list read successfully, so a later read that fails does not let non-ZDR models back into the picker; only a list that has *never* been read leaves filtering skipped. Video models are filtered like any other model -- and filtering is a statement about the **roster**, not about the transport, so a video or image-only model the roster names stays visible in the picker and is still refused at request time under `ZDR_ENFORCE`. The two answer different questions, and the picker is deliberately not narrowed to match.

- **`ZDR_ENFORCE`**
  - Forces `provider.zdr=true` on every request whose transport defines that key.
  - Leaves the judgement to OpenRouter: a model with no ZDR endpoint is sent with the flag and refused by OpenRouter. Enforcement never reads the ZDR list, so an outage of `/endpoints/zdr`, a failed `/models` read or a list that has never been read neither blocks a request nor lets one through unflagged. A routing variant (`:nitro`, `:floor`, `:online`, `:free`, `:thinking`) is sent with the flag like any other model.
  - The refusal is shown as the `MODEL_RESTRICTED_TEMPLATE` card with a `Restricted by` row naming the control that asked (`Enforce ZDR routing`, or `Request ZDR` when the user asked for it themselves), whether the reply is streamed or not and on `/responses`, on `/chat/completions` and on the fallback from one to the other. A machine caller with no chat to write a card into gets OpenRouter's own status and message instead. The request breaker does not count the refusal as a failure (`BREAKER_MAX_FAILURES`).
  - On a Fusion run the refusal is reported per member, naming the control that asked for ZDR, so the judge and the synthesis model are told the request was refused under the Zero Data Retention routing rather than that the model declined.
  - A background task (title, tags, follow-ups) on such a model is not retried: the refusal ends in the task failure notice, and the log carries OpenRouter's message.
  - Under enforcement a video model is not sent: OpenRouter's video schema defines one provider property, `options`, and no `zdr`, so a `zdr` sent there would be accepted and ignored -- the job would read as protected in OpenRouter's own logs while nothing enforced it. An **image-only** model (one that emits images and no text) is refused the same way, on the image API's six-key schema. A model that answers in text *and* pictures is a chat model that can draw and is not refused: it goes over `/chat/completions`, where `zdr` is a defined key.
  - That refusal is made **before anything is sent** -- before any acquire, any upload and any billed work -- and it uses the same machinery as every other pre-send refusal: the `MODEL_RESTRICTED_TEMPLATE` card, a `Restricted by` row naming the control that refused (`Enforce ZDR routing`, or `Request ZDR` when the user asked for it themselves) and the reason the request format cannot carry the control, and the same per-member report on a Fusion run, so a refused panel member is a visible failed member with a card rather than a member that dies silently. The task leg gets the task adapter's own refusal shape.
  - A request for an OpenRouter preset carries the flag merged into the provider settings the preset saved (for example `{"order": ["acme"], "zdr": true}`), because OpenRouter replaces a whole request field rather than merging into it. See [Model Variants & Presets](model_variants_and_presets.md#presets).

- **`ALLOW_USER_ZDR_OVERRIDE`**
  - Allows users to request ZDR per chat.
  - Ignored when `ZDR_ENFORCE` is enabled.
  - A reply a user sends in a chat another user owns also honours the chat owner's saved `REQUEST_ZDR`: the request is routed with ZDR when either the sender or the chat's owner asked for it. The owner's answer is read only on a reply by someone other than the chat's owner, and only in a saved chat - a temporary chat has no stored row to consult, so the sender's answer alone decides there. Telling whose chat it is costs one light ownership check on every such reply; the chat's owner (one column of the chat's row, never the conversation, unless that read fails, when the whole chat row is read instead) and the owner's saved answer are read only once someone else owns it. An owner's saved answer that cannot be read - the owner's stored value unparseable, or the owner's user row gone - is enforced, not dropped: the same fail-closed rule the sender's own unreadable answer follows, and the refusal names the chat owner's saved answer. A chat whose owner cannot be determined at all, because the chat's row cannot be read or records no owner, is enforced the same way, and its refusal says the chat's owner could not be determined.
  - If a user's stored `REQUEST_ZDR` value cannot be parsed, **or the stored row cannot be read at all**, the pipe cannot tell whether they opted in, so it enforces ZDR for that request rather than routing without it. The request carries the flag, and when OpenRouter refuses it the card's `Restricted by` row names the user's own `Request ZDR` preference rather than the `Enforce ZDR routing` valve, so an operator is not sent to a setting that is switched off. One unreadable settings row does not end that user's chat: the request is routed ZDR, or refused by OpenRouter and shown as that restriction card, and repairing the row restores the answer.
  - A failed read is grouped with the unparseable case because both are the same thing to the pipe — an answer it cannot evidence. A row that is read but holds a value that will not parse loses every field; a row that cannot be read loses every field **except** those Open WebUI's own instance demonstrably carries. Since `functions.py` substitutes a **default-constructed** instance when it cannot build one, and that instance carries nothing, an unreadable row on the host path loses every field — `REQUEST_ZDR` is reported unreadable and ZDR **is** enforced, for a user who may never have asked for it. One unreadable row does not end that user's chat: the request is routed ZDR, or refused by OpenRouter and shown as that restriction card, and repairing the row restores the answer. The user's own fields survive only on the narrow path where a **populated** instance did reach the pipe — Open WebUI having already parsed the row successfully on its way in — and there the pipe reports them as read rather than rejected.
  - The rule is fail-closed for **every** user valve, not only `REQUEST_ZDR`, and `REQUEST_ZDR` is merely the most consequential of them. A field the pipe could not read falls back to that user valve's own per-user default, never to the administrator's site-wide value (see [Valves & Configuration Atlas](valves_and_configuration_atlas.md)); `SHOW_TOOL_CARDS`, `THINKING_OUTPUT_MODE`, `REASONING_EFFORT` and `TOOL_EXECUTION_MODE` are decided the same way, and would be equally wrong to inherit the administrator's value.
  - This is a deliberate divergence from Open WebUI's own fail-open behaviour (`functions.py` substitutes a default-constructed instance with no diagnostic). That substitution is safe for Open WebUI's own valves, where it costs a preference; here the same substitution is read as a privacy decision, because the pipe routes provider retention from `REQUEST_ZDR` and decides what the model is handed from the tool and reasoning valves. The divergence is therefore scoped to the whole user-valve surface, not to `REQUEST_ZDR` alone.
  - The stored row is read **once per request**: every panel member, the judge and the synthesis are decided from that one snapshot, so a `REQUEST_ZDR` saved while a Fusion turn is already running cannot leave one stage routing with ZDR and another without it, inside that turn.

---

## User valve (per chat)

When `ALLOW_USER_ZDR_OVERRIDE` is enabled (and `ZDR_ENFORCE` is disabled), users can toggle:

- **`REQUEST_ZDR`**
  - Requests ZDR routing for that chat.
  - A reply sent in a chat another user owns also honours that chat owner's saved answer: ZDR applies when either the sender or the chat's owner asked for it.

---

## When the ZDR list is stale

The ZDR list serves the model picker and request admission under `ZDR_MODELS_ONLY`, and the retention line of the video help card. Enforcement does not read it, so none of what follows changes what an enforced request carries.

One rule governs both ways a read can fail: **a read that did not succeed carries the last read that did.** A `500`/`429` on `/models`, or an outage of `/endpoints/zdr`, leaves the previous endpoint list in force. That carry-forward is **per credential**: the pipe keeps one last-good list per OpenRouter account, kept for as long as that account keeps being used, so a different account sees no list rather than another's, and its picker is not filtered until its own read succeeds. An account nobody has asked about for four hours has its list dropped rather than kept; if it comes back it is re-read rather than answered from cache, because the clock that would have suppressed the read belongs to whichever account was read most recently. A least-recently-used ceiling of 64 sits above any live set, so a burst of rotated one-shot keys inside that window cannot grow the map without limit. A `/models` failure is the same case. During the failing credential's own backoff window — the one its second consecutive failure recorded — no read is attempted at all, and it is answered `None`: nothing read, nothing known, so `ZDR_MODELS_ONLY` skips filtering for it.

The carry-forward keeps the picker steady across an outage: without it, an outage of `/endpoints/zdr` would let every non‑ZDR model back into the picker for as long as the outage lasted.

What the list's freshness does and does not decide:

- **Nothing unverified gets through enforcement.** Every enforced request carries a per-request `provider.zdr=true`. That flag, not the catalogue, is what makes OpenRouter hold the request to a no-retention endpoint.
- **A negative answer still stands in the picker.** The carried list can prove a model is *not* ZDR-capable, and such a model stays hidden. A stale list cannot *manufacture* a capability claim for a model it says is capable-of — it only repeats the last verified answer.
- **Never-read skips filtering.** If no list has ever been read, `zdr_list_available()` is `False`, `is_zdr_capable()` is `None`, and `ZDR_MODELS_ONLY` skips filtering. An outage on the very first read must never read as "no model is zero-retention".


### The limits, stated plainly

- A model that **genuinely lost** its ZDR endpoints is noticed only once a read succeeds. The carry-forward can lag reality by at most one cache interval; the "bounded carry-over" and "staleness deadline" alternatives were both considered and rejected (the first is inert — a retired model cannot be requested; the second re-creates the very outage being removed).
- The image and video catalogue writers read the same kept list, so a media model can also carry a ZDR verdict from a list that is no longer current. This was a conscious choice for one rule over two. A spec that a refresh *preserves* (a media model no longer in `/models`) has its `zdr_capable` key re-stamped **from that carried roster** rather than carried over from the previous spec, so the key and the gate still answer alike; a refresh with no roster in force at all leaves the key absent for every spec, because a failed read does not mean "not zero-retention".
- `spec["zdr_capable"]` and `is_zdr_capable()` are reconciled from the same authority on every refresh. The spec key is a **fetch-time record** — stamped from the roster in force when the catalogue was read, and re-stamped by every credential that adopts a roster — while `is_zdr_capable()` is the **per-request authority** every gate uses, read from the credential in hand. A warm request re-stamps the table only when the roster object or the spec table has moved since the last stamp; a request whose roster is already the one in force on the specs it has re-stamped publishes the roster, changes nothing and keeps the enriched cache, which is why a credential with no list read for it no longer costs a sweep of the whole catalogue on every request. Where specs are re-stamped in place without a rebind — a rotation whose refresh failed — the marker is busted, so the next adopt sweeps rather than trusting a record some other writer left. The stamp comes from the gate's own candidate-key set, so a `~`-prefixed `-latest` alias row — the shape the default Fusion panels are made of — is stamped the way the gate answers it, not by plain membership.
- The row `list_models()` publishes is **derived at publish time from the credential in hand**, and names it. `zdr_capable` on a published row is that credential's own verdict for that model, computed from the roster its `_ZDR_KEY` context names — not a copy of the spec key, which is one shared record that whichever credential last stamped it wrote. So on a worker serving two credentials, each publish answers about its own account even when the two passes interleave, and the roster in hand is a term of the enriched cache, so one credential's rows are never handed to another. With no roster in force for the credential publishing, the key is **absent** from every row rather than `False`, which is the same state the spec key records and the same state the gate answers `None` in. The picker filter and request admission under `ZDR_MODELS_ONLY` are decided by `is_zdr_capable()` alone, which is why a setting that switches something off holds on every path; the row is what the pipe shows, not the decision anything is filtered from.

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
