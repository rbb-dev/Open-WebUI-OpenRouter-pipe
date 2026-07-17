"""Curated display metadata for the OpenRouter pipe's config valves.

Overlay ONLY. The Config tab reads live valve structure (type, default, bounds,
enum, secret-ness) from ``Valves.model_fields``; this module supplies just the
human title, the tree group, and the rich help text. A valve absent from this
map still renders and edits normally (humanised name, "Uncategorized" group, its
own ``Field(description=...)`` as help) with a "not documented yet" notice.

Structural facts are NEVER duplicated here (they can drift); only title/group/help.
"""

from __future__ import annotations

CONFIG_META: dict[str, dict[str, str]] = {
    "ADMIN_PROVIDER_ROUTING_MODELS": {
        "title": "Admin-enforced provider routing",
        "group": "Connection & Routing/Provider Routing",
        "detail": "Lists the models whose upstream provider an admin fixes for everyone — OpenRouter then sends each to the chosen providers rather than its default price-weighted pick.\n\nGive a comma-separated list of exact catalog ids such as `openai/gpt-4o, anthropic/claude-3.5-sonnet`; left empty it enforces nothing. Each listed model gets its own Provider entry in Open WebUI's Functions list, where an admin sets the routing — a fixed provider order, allowed or avoided providers, a sort by price, throughput, or latency, plus quantization, performance-floor, and price-cap limits — that every request to that model obeys, with no per-chat way to switch it off.\n\nListing a model under `User-adjustable provider routing` too doesn't add a second rule; it becomes one user-toggleable switch whose admin values act only as overridable defaults. A provider order fixed here also overrides the soft provider affinity from `Pin conversation to one provider`.\n\n**Warning:** Each slug must exactly match an OpenRouter catalog id, or that model is silently skipped and no routing is applied."
    },
    "ALLOW_INSECURE_HTTP": {
        "title": "Allow plaintext HTTP downloads",
        "group": "Security/Network (SSRF)",
        "detail": "Lets the pipe fetch remote images, files, and videos over plaintext `http://`, which is blocked by default.\n\nOff by default, and turning it on alone permits nothing — each host must also appear in `Plaintext HTTP host allowlist`, and an empty allowlist keeps every `http://` URL blocked. HTTP is refused by default because the transport is unencrypted: anyone on the network path can read the fetched file or silently substitute a malicious one. `https://` downloads are unaffected either way. Enable it only for an internal service that has no HTTPS endpoint, and pair it with a tight allowlist."
    },
    "ALLOW_INSECURE_HTTP_HOSTS": {
        "title": "Plaintext HTTP host allowlist",
        "group": "Security/Network (SSRF)",
        "detail": "Names the hosts allowed to serve plaintext `http://` images, files, and videos, as an exact-match, case-insensitive allowlist.\n\nIt narrows `Allow plaintext HTTP downloads`: that toggle must be on for any of these to work, and while it is off every `http://` URL is blocked no matter what this list holds. Give comma-separated hosts or `host:port` entries with no wildcards, e.g. `example.com, example.org:8080, 203.0.113.10` (bracket IPv6, as in `[::1]:8080`). A bare host allows any port, a `host:port` entry allows only that port, and a URL with no port is treated as port `80`. Empty — the default — blocks all plaintext HTTP, so matching `http://` media is dropped with a security-policy error.\n\n**Warning:** Matching is exact — `example.com` does not cover `www.example.com`, so list every host. Listed hosts are still reached over unencrypted HTTP, so keep the list to trusted internal addresses."
    },
    "ALLOW_UNKNOWN_SIZE_CLOUD_READS": {
        "title": "Read cloud files of unknown size",
        "group": "Security/Network (SSRF)",
        "detail": "When enabled, the pipe reads an Open WebUI file held on cloud storage even when its recorded size is missing or invalid.\n\nThe default `false` blocks only this narrow case — a non-local backend such as S3, GCS, or Azure combined with an unusable size value — to avoid an unbounded download; local storage and files that report a valid size are never affected. When enabled, the file is still copied to a private temporary file and rejected if it exceeds `Maximum base64 upload size` after download, so the size ceiling still holds.\n\n**Tip:** Leave disabled unless cloud-hosted files fail to attach with an unknown-size error."
    },
    "ALLOW_USER_ZDR_OVERRIDE": {
        "title": "Allow user ZDR opt-in",
        "group": "Models & Catalog/ZDR",
        "detail": "Lets each user turn on Zero Data Retention for their own chats through the per-chat `Request ZDR` toggle, without making it mandatory for everyone.\n\nWhen a user opts in, their requests route only to ZDR endpoints and a model with none is rejected for them — the same guard as `Enforce ZDR routing`, scoped to one person. Switch this off and the per-chat toggle is ignored, so ZDR then comes only from `Enforce ZDR routing`, which itself overrides this whenever it is on."
    },
    "ANTHROPIC_PROMPT_CACHE_TTL": {
        "title": "Cached prompt lifetime",
        "group": "Prompt Caching/General",
        "detail": "How long Claude keeps a cached prompt prefix reusable before it expires and has to be written again.\n\nApplies only while `Enable Claude prompt caching` is on and the model is Anthropic Claude. A longer lifetime keeps the prefix warm across slower conversations so fewer turns pay to rewrite it, but each longer-lived write is billed at a higher rate — check current cache-write and cache-read pricing on OpenRouter.\n\n- `5m` — expires quickly; keeps write cost lowest and suits chats whose turns arrive close together.\n- `1h` — survives long gaps between turns; costs more per write but can be cheaper over an extended session by avoiding repeated rewrites."
    },
    "API_KEY": {
        "title": "OpenRouter API key",
        "group": "Connection & Routing/Auth",
        "detail": "The OpenRouter API key used as the bearer token that authenticates every request this pipe sends to OpenRouter.\n\nSensitive: stored encrypted at rest and write-only in the panel. When unset, it falls back to the `OPENROUTER_API_KEY` environment variable; if both are empty, requests fail with a \"not configured\" error.\n\n**Warning:** If `WEBUI_SECRET_KEY` changes after the key is saved, the stored value cannot be decrypted and requests fail until it is re-entered."
    },
    "ARTIFACT_CLEANUP_DAYS": {
        "title": "Artifact retention period",
        "group": "Storage/Cleanup",
        "detail": "How long an artifact can go untouched before the background cleanup sweep deletes it.\n\nThe clock runs from each artifact's last database read, not its original creation: reloading a chat that pulls its artifacts from the database resets their age, so records in actively-used conversations survive and only those left unread for the whole window are purged. Shorten it to reclaim database space sooner, lengthen it to keep more history. There is no off switch — retention always applies, so choose a window rather than trying to disable it. `Cleanup sweep frequency` sets how often the sweep runs."
    },
    "ARTIFACT_CLEANUP_INTERVAL_HOURS": {
        "title": "Cleanup sweep frequency",
        "group": "Storage/Cleanup",
        "detail": "How often the background worker wakes to delete artifacts that have aged out.\n\nThis changes only the schedule, not what gets removed — the age cutoff is `Artifact retention period`. Each wake adds a small random jitter (up to a quarter of the interval, capped at ten minutes) so multiple workers don't all sweep at the same instant. Run it more often for timelier deletion at the cost of extra database load, less often to lighten that load while expired rows linger a little longer.\n\n**Tip:** Rarely needs changing; leave it as is unless the sweep noticeably loads the database."
    },
    "ARTIFACT_ENCRYPTION_KEY": {
        "title": "Storage encryption key",
        "group": "Security/Encryption",
        "detail": "The secret that switches on at-rest encryption for everything the pipe persists to its database — reasoning traces, tool results, and other artifacts.\n\nLeft unset (the default) nothing is encrypted: every artifact is written as plaintext JSON, including reasoning tokens. Setting a value turns encryption on, and `Encrypt all stored data` then decides the scope — every artifact, or reasoning tokens only. Use a long, random secret. The value itself is sensitive: it is stored encrypted at rest under `WEBUI_SECRET_KEY` and shown write-only here.\n\n**Warning:** The artifact table is named from a hash of this key, so changing it sends new writes to a fresh table and permanently strands everything saved under the old key. Decide it before first launch and keep it stable."
    },
    "AUTHENTICATION_ERROR_TEMPLATE": {
        "title": "Authentication failure message",
        "group": "Error Messages/Templates",
        "detail": "The message users see when OpenRouter rejects a request as unauthorized (`401`) — or when the pipe can't resolve an API key.\n\nThe template supports `{placeholder}` substitution and `{{#if}}` conditional blocks; representative fields include `{error_id}`, `{openrouter_message}`, and `{support_email}`. A line is dropped automatically when its referenced placeholder has no value, so optional rows never render as blanks. Editing it customizes the wording and branding of authentication failures; set `Support email` or `Support link` to fill the matching support placeholders.\n\n**Warning:** Clearing this field falls back to the generic error message, not the authentication-specific default."
    },
    "AUTO_ATTACH_DIRECT_UPLOADS_FILTER": {
        "title": "Attach Direct Uploads to models",
        "group": "Filters & Integrations/Direct Uploads",
        "detail": "Gives users a per-chat Direct Uploads switch on every model that accepts a direct file, audio, or video input, automatically.\n\nWithout it, that switch appears only on models an admin equips one at a time; models accepting none of those input types are skipped either way. On by default, and it needs the capability present first (see `Install Direct Uploads filter`). A single model can still opt out with the `disable_direct_uploads_auto_attach` advanced parameter, even while this is on. The switch alone diverts nothing until a user also turns on file, audio, or video in its settings — each starts off."
    },
    "AUTO_ATTACH_FUSION_FILTER": {
        "title": "Attach Fusion to its model",
        "group": "Filters & Integrations/Fusion",
        "detail": "Puts Fusion's per-chat controls on the `openrouter/fusion` model automatically, so its panel, judge, and `Always run Fusion` options appear in that model's chat settings without an admin adding them one at a time.\n\nThey reach only that one model and never any other, because who may use Fusion is already governed by who can open the `openrouter/fusion` model in Open WebUI. Without this, an admin equips that model by hand. Needs the controls present first (see `Install Fusion filter`) and the master `Enable OpenRouter Fusion` on. To offer Fusion on a different model, an admin adds it there by hand and switches on the filter's `ALLOW_ON_NON_FUSION_MODELS` valve."
    },
    "AUTO_ATTACH_IMAGE_FILTERS": {
        "title": "Attach image controls to models",
        "group": "Filters & Integrations/Image",
        "detail": "Puts the right image controls on every native image model, automatically.\n\nEach model gets the generic controls plus its family's controls where one exists. On by default; a single model can opt out with the `disable_image_filter_auto_attach` advanced parameter. Needs the filters present first (see `Install native image filters`)."
    },
    "AUTO_ATTACH_IMAGE_GEN_FILTER": {
        "title": "Attach Image Generation toggle",
        "group": "Filters & Integrations/Image",
        "detail": "Gives users a per-chat image-generation switch on every model, automatically.\n\nThe switch lets a user turn on inline image generation for a conversation; without this, an admin adds it to models one at a time. On by default. Needs the capability present first (see `Install Image Generation filter`) and `Enable image generation` on."
    },
    "AUTO_ATTACH_VIDEO_FILTERS": {
        "title": "Attach Video Generation to models",
        "group": "Filters & Integrations/Video",
        "detail": "Puts each video model's controls on that model automatically, so its duration/resolution/audio options appear in chat.\n\nWithout it, an admin adds the filter to each video model by hand. On by default; a single model can opt out with the `disable_video_gen_auto_attach` advanced parameter. Needs the filters present first (see `Install Video Generation filters`)."
    },
    "AUTO_ATTACH_WEB_TOOLS_FILTER": {
        "title": "Attach web tools to models",
        "group": "Filters & Integrations/Web Tools",
        "detail": "Gives users a per-chat web tools switch on every model that can call tools, automatically.\n\nWithout it, that switch appears only on models an admin equips one at a time. Image-output and video-generation models are skipped, since they can't call tools. On by default; needs the capability present first (see `Install web tools filter`). A single model can still opt out with the `disable_web_tools_auto_attach` advanced parameter, even while this is on."
    },
    "AUTO_CONTEXT_TRIMMING": {
        "title": "Auto-trim overlong prompts",
        "group": "Reasoning & Thinking/Context",
        "detail": "Keeps a long conversation working when it grows past the model's context window, instead of the request failing with a context-length error.\n\nWith this on, OpenRouter's `context-compression` plugin removes or shortens messages from the middle of the prompt until it fits, so the chat continues — the trade-off is recall, since the model no longer sees that dropped middle, so prefer it where perfect recall of every earlier turn isn't essential. It works on OpenRouter's `/responses` endpoint (the pipe's default); a model routed to `/chat/completions` is unaffected either way. On by default. With it off, an over-length `/responses` request fails outright rather than being trimmed — suitable only where the deployment trims or summarizes context itself."
    },
    "AUTO_DEFAULT_FUSION_FILTER": {
        "title": "Pre-enable Fusion per chat",
        "group": "Filters & Integrations/Fusion",
        "detail": "Starts each new `openrouter/fusion` chat with Fusion's per-chat controls already switched on, so its panel and judge options are in effect from the first message without the user enabling anything.\n\nThis only readies the controls — it does not make Fusion run on every message. That is the separate per-user `Always run Fusion` switch, which stays off, so by default the model still decides per prompt whether to deliberate. A cleared default comes back on its own — the pipe keeps reapplying it rather than setting it once. Works only while `Enable OpenRouter Fusion` is on and Fusion's controls are on the model (see `Attach Fusion to its model`)."
    },
    "AUTO_DEFAULT_IMAGE_FILTERS": {
        "title": "Pre-enable image controls per chat",
        "group": "Filters & Integrations/Image",
        "detail": "Starts each new chat with a native image model's generation controls already switched on, so its size, aspect-ratio, and provider options apply from the first message.\n\nThese models return images whether or not the controls are on — switching them off just falls back to the model's own defaults and does not stop image output. So this is a convenience: the size, aspect-ratio, and provider knobs are ready without the user enabling anything, and they can still turn them off per chat. On by default, and a cleared default comes back on its own. Effective only on models that carry those controls (see `Attach image controls to models`)."
    },
    "AUTO_DEFAULT_VIDEO_FILTERS": {
        "title": "Pre-enable Video Generation per chat",
        "group": "Filters & Integrations/Video",
        "detail": "Starts each new chat with a video model's controls already switched on, so its generation settings — duration, resolution, audio, and model-specific options — apply from the first message.\n\nThose controls are where a model's per-model settings live, including ones a model requires: Veo, for one, won't generate until its person-generation policy is chosen. Pre-enabling the controls surfaces them from the first message so the user can set such a value without first switching the filter on — though a genuinely required one still has to be picked. A model with no required setting renders from a bare prompt either way. On by default, and a cleared default comes back on its own. Effective only on models that carry those controls (see `Attach Video Generation to models`)."
    },
    "AUTO_DEFAULT_WEB_TOOLS_FILTER": {
        "title": "Enable web tools by default",
        "group": "Filters & Integrations/Web Tools",
        "detail": "Starts each new chat with the web tools switch already on, so users get web search without flipping it themselves.\n\nUsers can always turn it off in a chat. It applies to tool-capable models as they gain the switch (see `Attach web tools to models`). A single model can be excluded with the `disable_web_tools_default_on` advanced parameter, so the switch appears there but starts off."
    },
    "AUTO_FALLBACK_CHAT_COMPLETIONS": {
        "title": "Automatic endpoint fallback",
        "group": "Connection & Routing/Endpoints",
        "detail": "When enabled, a request the `/responses` endpoint rejects as unsupported is automatically retried once against `/chat/completions`.\n\nThe retry fires only when the failure looks like a model or endpoint that cannot serve `/responses`, and only before any visible output has streamed. The default `true` keeps chats working when a model routed to `/responses` — whether by `Default API endpoint` or a force pattern — cannot actually serve it. Disable it to make such failures surface as errors instead of silently switching endpoints."
    },
    "AUTO_INSTALL_DIRECT_UPLOADS_FILTER": {
        "title": "Install Direct Uploads filter",
        "group": "Filters & Integrations/Direct Uploads",
        "detail": "Keeps the Direct Uploads integration available in this workspace and up to date — the capability that lets a user send an uploaded file, audio, or video straight to the model instead of through Open WebUI's usual text extraction and retrieval.\n\nThis is the master presence switch: `Attach Direct Uploads to models` stays inert until it is on. An admin manages the integration in Open WebUI's Functions list, setting its per-type size limits and the file, audio, and video allowlists there. On by default. With it off, an admin can still add the integration by hand and that copy is left untouched; but with none present at all, the per-chat Direct Uploads switch cannot appear for anyone."
    },
    "AUTO_INSTALL_FUSION_FILTER": {
        "title": "Install Fusion filter",
        "group": "Filters & Integrations/Fusion",
        "detail": "Keeps the OpenRouter Fusion controls present in this workspace and up to date, so the multi-model panel, judge, and per-chat options stay current without manual upkeep.\n\nThis is the presence switch for those controls: `Attach Fusion to its model` and `Pre-enable Fusion per chat` do nothing until it is on. It also needs the master `Enable OpenRouter Fusion`; with that off, the existing controls are switched off rather than refreshed. Turn this off and an admin can still add or edit the Fusion entry by hand in Open WebUI's Functions list — a copy added that way stays in place and is no longer updated."
    },
    "AUTO_INSTALL_IMAGE_FILTERS": {
        "title": "Install native image filters",
        "group": "Filters & Integrations/Image",
        "detail": "Keeps the controls for OpenRouter's native image models available in this workspace and up to date.\n\nEvery native image model gets a generic set of size and aspect-ratio controls, and families such as Gemini and Sourceful add extra controls for their own settings. On by default; `Attach image controls to models` and `Pre-enable image controls per chat` do nothing until this is on, and it needs `Show native image models` on."
    },
    "AUTO_INSTALL_IMAGE_GEN_FILTER": {
        "title": "Install Image Generation filter",
        "group": "Filters & Integrations/Image",
        "detail": "Keeps the inline image-generation capability available in this workspace and up to date.\n\nThis is the master presence switch for it: `Attach Image Generation toggle` does nothing until it is on, and it also needs `Enable image generation`. On by default. Admins set the drawing model and moderation on the Image Generation entry in Open WebUI's Functions list."
    },
    "AUTO_INSTALL_VIDEO_FILTERS": {
        "title": "Install Video Generation filters",
        "group": "Filters & Integrations/Video",
        "detail": "Keeps the per-model video controls available in this workspace and up to date.\n\nEach video model gets its own filter exposing that model's options — duration, resolution, audio, and so on; this installs and refreshes them. On by default; `Attach Video Generation to models` and `Pre-enable Video Generation per chat` do nothing until it is on, and it needs `Enable video generation` on."
    },
    "AUTO_INSTALL_WEB_TOOLS_FILTER": {
        "title": "Install web tools filter",
        "group": "Filters & Integrations/Web Tools",
        "detail": "Keeps the web tools — search, fetch, datetime, advisor, subagent, and model search — available in this workspace and up to date.\n\nThis is the master presence switch: `Attach web tools to models` and `Enable web tools by default` do nothing until it is on. On by default. Turning off every individual web tool removes the capability on its own. Admins configure the tools' engines, limits, models, and the `SERVER_TOOLS_MAX_COST_USD` cap on the web tools entry in Open WebUI's Functions list."
    },
    "BASE64_MAX_SIZE_MB": {
        "title": "Maximum base64 upload size",
        "group": "Files & Media/Uploads & Limits",
        "detail": "How large a base64-encoded file or image may get before the pipe decodes it and forwards it to OpenRouter.\n\nThere is no unlimited setting; the default covers typical document and image uploads while blocking payloads large enough to spike memory or bloat the outgoing HTTP request. Oversized inline data URLs are dropped and Open WebUI-hosted files are rejected with a preparation error, so raise it only when users must attach larger files.\n\n**Tip:** This limit is separate from `Maximum remote file size`, which caps remote-URL downloads; large attachments may require raising both."
    },
    "BASE_URL": {
        "title": "OpenRouter API base URL",
        "group": "Connection & Routing/Endpoints",
        "detail": "The root URL the pipe prepends to every OpenRouter API path, such as `/responses`, `/chat/completions`, and `/models`.\n\nBy default it targets OpenRouter directly; when left unset in the panel it is seeded from the `OPENROUTER_API_BASE_URL` environment variable. Override it only to route through a gateway or proxy that mirrors the OpenRouter API, for example `https://gateway.internal/or/api/v1`. Trailing slashes are trimmed automatically, and the value applies to every request from this process, for all users.\n\n**Warning:** A custom endpoint must expose the full OpenRouter surface (chat, `/models`, and the media paths); a proxy missing any of these breaks model listing or generation."
    },
    "BREAKER_HISTORY_SIZE": {
        "title": "Retained failure history",
        "group": "Reliability/Circuit Breaker",
        "detail": "How many recent failure timestamps the per-user database breaker keeps — the memory that caps how high its trip count can climb.\n\nThis sizes only the storage breaker; the request and per-tool breakers track their own history against the trip count automatically. Because a breaker can never hold more failures than it remembers, keep this at or above `Failures before tripping` — set it lower and the storage breaker can never gather enough failures to open, silently leaving artifact writes unprotected during a database outage. Out of the box the two already line up, so it usually only needs raising when `Failures before tripping` is raised."
    },
    "BREAKER_MAX_FAILURES": {
        "title": "Failures before tripping",
        "group": "Reliability/Circuit Breaker",
        "detail": "How many failures one user may rack up inside the breaker's rolling window before the pipe starts refusing that user's work.\n\nA single count drives three self-healing breakers, each scoped to one user so a struggling user never blocks anyone else: repeated request failures short-circuit that user's next requests with a retry notice; repeated failures of one tool type skip just that tool; repeated database-write failures skip saving artifacts, so chats still answer but stop persisting traces. Failures are tallied over the span set by `Failure counting window`. Raise it to absorb transient errors before anyone is blocked; lower it to fail fast and protect the backend.\n\n**Tip:** Raising this well up? Raise `Retained failure history` to match, or the database breaker's shorter memory never lets it reach the higher trip count."
    },
    "BREAKER_WINDOW_SECONDS": {
        "title": "Failure counting window",
        "group": "Reliability/Circuit Breaker",
        "detail": "How far back the breakers look when tallying a user's failures — the rolling window that decides whether recent trouble adds up to a trip.\n\nEvery breaker counts only failures inside this trailing span, and older ones drop off on their own — which is exactly how a blocked user recovers with no operator action. Widen it and failures spread further apart still add up, so a breaker trips more readily and a tripped user stays blocked longer; narrow it and only tightly clustered failures trip, clearing within moments — faster recovery, but weaker cover against a slow drip of errors. `Failures before tripping` sets how many must land inside this window to open a breaker."
    },
    "CONNECTION_ERROR_TEMPLATE": {
        "title": "Connection failure message",
        "group": "Error Messages/Templates",
        "detail": "The message users see when the pipe cannot open a network connection to OpenRouter's servers.\n\nIt appears only when the connection attempt itself fails — a DNS failure, a blocked or refused connection — not when a request times out, which uses `Network timeout message`. The string supports `{placeholder}` substitution, with representative fields `{error_id}`, `{error_type}`, and `{timestamp}`, plus `{{#if field}}...{{/if}}` conditionals that render a wrapped block only when that field has a value. The default prints a `Connection Failed` notice with likely causes and fixes; `{support_email}` is filled from the `Support email` valve and shown only when that valve is set.\n\n**Warning:** Clearing this field does not restore the built-in connection message — an empty value renders a generic OpenRouter error notice instead."
    },
    "COSTS_REDIS_DUMP": {
        "title": "Publish cost snapshots to Redis",
        "group": "Reliability/Redis",
        "detail": "Writes a per-request usage snapshot into Redis after each answer, for an outside job to collect into cost, chargeback, or billing analytics.\n\nOff by default. Each snapshot carries the model, the token-usage figures, and the user's identity — their Open WebUI id, email, and name — under a `costs:` key the pipe writes but never reads back, so real personal data lands in the shared Redis for whatever external collector consumes it. It only writes while the Redis write-behind cache is active (same prerequisites as `Enable Redis write-behind cache`), and each key expires after `Cost snapshot retention`.\n\n**Tip:** Leave off unless such a collector actually exists — and treat the `costs:` keys as sensitive when it is on."
    },
    "COSTS_REDIS_TTL_SECONDS": {
        "title": "Cost snapshot retention",
        "group": "Reliability/Redis",
        "detail": "How long each cost snapshot survives in Redis before it auto-expires and is gone.\n\nIt applies only when `Publish cost snapshots to Redis` is on and the Redis cache is active. The window has to outlast the collector's polling interval: a snapshot that expires before the external job reads it is lost, and that request's cost data with it. Shorten it to purge cost data quickly and spare Redis memory; lengthen it for a collector that runs only occasionally."
    },
    "DB_BATCH_SIZE": {
        "title": "Database commit batch size",
        "group": "Streaming & Performance/Concurrency",
        "detail": "How many artifact rows the pipe writes to its database in a single transaction.\n\nThe direct write path commits pending artifacts in chunks this size; when artifacts are buffered through Redis, the same limit caps how many are drained from the queue and written per flush. Larger batches mean fewer, bigger transactions that hold database locks longer, while smaller ones commit more often with lighter locking.\n\n**Tip:** Rarely needs changing; revisit only if database write contention or oversized transactions appear."
    },
    "DEFAULT_LLM_ENDPOINT": {
        "title": "Default API endpoint",
        "group": "Connection & Routing/Endpoints",
        "detail": "Which OpenRouter API path the pipe uses by default — the newer `/responses` or the classic `/chat/completions` — for any model no force pattern pins elsewhere.\n\n`Models forced to responses` and `Models forced to chat completions` override it per model; it decides the endpoint only for models that match no force pattern.\n\n- `responses`: the widest feature coverage; for Claude, `Enable Claude prompt caching` uses a single automatic top-level breakpoint that caches only when OpenRouter routes to Anthropic directly.\n- `chat_completions`: uses explicit per-block cache breakpoints instead — the form that also caches Bedrock- and Vertex-routed Claude — plus some other provider features.\n\nMost deployments want the default. Switch only for a provider needing `/chat/completions` or for Bedrock/Vertex Claude caching; even then, `Automatic endpoint fallback` retries a `/responses` rejection on `/chat/completions` automatically."
    },
    "DIRECT_UPLOAD_FAILURE_TEMPLATE": {
        "title": "Upload failure message",
        "group": "Error Messages/Templates",
        "detail": "Markdown shown to the end user when uploaded files, audio, or video cannot be attached to an outgoing request through OpenRouter Direct Uploads.\n\nIt fires on a pre-flight failure, such as an incompatible mix of attachments, and fully replaces the model reply, so keep the wording actionable. The text supports `{placeholder}` substitution plus `{{#if}}` conditional blocks; representative fields are `{error_id}`, `{requested_model}`, and `{reason}` (the underlying cause). The default lists retry steps like splitting attachments across separate messages, and clearing the field falls back to the generic OpenRouter error notice.\n\n**Tip:** The `{support_email}` line renders only when the `Support email` valve is set, so add a contact there before relying on it."
    },
    "ENABLE_ADVISOR": {
        "title": "Enable advisor tool",
        "group": "Tools/Built-in Server Tools",
        "detail": "Gives models OpenRouter's advisor server tool — one they can call to consult a stronger model partway through an answer.\n\nAt a decision point — before committing, when stuck, or before calling a task done — the model calls the advisor; OpenRouter runs that model server-side and returns its guidance to fold in. An admin sets which model advises in the Web Tools filter (`ADVISOR_MODEL`; empty uses the chat's own model), and each user gets a per-chat advisor switch, off by default. Because it spends on an extra model call, bound a request's tool spend with the filter's `SERVER_TOOLS_MAX_COST_USD`; check current rates on OpenRouter. On by default."
    },
    "ENABLE_ANTHROPIC_INTERLEAVED_THINKING": {
        "title": "Claude interleaved thinking",
        "group": "Reasoning & Thinking/General",
        "detail": "Lets Anthropic Claude models keep thinking between their tool calls within one answer, instead of doing all their reasoning up front.\n\nOn interleaved-capable Claude models this makes multi-step work more adaptive — the model can reflect on a tool's result before choosing its next step. It applies only to Anthropic Claude models (via Anthropic's `interleaved-thinking-2025-05-14` beta); every other provider ignores it. On by default site-wide, but each user can override it in their own settings, and their choice wins. The extra thinking bills as output tokens — check OpenRouter's pricing."
    },
    "ENABLE_ANTHROPIC_PROMPT_CACHING": {
        "title": "Enable Claude prompt caching",
        "group": "Prompt Caching/General",
        "detail": "Lets Claude reuse a large, stable prompt prefix — system prompt, tool definitions, RAG context — instead of paying to reprocess it on every turn.\n\nApplies only to Anthropic Claude models. The endpoint decides how it caches: the default `/responses` path (see `Default API endpoint`) uses one automatic breakpoint that works only when OpenRouter routes to Anthropic directly, so Bedrock- and Vertex-hosted Claude are excluded; `/chat/completions` instead places explicit per-block breakpoints that also cache on Bedrock and Vertex.\n\nNothing is cached until the repeated prefix passes the model's minimum cacheable size. Reads are discounted but the first write can cost more than a normal prompt, so savings come from reuse — check current cache pricing on OpenRouter. `Cached prompt lifetime` sets how long an entry survives, and `Pin conversation to one provider` keeps hits high."
    },
    "ENABLE_DATETIME": {
        "title": "Enable date and time tool",
        "group": "Tools/Built-in Server Tools",
        "detail": "Gives models OpenRouter's datetime server tool — one they can call to look up the current date and time.\n\nThe model calls it when an answer depends on the current moment — today's date, a relative day such as next Tuesday, or elapsed time — so it isn't guessing from its training cutoff. OpenRouter runs it server-side; each user can set a timezone per chat in the Web Tools filter (empty uses UTC). This tool is free — no per-use charge. On by default, and on per chat."
    },
    "ENABLE_IMAGE_GENERATION": {
        "title": "Enable image generation",
        "group": "Files & Media/Image Generation",
        "detail": "Gives models OpenRouter's image-generation server tool — one they can call to create an image from a text prompt while answering.\n\nWhen a request calls for a picture, the model writes a prompt, OpenRouter generates the image server-side (one or more per turn) and returns it inline. An admin picks which model draws (`IMAGE_GENERATION_MODEL`) and a moderation level in the Image Generation filter; each user gets per-chat controls for quality, size, aspect ratio, background, and format behind a per-chat image switch that starts off. Generating images bills extra — per image or per token depending on the model; check current rates on OpenRouter. On by default."
    },
    "ENABLE_LZ4_COMPRESSION": {
        "title": "Compress stored artifacts",
        "group": "Storage/Compression",
        "detail": "When enabled, large artifact payloads are LZ4-compressed before being written to the pipe's database, cutting stored row size and read/write overhead.\n\nThe default `true` fits most deployments. Compression happens as part of encryption, so it runs only on artifacts that are actually encrypted and does nothing unless `Storage encryption key` is set; raise `Minimum size to compress` to leave small payloads uncompressed. When disabled, artifacts are stored uncompressed, though previously compressed rows still decompress automatically.\n\n**Tip:** Effective only when the `lz4` package is installed; otherwise the pipe stores data uncompressed and logs a warning."
    },
    "ENABLE_OPENROUTER_FUSION": {
        "title": "Enable OpenRouter Fusion",
        "group": "Filters & Integrations/Fusion",
        "detail": "Adds an `openrouter/fusion` model to the picker that answers a hard prompt by deliberation, not in a single pass — a panel of up to eight models replies in parallel, a judge model weighs them for consensus, contradictions, and gaps, and a final model writes the answer from that analysis.\n\nThis is Fusion's master switch: while on, that model stays in the picker and each chat shows the panel's takes, the judge's analysis, and the final answer. The preset, panel models, and judge are per-chat controls on the OpenRouter Fusion filter, and each user can force deliberation every turn with its `Always run Fusion` switch. Running a whole panel plus a judge for one answer costs far more than a normal completion — reserve it for research or high-stakes questions, and check OpenRouter's pricing. Turn it off and Fusion is off for everyone."
    },
    "ENABLE_OPENROUTER_IMAGE_GENERATION": {
        "title": "Show native image models",
        "group": "Files & Media/Image Generation",
        "detail": "Adds OpenRouter's dedicated image-generating models to the model picker, so users can choose a model whose whole job is producing images.\n\nThese native image-output models (such as Flux or Seedream) return pictures instead of text; the pipe discovers them and lists them as selectable models. Multimodal text-and-image models (the GPT-5 and Gemini image variants) stay in the normal chat list and gain image controls there instead. Each surfaced model carries an image filter exposing its generation settings — size, aspect ratio, and provider-specific options. Generating images bills extra; check current rates on OpenRouter. On by default.\n\n**Note:** Unlike `Enable image generation` (a chat model making a picture inline), this adds separate image-only models to pick from."
    },
    "ENABLE_PLUGIN_SYSTEM": {
        "title": "Enable plugin system",
        "group": "Plugins/General",
        "detail": "When enabled, the pipe activates its plugin system, letting its plugins intercept and transform requests, react to the model list, and wrap streamed replies.\n\nWhen disabled (the default), every plugin hook is skipped and plugins stay inert no matter which are present. The switch is re-read on each request, so flipping it applies to the next message with no restart. It is also the master gate for the bundled Pipe Dashboard: with this off, `Enable Pipe Dashboard` and its usage settings have no effect."
    },
    "ENABLE_REASONING": {
        "title": "Request reasoning traces",
        "group": "Reasoning & Thinking/General",
        "detail": "Whether reasoning-capable models are asked to return their step-by-step thinking alongside the answer, not just the final text.\n\nIt is on by default and applies site-wide, but each user can override it in their own Open WebUI settings, and their choice wins for their chats. Models without reasoning support are unaffected either way. `Default reasoning effort` and `Reasoning summary detail` shape what gets requested, while `Thinking display location` decides where a returned trace appears. The thinking itself is billed as output tokens, so leaving it on adds some cost on reasoning models — check OpenRouter's pricing."
    },
    "ENABLE_REDIS_CACHE": {
        "title": "Enable Redis write-behind cache",
        "group": "Reliability/Redis",
        "detail": "Lets the pipe reuse Open WebUI's Redis to buffer artifact writes and share cached reads across workers, instead of every worker hitting the database directly.\n\nIt engages only in a true multi-worker, Redis-backed deployment — needing `REDIS_URL`, `WEBSOCKET_MANAGER` set to `redis` with `WEBSOCKET_REDIS_URL`, more than one worker, and `redis-py` — so on a single-worker host it is a harmless no-op. When active, writes queue in Redis and flush to the database in the background. This is the master switch for the Redis settings below; turn it off to force direct database writes even in a cluster."
    },
    "ENABLE_SEARCH_MODELS": {
        "title": "Enable model catalog search",
        "group": "Tools/Built-in Server Tools",
        "detail": "Gives models an OpenRouter server tool they can call to look up which models the catalog offers and what they cost or do.\n\nWhen a user asks which model is best for a task, the model calls the tool, OpenRouter answers from the live catalog server-side, and the model replies from that instead of guessing. Each user gets a per-chat model-search switch that starts off. It is free, and unlike the other web tools it has no engine or model setting to configure. On by default."
    },
    "ENABLE_SSRF_PROTECTION": {
        "title": "Enable SSRF protection",
        "group": "Security/Network (SSRF)",
        "detail": "Blocks a chat-supplied image, file, or video download when the URL's host resolves to a private, loopback, link-local, or otherwise reserved address.\n\nOn by default, it defends against server-side request forgery — a link crafted to make the server reach places a user never could, such as `10.x` or `192.168.x` intranet hosts, `localhost`, or a cloud provider's `169.254.169.254` metadata endpoint. The host is resolved once and every resolved IPv4 and IPv6 address must be public; if any is internal the download is refused before a single byte is fetched, and a host that resolves to a mix of public and internal addresses is rejected outright. The connection is then pinned to the validated address so the name cannot be re-resolved to an internal target mid-download (a DNS-rebinding defense).\n\nTurning it off removes all of this, so the pipe will then follow a download URL to any address it resolves to — a genuine SSRF exposure. Plaintext HTTP is governed separately by `Allow plaintext HTTP downloads` and stays enforced even when this is off."
    },
    "ENABLE_STRICT_TOOL_CALLING": {
        "title": "Strict tool schemas",
        "group": "Tools/Execution",
        "detail": "Forces each tool's arguments into a strict JSON schema before the request reaches the model, so it can't return malformed or unexpected fields.\n\nEvery property becomes required (optional ones instead accept null), `additionalProperties` is forbidden, and any missing field type is filled in — which makes tool calls far more reliable. On by default. It applies only when `Tool execution location` is `Pipeline`; in `Open-WebUI` mode schemas are forwarded untouched. The strict schema is sent as-is to whatever provider serves the model, with no per-provider relaxation, so if a model or provider that lacks strict function calling starts rejecting calls, turn this off for it."
    },
    "ENABLE_SUBAGENT": {
        "title": "Enable subagent delegation",
        "group": "Tools/Built-in Server Tools",
        "detail": "Gives models OpenRouter's subagent server tool — one they can call to hand a self-contained subtask to a cheaper worker model.\n\nFor routine work the main model needn't do itself — summarizing, extracting data, drafting boilerplate — the model calls the subagent; OpenRouter runs a worker server-side that sees only that task (no conversation, no memory) and returns the result. An admin sets the worker in the Web Tools filter (`SUBAGENT_MODEL`; empty uses the chat's own model), and each user gets a per-chat subagent switch, off by default. It spends on an extra model call — usually cheaper than the main model, and boundable with the filter's `SERVER_TOOLS_MAX_COST_USD`; check current rates on OpenRouter. On by default."
    },
    "ENABLE_TIMING_LOG": {
        "title": "Enable function timing log",
        "group": "Logging/General",
        "detail": "A performance profiler that records how long the pipe's own internal functions take while handling a request.\n\nOff by default, the instrumentation short-circuits with effectively no overhead and writes nothing. When on, timings append to the file named in `Timing log file path`, and switching it on takes effect from the next request with no restart. It captures only function names, their durations, and a request id for correlation — never prompt or response text — so it is safe to enable for diagnosis.\n\n**Tip:** Enable only while profiling; the file is appended every request and never rotated."
    },
    "ENABLE_VIDEO_GENERATION": {
        "title": "Enable video generation",
        "group": "Files & Media/Video Generation",
        "detail": "Adds OpenRouter's video-generating models to the model picker, so users can pick one and produce a video from a text prompt (optionally starting from an image).\n\nThe chosen model looks like any other in the list; the user types a prompt, the pipe submits the job to OpenRouter, waits while it renders, then downloads the finished clip and shows it inline. Each video model carries a per-chat filter for duration, aspect ratio, resolution, audio, and a negative prompt. Generating video bills extra per clip — check current rates on OpenRouter. On by default.\n\n**Note:** Video models are always treated as non-ZDR, so they're excluded when the catalog is limited to zero-data-retention models."
    },
    "ENABLE_WEB_FETCH": {
        "title": "Enable web page fetch",
        "group": "Tools/Built-in Server Tools",
        "detail": "Gives models OpenRouter's web-fetch server tool — one they can call to open a specific web page or PDF the user names and read its contents.\n\nWhere web search finds pages, this reads a URL already in the conversation: paste a link and ask for a summary, and OpenRouter fetches it server-side and returns the text, following several URLs per turn. The Web Tools filter's settings choose the fetch engine, cap how many URLs and how much content per fetch, and allow or block domains. It bills on top of tokens (boundable by the filter's `SERVER_TOOLS_MAX_COST_USD`); check current rates on OpenRouter. On by default, but the per-chat switch starts off, so users opt in."
    },
    "ENABLE_WEB_SEARCH": {
        "title": "Enable web search",
        "group": "Tools/Built-in Server Tools",
        "detail": "Gives models OpenRouter's web-search server tool — one they can call to pull live web information into an answer, with citations.\n\nOpenRouter runs the search server-side when the model decides a question needs current facts. Enabling it is only the entry point: the Web Tools filter's settings choose the engine (OpenRouter's auto pick, the model's native engine, or a provider such as Exa or Perplexity), cap results and characters, and allow or exclude domains; each user sets search depth and location per chat. It bills per search on top of tokens — cap a request's total tool spend with the filter's `SERVER_TOOLS_MAX_COST_USD`, and check current rates on OpenRouter. On by default, and on per chat.\n\n**Note:** When on, it takes over from Open WebUI's own web search in that chat, so the two don't both run."
    },
    "ENCRYPT_ALL": {
        "title": "Encrypt all stored data",
        "group": "Security/Encryption",
        "detail": "Whether a set `Storage encryption key` protects every stored artifact or only reasoning traces.\n\nBy default it encrypts every persisted artifact at rest. Turning it off narrows encryption to reasoning-token artifacts and leaves the rest — tool-call results and other records — as plaintext JSON in the database. Either way it does nothing until `Storage encryption key` holds a value; with no key, everything is stored plaintext. Leave it on unless only reasoning content is sensitive and there is a reason to keep other records readable."
    },
    "ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE": {
        "title": "Endpoint conflict message",
        "group": "Error Messages/Templates",
        "detail": "The message users see when a request is refused because its content needs a different OpenRouter endpoint than the model is forced to.\n\nIt appears, and the reply is replaced, when an attachment or `preset` parameter that needs `/chat/completions` — a direct video upload is the common case — is sent to a model pinned to `/responses` by `Models forced to responses`. The text supports `{placeholder}` substitution and Handlebars-style `{{#if …}}` conditionals; representative fields include `{requested_model}`, `{enforced_endpoint}`, and `{reason}`, and any line whose placeholder is empty is dropped. The default explains the clash and advises removing the attachment or asking an admin to change the override. Clearing this field does not restore that built-in message — an empty value falls back to the generic OpenRouter error notice instead.\n\n**Tip:** Editing this only rewords the message; to actually let such requests through, adjust `Models forced to responses` instead."
    },
    "FALLBACK_STORAGE_EMAIL": {
        "title": "Fallback storage owner email",
        "group": "Files & Media/Fallback Storage",
        "detail": "Which Open WebUI account owns files the pipe re-hosts when a request arrives with no signed-in user behind it — an API automation, for example.\n\nThe pipe looks the account up by this email: an existing one becomes the owner; if none matches, a dedicated service account is auto-created using `Fallback storage display name` and `Fallback storage account role`. Point it at a real person's account to hand ownership there instead. Ownership decides who can later read these files — that account, an admin, or a user explicitly granted access — so treat it as an access decision. Clearing it falls back to the pipe's built-in service account; separately, the `OPENROUTER_STORAGE_USER_EMAIL` environment variable seeds the value pre-filled in this field.\n\n**Tip:** The resolved account is cached for the life of the process, so restart the pipe for a change here to take effect."
    },
    "FALLBACK_STORAGE_NAME": {
        "title": "Fallback storage display name",
        "group": "Files & Media/Fallback Storage",
        "detail": "The display name carried by the service account that owns uploads made when no signed-in user is present.\n\nPurely cosmetic — it labels that account wherever Open WebUI lists users. The account itself is found or created from `Fallback storage owner email`, and `Fallback storage account role` sets what it can do.\n\n**Warning:** The name is written only when the account is first created; changing it later does not rename an account that already exists."
    },
    "FALLBACK_STORAGE_ROLE": {
        "title": "Fallback storage account role",
        "group": "Files & Media/Fallback Storage",
        "detail": "The Open WebUI role handed to the service account the pipe auto-creates for uploads that arrive with no signed-in user.\n\nIt bites only when that account is first created — changing it later does not re-role an account that already exists, and clearing it restores the default. This matters only for the user-less path; ordinary chat uploads reuse the signed-in user and never create an account. The built-in default is Open WebUI's low-privilege `pending` role, the safe choice for an unattended, auto-created identity.\n\n**Warning:** Setting it to `admin`, `system`, or `owner` (matched case-insensitively) mints a standing account with broad access to the whole deployment — which the pipe also warns about — so keep it least-privilege."
    },
    "FINAL_USAGE_STATUS_STYLE": {
        "title": "Usage summary label style",
        "group": "Usage & Status/Status Display",
        "detail": "Whether the final usage status line labels each field with a word or a compact icon glyph.\n\nApplies only when `Show cost and token usage` is on; with that off, no usage line appears and this has no effect. Both styles carry the same figures — elapsed time, cost, and token counts — and differ only in how each field is labelled.\n\n- `text` — prefixes each field with a word, such as `Time:`, `Cost $`, and `Total tokens:`.\n- `icons` — swaps those words for the glyphs defined in `Usage summary icons`, giving a shorter line."
    },
    "FORCE_CHAT_COMPLETIONS_MODELS": {
        "title": "Models forced to chat completions",
        "group": "Connection & Routing/Endpoints",
        "detail": "Forces the listed models to use OpenRouter's `/chat/completions` endpoint instead of the default, via comma-separated glob patterns matched against each model id.\n\nEmpty by default, so no model is forced and endpoint choice follows `Default API endpoint`. A value such as `anthropic/*, openai/gpt-4.1-mini` routes every matching model to `/chat/completions`, which enables per-block prompt caching for Bedrock/Vertex-routed Claude. Patterns match both slash (`anthropic/*`) and dotted (`anthropic.*`) id forms and are case-sensitive. Globs are literal about the `~` prefix — `anthropic/*` does not cover `~anthropic/...` router aliases, so list `~anthropic/*` as its own pattern to include them.\n\nA trailing date stamp is stripped before matching, so `openai/gpt-4o` also catches `openai/gpt-4o-2026-01-15`, while a colon variant suffix like `:nitro` is kept and must be matched explicitly or with a `*`.\n\n**Tip:** `Models forced to responses` takes precedence, so a model matched by both lists still uses `/responses`."
    },
    "FORCE_RESPONSES_MODELS": {
        "title": "Models forced to responses",
        "group": "Connection & Routing/Endpoints",
        "detail": "Forces every model whose id matches one of these comma-separated glob patterns onto OpenRouter's `/responses` endpoint, ignoring the default.\n\nPatterns use shell-style wildcards (e.g. `anthropic/*, openai/gpt-4.1-mini`) and match a model's id in both slash and dotted form. Empty by default, so nothing is pinned and every request follows `Default API endpoint`. When a model matches both this list and `Models forced to chat completions`, this valve wins and the request uses `/responses`.\n\nA trailing date stamp is stripped before matching, so `openai/gpt-4o` also catches dated snapshots like `openai/gpt-4o-2026-01-15`; a colon variant suffix such as `:nitro` is kept, so match it explicitly or end the pattern with `*`. Matching is case-sensitive and OpenRouter ids are lowercase — an uppercase pattern like `ANTHROPIC/*` matches nothing. Globs are also literal about the `~` prefix, so `anthropic/*` does not cover `~anthropic/...` router aliases; add `~anthropic/*` as its own pattern.\n\n**Note:** A model pinned here still can't accept content that needs `/chat/completions` — a direct video upload or a `preset` parameter; such a request is refused with `Endpoint conflict message` rather than served."
    },
    "FREE_MODEL_FILTER": {
        "title": "Free model visibility",
        "group": "Models & Catalog/Catalog & Access",
        "detail": "Chooses whether users see only free OpenRouter models, never see them, or see everything — filtering the catalog by each model's price.\n\nA model counts as free only when its pricing is known and every pricing field sums to exactly zero; a model whose pricing is missing or unknown is treated as paid. This choice stacks with `Tool-calling model filter` and `Show only ZDR models`, so a model must clear every active filter to stay listed, and a model hidden here is also refused at request time with the `Blocked model message` rather than merely dropped from the picker.\n\n- `all` — no price filter; every imported model stays visible.\n- `only` — keeps just the free models, hiding everything with a price.\n- `exclude` — hides the free models, leaving the priced ones.\n\n**Tip:** Under `only`, a model with no pricing data in the catalog is treated as paid and dropped, even if it is in fact free."
    },
    "GEMINI_THINKING_BUDGET": {
        "title": "Gemini thinking budget",
        "group": "Reasoning & Thinking/General",
        "detail": "The baseline number of tokens Gemini 2.5 models may spend on hidden thinking before they answer.\n\nIt applies only to the Gemini 2.5 family; other models ignore it. `Default reasoning effort` scales this baseline — roughly a quarter of it at the lowest depth up to about four times it at the highest — so effort, not this number alone, sets the thinking each request gets. Those thinking tokens are billed as output; check OpenRouter's pricing.\n\n**Tip:** Set it to `0` to switch Gemini 2.5 thinking off entirely — the model then answers with no reasoning whatever effort is requested."
    },
    "HTTP_CONNECT_TIMEOUT_SECONDS": {
        "title": "Connection timeout",
        "group": "Connection & Routing/HTTP & Timeouts",
        "detail": "How long the pipe waits to open the TCP/TLS connection to OpenRouter before failing, in seconds.\n\nThere is no upper bound; the default drops unreachable or stalled endpoints fast while tolerating normal latency. Lower it to fail fast when OpenRouter is down, or raise it on slow networks where the handshake needs longer. It limits only connection setup — the length of an active request is governed separately by `Total request timeout` and `Idle read timeout` — and it also seeds the connect timeout for downloading remote images and files, capped there at 60 seconds."
    },
    "HTTP_REFERER_OVERRIDE": {
        "title": "App attribution URL",
        "group": "Usage & Status/Request Identity",
        "detail": "The app-attribution URL this deployment reports to OpenRouter in the `HTTP-Referer` header.\n\nMust be a full URL including the scheme, such as `https://chat.example.com`; a bare hostname or any value missing `http://` or `https://` is ignored and a warning is shown, then requests fall back to the pipe's built-in project URL. When left blank (the default), that same built-in URL is used, so most deployments can leave this empty. The chosen value applies globally to every OpenRouter chat, image, and video request from this pipe.\n\n**Tip:** Leave blank unless OpenRouter-side traffic should be attributed to this deployment's own URL instead of the pipe's default."
    },
    "HTTP_SOCK_READ_SECONDS": {
        "title": "Idle read timeout",
        "group": "Connection & Routing/HTTP & Timeouts",
        "detail": "How long an OpenRouter connection may sit idle — no streamed data arriving — before it is aborted, in seconds.\n\nThere is no ceiling; the default is deliberately generous so slow providers that pause between tokens are not cut off mid-stream. Lower it to drop stalled or dead connections sooner and free the request, or raise it only for providers with unusually long gaps between chunks.\n\n**Warning:** This applies only while `Total request timeout` is disabled (unset); when a total timeout is set, this value has no effect."
    },
    "HTTP_TOTAL_TIMEOUT_SECONDS": {
        "title": "Total request timeout",
        "group": "Connection & Routing/HTTP & Timeouts",
        "detail": "An overall time limit, in seconds, for each OpenRouter HTTP request, including the full streamed response.\n\nIt defaults to unset (`null`), which disables the total timeout so long streaming replies are never cut off partway; while unset, the idle-read limit `Idle read timeout` (default `300` seconds) instead guards against a stalled stream. Provide any value of `1` or higher to cap the entire request — connect, read, and streaming combined — after that many seconds; once a total is set, that idle-read limit no longer applies. Set a bound like `600` only to force runaway requests to fail hard.\n\n**Warning:** A total timeout also bounds active streams, so a value shorter than the model's real generation time aborts long answers mid-response; leave it unset for open-ended streaming."
    },
    "IMAGE_INPUT_SELECTION": {
        "title": "Image input reuse",
        "group": "Files & Media/Uploads & Limits",
        "detail": "Which images are forwarded to the model for the current user message.\n\nWith the default, when the user sends text and no new image, the most recent assistant-generated image is reused, so follow-ups like \"make it brighter\" keep acting on the prior picture. Neither mode resends images from earlier user turns. The forwarded count is capped by `Maximum images per request`, and the fallback fires only when the model accepts image input.\n\n- `user_turn_only` — forwards only images attached to the current message; no reuse.\n- `user_then_assistant` — adds the latest assistant-generated image(s) when the user attaches none (default)."
    },
    "IMAGE_UPLOAD_CHUNK_BYTES": {
        "title": "Image read buffer size",
        "group": "Files & Media/Uploads & Limits",
        "detail": "How much of an Open WebUI-hosted image or file the pipe reads into memory at a time while base64-encoding it to forward to a provider.\n\nEach concurrent encode holds one buffer, so a smaller value reduces peak memory when many users inline files at once, while a larger one costs memory for slightly fewer read passes on big files. The effective size is also capped by `Maximum base64 upload size`, so if that limit is smaller than this value, the smaller one applies.\n\n**Tip:** Rarely needs changing; lower it only if a worker shows memory pressure while many users attach images at once."
    },
    "INSUFFICIENT_CREDITS_TEMPLATE": {
        "title": "Out of credits message",
        "group": "Error Messages/Templates",
        "detail": "Markdown message shown to the end user when OpenRouter rejects a request with HTTP `402` because the account is out of credits.\n\nIt is selected automatically only for status `402`; other errors use their own templates. The text supports `{placeholder}` substitution — for example `{error_id}`, `{required_cost}`, and `{account_balance}` — plus `{{#if placeholder}}...{{/if}}` conditional blocks that render a line only when its value is present, so missing fields leave no blank rows. The default links to the OpenRouter credits and usage pages, and clearing the field falls back to the generic OpenRouter error notice.\n\n**Tip:** The `{support_email}` line stays hidden unless the `Support email` valve is filled in."
    },
    "INTERNAL_ERROR_TEMPLATE": {
        "title": "Unexpected error message",
        "group": "Error Messages/Templates",
        "detail": "Defines the Markdown message shown to the user when an unexpected, uncategorized internal error interrupts a request.\n\nServes as the catch-all rendered only after more specific handlers (API, timeout, connection, and `5xx` service errors) don't match, so it surfaces for genuinely unexpected failures in both streaming and non-streaming replies. Supports `{placeholder}` substitution for values such as `{error_id}`, `{error_type}`, and `{support_email}`, plus Handlebars-style `{{#if support_email}}...{{/if}}` blocks that render a section only when its value is set. The `{support_email}` and `{support_url}` slots are filled from the `Support email` and `Support link` valves, and clearing this field falls back to the generic OpenRouter error notice.\n\n**Tip:** Keep the `{error_id}` line, as it is the only handle a user can quote to tie their report back to the server logs."
    },
    "LOG_LEVEL": {
        "title": "Log verbosity level",
        "group": "Logging/General",
        "detail": "How much detail the pipe writes to its logs, from full `DEBUG` tracing down to `CRITICAL`-only.\n\nEach level hides everything less severe than itself. The effective default follows the `GLOBAL_LOG_LEVEL` environment variable, resolving to `INFO` when that variable is unset or holds an unrecognized value. It applies to every request and cannot be overridden per user.\n\n- `DEBUG` — full internal detail; very high volume, intended for development.\n- `INFO` — routine operational messages (the fallback default).\n- `WARNING` — only warnings and more severe events.\n- `ERROR` — only failed operations and worse.\n- `CRITICAL` — only fatal, process-threatening errors.\n\nKeep `INFO` or `WARNING` for production; raise verbosity to `DEBUG` only while diagnosing an issue, then revert to avoid oversized, noisy log files."
    },
    "MAX_CONCURRENT_REQUESTS": {
        "title": "Concurrent request limit",
        "group": "Streaming & Performance/Concurrency",
        "detail": "How many OpenRouter requests may be in flight at once across the entire process.\n\nEach active request holds one slot and the rest wait for one to free, so a higher ceiling allows more simultaneous traffic while a lower one protects memory and upstream rate limits. The limit is process-global — one shared semaphore across every user, not a per-request cap.\n\n**Warning:** Raising it takes effect immediately, but lowering it only applies after a restart; until then the process keeps the previous, higher limit."
    },
    "MAX_CONCURRENT_VIDEO_GENS": {
        "title": "Maximum concurrent video jobs",
        "group": "Files & Media/Video Generation",
        "detail": "How many video generation jobs may run at once across the whole pipe process, counting all users together.\n\nBy default only two jobs run process-wide, so a third waits for a free slot rather than being rejected. A slot is held for a job's entire lifecycle — from submission through status polling until it finishes, times out, or fails — then handed to the next waiter. Each user is separately capped by `Concurrent video jobs per user`, checked first, which turns away a user's excess jobs instead of queuing them.\n\n**Tip:** Keep it low to bound provider cost and the memory of concurrent video downloads; raise it only on a well-resourced host."
    },
    "MAX_CONCURRENT_VIDEO_GENS_PER_USER": {
        "title": "Concurrent video jobs per user",
        "group": "Files & Media/Video Generation",
        "detail": "How many video generation jobs one user may have running at the same time within a single pipe process.\n\nBy default each user may run two videos at once. Once a user is already at the cap, a further request — a new generation or a resumed one — is rejected immediately with a \"limit reached\" notice rather than queued. Each request must also fit under the process-wide `Maximum concurrent video jobs` pool, so this only bounds one user's share.\n\n**Tip:** Enforced per worker process, so a user's real ceiling is this value times the number of pipe workers."
    },
    "MAX_FUNCTION_CALL_LOOPS": {
        "title": "Maximum tool call rounds",
        "group": "Tools/Execution",
        "detail": "Caps how many model-and-tool rounds one reply runs before the pipe stops and makes the model write its final answer.\n\nOne round is a model to tools to model pass. The generous default suits deep agentic chains, and there's no hard ceiling — raise it for longer multi-step work, or lower it (say `5`) to rein in runaway loops. When the cap is hit, pending calls receive a skipped-tool stub so the model answers from what it has rather than erroring. Only active under `Tool execution location` `Pipeline`; otherwise Open WebUI drives the loop.\n\n**Warning:** A very low value such as `1` stubs tool requests almost immediately, so complex tasks may answer from incomplete results."
    },
    "MAX_INPUT_IMAGES_PER_REQUEST": {
        "title": "Maximum images per request",
        "group": "Files & Media/Uploads & Limits",
        "detail": "How many input images the newest user message may forward to the provider in one request.\n\nIt applies only to the latest user turn and only when the model accepts image input; any images beyond the cap are dropped, with a `Dropped N extra image(s)` status shown in the chat. When that newest message carries no images of its own and `Image input reuse` is set to fall back to assistant output, this same cap limits how many recently generated images are resent instead. Set it to match what the vision models handle well per turn."
    },
    "MAX_PARALLEL_TOOLS_GLOBAL": {
        "title": "Global parallel tool limit",
        "group": "Tools/Execution",
        "detail": "Caps how many tool calls the whole pipe process may run at once, pooled across every active chat.\n\nIt sits above the per-chat limit set by `Per-request parallel tool limit`: each tool call must claim one of these process-wide slots as well, so this ceiling bounds the combined tool load of all requests together. The default is deliberately high and most deployments never reach it — lower it only when tool execution is overwhelming a shared backend. Like its sibling, it only bites while `Tool execution location` is `Pipeline`, since Open WebUI runs the tools in the other mode.\n\n**Warning:** Raising it takes effect immediately, but lowering it needs a pipe restart."
    },
    "MAX_PARALLEL_TOOLS_PER_REQUEST": {
        "title": "Per-request parallel tool limit",
        "group": "Tools/Execution",
        "detail": "Caps how many of a single request's tool calls run at once when the pipe executes tools itself.\n\nThe same value sets how many tool-worker tasks the pipe spawns for that request, so it fixes both the parallelism and the pool size; the request's remaining calls queue until a worker frees up. Set it to `1` for one-at-a-time execution, or raise it for requests that fan out to many independent tools. Every call must also claim a slot in the process-wide `Global parallel tool limit`. Only relevant while `Tool execution location` is `Pipeline`; `Open-WebUI` mode runs tools inside Open WebUI."
    },
    "MIDDLEWARE_STREAM_QUEUE_MAXSIZE": {
        "title": "Middleware stream buffer limit",
        "group": "Streaming & Performance/Streaming",
        "detail": "How many streamed items the pipe buffers per request in Open WebUI's middleware streaming bridge — the generator that feeds chunks to the UI.\n\nSetting `0` leaves the buffer unbounded, so the pipe never pauses while emitting and nothing is dropped, though a slow or stalled client lets buffered items pile up in memory. A positive cap bounds that memory: once the buffer fills, each further item waits up to `Middleware buffer wait timeout` and is then dropped, trading a little lost output for a fixed memory ceiling. The cap is per streaming request, so total buffering grows with the number of concurrent streams. Unlike the raw-chunk and event buffers, this one applies to every streaming reply regardless of endpoint.\n\n**Tip:** Rarely worth changing; leave it unbounded unless streaming responses consume too much memory under slow clients."
    },
    "MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS": {
        "title": "Middleware buffer wait timeout",
        "group": "Streaming & Performance/Streaming",
        "detail": "How long the pipe waits to hand one streamed update to a size-limited middleware buffer before giving up, in seconds.\n\nIt matters only when `Middleware stream buffer limit` is above `0`; with the default unbounded buffer it is never consulted. When a full buffer makes the wait expire, that single update is dropped and the stream continues — the pipe does not abort or disconnect the request. Setting `0` removes the wait, so a stalled client can block the pipe's producers indefinitely.\n\n**Tip:** Rarely worth changing; raise it only if a bounded buffer is dropping updates for slow clients."
    },
    "MIN_COMPRESS_BYTES": {
        "title": "Minimum size to compress",
        "group": "Storage/Compression",
        "detail": "The size an artifact must reach before the pipe bothers trying to LZ4-compress it.\n\nCompression is attempted before encryption, and only on artifacts that are actually being encrypted — so this and `Compress stored artifacts` have no effect unless `Storage encryption key` is set. At the default floor every payload is a candidate; raise it so payloads below the chosen size skip the compression attempt.\n\n**Tip:** Rarely worth changing. The store already keeps the original whenever compression fails to shrink a payload, so raise this only to save CPU when many tiny artifacts are persisted."
    },
    "MODEL_CATALOG_REFRESH_SECONDS": {
        "title": "Catalog refresh interval",
        "group": "Models & Catalog/Catalog & Access",
        "detail": "How long the pipe keeps its cached copy of the OpenRouter model list before fetching a fresh one, in seconds.\n\nAt its default the list refreshes about once an hour, trading catalog freshness against OpenRouter API traffic. Shorten it to surface newly added or withdrawn models within minutes at the cost of more frequent fetches; lengthen it when the model list rarely changes. If a fetch fails, the pipe keeps serving the last good catalog and retries with an exponential backoff that never waits longer than this interval. The same interval also governs the video and image catalogs when `Enable video generation` or `Show native image models` is on."
    },
    "MODEL_ID": {
        "title": "Model allowlist",
        "group": "Models & Catalog/Catalog & Access",
        "detail": "Comma-separated allowlist of OpenRouter model IDs the pipe publishes in Open WebUI and permits at request time.\n\nLeaving it at `auto` (or blank) imports every available model; listing IDs in the native `author/model` form narrows the catalog to those and turns away any other model with the `Blocked model message`. Matching is exact after normalization (letter case and trailing date stamps are ignored) — it is not a wildcard or glob, so a pattern such as `anthropic/*` matches nothing. A `:tag` or `@preset/slug` entry is permitted at request time but is surfaced in the picker only through `Model routing variants`.\n\n**Warning:** If none of the listed IDs match the catalog, the pipe publishes every model rather than none — so a fully mistyped list silently exposes the whole catalog."
    },
    "MODEL_RESTRICTED_TEMPLATE": {
        "title": "Blocked model message",
        "group": "Error Messages/Templates",
        "detail": "Defines the Markdown notice shown in chat when the pipe rejects the requested model instead of sending it to OpenRouter.\n\nIt appears whenever a model is disallowed — outside the `Model allowlist` or catalog, filtered by `Free model visibility` or `Tool-calling model filter`, or blocked by `Enforce ZDR routing` or `Show only ZDR models`. The default renders a `Model restricted` block naming the rejected model and the specific reasons, and clearing the field falls back to the generic OpenRouter error notice. It supports `{placeholder}` substitution and `{{#if name}}...{{/if}}` conditional sections; common placeholders include `{requested_model}`, `{restriction_reasons}`, and `{support_email}`.\n\n**Tip:** A line whose placeholder resolves to empty is omitted, so wrap optional fields in `{{#if name}}...{{/if}}` to avoid blank bullets."
    },
    "NETWORK_TIMEOUT_TEMPLATE": {
        "title": "Network timeout message",
        "group": "Error Messages/Templates",
        "detail": "Defines the Markdown message shown in the chat when a request to OpenRouter exceeds its time budget and is aborted.\n\nIt renders only on a network timeout; connection failures and HTTP errors use their own templates. The text supports `{placeholder}` substitution — representative fields include `{timeout_seconds}` (the elapsed limit), `{error_id}` (a support-correlation code), and `{support_email}` (drawn from the `Support email` valve) — plus Handlebars-style `{{#if variable}}...{{/if}}` blocks that show a section only when that variable is set. The default supplies an explanation and troubleshooting steps, and clearing the field falls back to the generic OpenRouter error notice.\n\n**Tip:** Keep `{timeout_seconds}` in the template so the message states how long the pipe waited; other unknown `{name}` tokens render literally."
    },
    "NEW_MODEL_ACCESS_CONTROL": {
        "title": "Default access for new models",
        "group": "Models & Catalog/Catalog & Access",
        "detail": "Decides who can see and use each freshly imported OpenRouter model the moment the pipe first adds it to Open WebUI's model list.\n\nIt applies only at insert: a model the pipe later re-syncs keeps whatever access it already has, and grants an admin set by hand are never overwritten. `admins` is the cautious choice — vet new models before sharing them; `public` exposes everything the pipe imports to all users at once.\n\n- `admins` — creates no access grants, so the model stays private; administrators reach it only through Open WebUI's `BYPASS_ADMIN_ACCESS_CONTROL` setting, and everyone else needs an explicit grant.\n- `public` — grants read access to every user through a wildcard grant, so the model is visible to all.\n\n**Tip:** Because it acts only at insert, switching this valve later does not retroactively open or close models already imported — grant or revoke those in Open WebUI's model settings."
    },
    "OPENROUTER_ERROR_TEMPLATE": {
        "title": "Rejected request message",
        "group": "Error Messages/Templates",
        "detail": "Defines the Markdown message shown to end users when OpenRouter rejects a request, typically an HTTP `400` (bad request).\n\nIt is also the fallback for rejection statuses that have no dedicated template, such as `403` or `422`. The text supports `{placeholder}` substitution — for example `{heading}`, `{sanitized_detail}`, and `{request_id}` — plus Handlebars-style `{{#if name}}...{{/if}}` blocks that render only when the value is present, and any line whose placeholder resolves to a missing or empty value is dropped automatically. Leaving this blank does not produce an empty error — a minimal built-in provider-error message is shown instead.\n\n**Tip:** Placeholder names are matched exactly; a mistyped or unknown name like `{requst_id}` is left in the output verbatim rather than removed."
    },
    "PAYLOAD_TOO_LARGE_TEMPLATE": {
        "title": "Oversized request message",
        "group": "Error Messages/Templates",
        "detail": "Markdown shown to the user when OpenRouter rejects a request with HTTP `413` because the payload is too large.\n\nIt is rendered in place of the reply, so edit it to reword the notice, add local guidance, or translate it. The text supports `{placeholder}` substitution plus `{{#if placeholder}}...{{/if}}` blocks that render only when that value is present. Representative placeholders are `{error_id}` (a support-correlation code), `{model_identifier}`, and `{openrouter_message}`; `{support_email}` is filled from the `Support email` valve, and clearing the field falls back to the generic OpenRouter error notice.\n\n**Tip:** Keep `{error_id}` in the text so users can quote it — the same code is written to the server logs for correlation."
    },
    "PERSIST_REASONING_TOKENS": {
        "title": "Reasoning retention",
        "group": "Reasoning & Thinking/General",
        "detail": "How long a model's reasoning — its internal thinking — is kept and replayed back to it on later turns of the same chat, so it can build on what it worked out before.\n\nAs the site-wide default this keeps reasoning for the whole chat. Each user can override it in their own settings, and their choice wins; the per-user control itself defaults to a shorter window — only until the next reply — so many users retain less than the site default unless they raise it.\n\n- `disabled` — nothing is stored; every turn starts fresh, with none of the model's earlier thinking.\n- `next_reply` — thinking is kept only until the following reply finishes, then dropped.\n- `conversation` — thinking is kept across the whole chat and replayed on later turns."
    },
    "PERSIST_TOOL_RESULTS": {
        "title": "Keep tool results across turns",
        "group": "Tools/Execution",
        "detail": "When enabled, results returned by tool calls are retained across turns of the same conversation instead of being discarded after the current reply.\n\nOff by default: later turns see the assistant's own summaries of what tools found, and the model can re-run a tool when it needs the source again — keeping long conversations lean, since retained raw outputs ride every subsequent request as input tokens. Enable it when chats need to quote exact earlier tool outputs without re-fetching. This is a site-wide default each user can override in their own Open WebUI settings, and their choice wins for their chats. It has no effect while `Tool execution location` is `Open-WebUI`, where tool results are always ephemeral."
    },
    "RATE_LIMIT_TEMPLATE": {
        "title": "Rate limit message",
        "group": "Error Messages/Templates",
        "detail": "Markdown shown to the end user when OpenRouter rejects a request with HTTP `429` because too many requests were sent too quickly.\n\nApplies only to `429` responses; other status codes render their own templates. It supports `{placeholder}` substitution — for example `{error_id}`, `{retry_after_seconds}`, and `{rate_limit_type}` — plus `{{#if placeholder}}...{{/if}}` blocks that appear only when that value is present. The default renders a heading, the error ID, a retry-after hint, and back-off tips, its `{support_email}` line shows only when `Support email` is filled in, and clearing the field falls back to the generic OpenRouter error notice.\n\n**Tip:** Any line that mixes static text with a placeholder is dropped entirely when that value is missing or empty, so keep must-show wording on its own line."
    },
    "REASONING_EFFORT": {
        "title": "Default reasoning effort",
        "group": "Reasoning & Thinking/General",
        "detail": "How hard supported models think by default before answering, used whenever a request doesn't set its own level.\n\nThis is the site-wide default; each user can choose their own reasoning depth in Open WebUI, and their setting wins. Deeper effort tends to sharpen hard answers but spends more reasoning tokens — billed as output, so cost and latency climb; check OpenRouter's pricing. On Gemini 2.5 models the level scales `Gemini thinking budget`, and on Claude Opus/Sonnet the top level also requests the most detailed output. Background jobs such as titles and tags use `Background task reasoning effort` instead.\n\n- `none` — no reasoning is requested; the model answers directly.\n- `minimal` — the smallest amount of thinking.\n- `low` — light reasoning.\n- `medium` — a balanced, everyday amount.\n- `high` — deeper reasoning for harder problems.\n- `xhigh` — the most thinking, honored only by models built for that depth."
    },
    "REASONING_SUMMARY_MODE": {
        "title": "Reasoning summary detail",
        "group": "Reasoning & Thinking/General",
        "detail": "How much of a plain-language summary of its own reasoning a supported model is asked to produce.\n\nThis is the site-wide default; each user can override it in their own settings, and their choice wins. Only models that expose reasoning summaries honor it, and it shapes a trace only while reasoning is actually being requested (see `Request reasoning traces`).\n\n- `auto` — the model decides how long its summary should be.\n- `concise` — asks for a short, high-level summary.\n- `detailed` — asks for a fuller, more thorough summary.\n- `disabled` — no reasoning summary is requested at all."
    },
    "REDIS_CACHE_TTL_SECONDS": {
        "title": "Artifact cache lifetime",
        "group": "Reliability/Redis",
        "detail": "How long a just-written artifact stays readable in Redis before it expires and readers fall back to the database.\n\nWhile the write-behind cache is active, freshly persisted rows are cached so a peer worker can read them before they reach the database; this sets that cache's lifetime. Too short and entries vanish before another worker reads them, forcing an early database hit; longer keeps them available at the cost of more Redis memory. The same lifetime also floors how long re-queued writes survive when a database flush fails. It does nothing unless `Enable Redis write-behind cache` is active."
    },
    "REDIS_FLUSH_FAILURE_LIMIT": {
        "title": "Flush-failure alert threshold",
        "group": "Reliability/Redis",
        "detail": "How many flush cycles must fail in a row before the pipe raises a single critical alert that Redis write-behind is in trouble.\n\nThe count is consecutive, so one good flush resets it and only sustained failure reaches the threshold, where exactly one critical line fires. Hitting it does **not** switch write-behind off: the flusher keeps retrying on a widening back-off — seconds at first, stretching to a few minutes — and resumes the moment flushes succeed, while new writes fall back to direct database writes meanwhile. So this tunes when the alert fires, not when the pipe gives up (it never does). Lower it to catch a brief blip; raise it to alert only on a prolonged outage.\n\n**Tip:** No effect unless `Enable Redis write-behind cache` is on and Redis is active."
    },
    "REDIS_PENDING_WARN_THRESHOLD": {
        "title": "Pending queue warning threshold",
        "group": "Reliability/Redis",
        "detail": "The pending-write backlog at which the flusher starts logging a warning that Redis buffering is falling behind.\n\nOn each flush cycle the background flusher measures how many artifact writes are still queued in Redis; above this count it logs a `WARNING` instead of a quiet `DEBUG` line, flagging that writes are piling up faster than they drain to the database. It is purely diagnostic — it never throttles writes, drops data, or disables the cache — and runs only while `Enable Redis write-behind cache` is active.\n\n**Tip:** Lower it to hear about a backlog sooner; raise it if short bursts make the log noisy."
    },
    "REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS": {
        "title": "Initial download retry delay",
        "group": "Files & Media/Remote Downloads",
        "detail": "How long the pipe waits before the first retry after a remote image or file download fails, in seconds.\n\nIt also sets the floor for the backoff: each later retry waits roughly twice as long as the one before, and no single wait may exceed the `Download retry time budget`. A server's `Retry-After` header can push an individual wait higher still. Raising this slows every retry, so keep it well under the retry-time budget — one long opening pause can spend the whole budget and abandon the download before a second try. It has no effect when `Maximum download retries` is `0`."
    },
    "REMOTE_DOWNLOAD_MAX_RETRIES": {
        "title": "Maximum download retries",
        "group": "Files & Media/Remote Downloads",
        "detail": "How many extra times the pipe re-attempts a remote image or file download after a transient failure.\n\nThese retries are on top of the first try, which always runs — so `0` means one attempt and no retry, making a single failure final. Only transient problems requeue: dropped connections, timeouts, and responses such as `429`, `408`, or a `5xx`; a definite reply like `404` fails at once. Raise it to ride out flaky hosts, at the cost of a slower give-up on a truly dead link.\n\n**Tip:** The `Download retry time budget` can end the loop first — once that wall-clock budget is spent no further attempt starts, so a high count does nothing when the budget is small."
    },
    "REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS": {
        "title": "Download retry time budget",
        "group": "Files & Media/Remote Downloads",
        "detail": "How long, in wall-clock seconds, the pipe keeps retrying one failed remote image or file download before giving up and skipping it.\n\nBefore each retry it checks elapsed time, and once this budget is spent the download is abandoned — so a slow or repeatedly failing host can't stall a chat turn. Whichever trips first, this budget or the `Maximum download retries` count, ends the loop, and it also caps any single backoff wait to this length. It applies per process to every remote download.\n\n**Tip:** Only retries are time-checked, never the first attempt, so it has no effect when `Maximum download retries` is `0`."
    },
    "REMOTE_FILE_MAX_SIZE_MB": {
        "title": "Maximum remote file size",
        "group": "Files & Media/Remote Downloads",
        "detail": "How large a remote file or image the pipe will pull from a URL, in megabytes, before skipping it.\n\nThe download is aborted the moment its `Content-Length` header or streamed byte count crosses the limit, so an oversized file is dropped mid-transfer, not attached. Raise it to allow large media; lower it to bound per-request memory and bandwidth on untrusted links. A URL must also clear `Enable SSRF protection` and the plaintext-HTTP policy, which can block it outright first.\n\nOpen WebUI's RAG (document-retrieval) upload limit can override this: set higher than the RAG limit, downloads are still held to it; left at the default while RAG allows more, the cap rises to the RAG limit. Set a value at or below the RAG limit to keep full control."
    },
    "REMOTE_VIDEO_MAX_SIZE_MB": {
        "title": "Maximum generated video size",
        "group": "Files & Media/Video Generation",
        "detail": "The largest a generated video may grow, in megabytes, before its download is aborted and the generation job fails.\n\nThe default fits typical short generated clips. This same limit also caps any input video or audio reference attachment encoded into a generation request. Raise it for longer or higher-resolution outputs; set it below the real file size and a finished video is discarded before the download completes, surfacing as a failed job rather than a delivered result."
    },
    "SAVE_FILE_DATA_CONTENT": {
        "title": "Re-host inline file data",
        "group": "Files & Media/Uploads & Limits",
        "detail": "When on, base64 blobs and remote links carried in an uploaded file's `file_data` field are pulled into Open WebUI storage and swapped for a storage reference.\n\nOn by default, which keeps chat history from ballooning with raw inline payloads. Turn it off and `file_data` is forwarded untouched, leaving the original base64 or link inline in every saved turn — and, for a remote link, sent on to OpenRouter as-is. This governs only the `file_data` field; the separate `file_url` field is handled by `Re-host remote file URLs`."
    },
    "SAVE_REMOTE_FILE_URLS": {
        "title": "Re-host remote file URLs",
        "group": "Files & Media/Uploads & Limits",
        "detail": "When on, `http`/`https` and `data:` links in an uploaded file's `file_url` field are downloaded or decoded and re-hosted in Open WebUI storage, replacing the link with a storage reference.\n\nOn by default, so a chat stays replayable even if the original link later dies — at the cost of storage growth. Turn it off to forward the original URL to OpenRouter untouched: storage stays lean, but replay then depends on that third-party link surviving. This covers only the `file_url` field; inline `file_data` is handled by `Re-host inline file data`, and remote fetches remain subject to `Enable SSRF protection`."
    },
    "SEND_CACHE_SESSION_ID": {
        "title": "Pin conversation to one provider",
        "group": "Prompt Caching/General",
        "detail": "Sends OpenRouter a stable per-conversation key so every turn of one chat routes to the same provider, keeping that provider's prompt cache warm.\n\nThe key is an opaque `HMAC-SHA256` digest of the Open WebUI chat id — the raw id never leaves the pipe and the provider cannot reverse it — but it is deterministic, so each conversation yields one steady token. The provider can pin that chat's turns together, yet cannot link the token to the chat, the user, or any other conversation.\n\nWith the key, affinity starts from the very first request rather than only after OpenRouter observes a cache hit. Separate from `Attach session ID`, which sends the raw session id as attribution metadata and does not affect routing.\n\n**Warning:** Has no effect unless `WEBUI_SECRET_KEY` is set, and a manually set provider-routing order takes precedence over it."
    },
    "SEND_CHAT_ID": {
        "title": "Attach conversation ID",
        "group": "Usage & Status/Request Identity",
        "detail": "Adds the raw Open WebUI chat id to a request's `metadata.chat_id`, so the provider can group every turn of one conversation.\n\nOff by default, metadata-only, with no effect on routing or billing. The value is the real Open WebUI conversation GUID in clear text, identical on every turn of that thread, so the provider can re-identify and follow an entire conversation over time.\n\nThis is the very chat id that `Pin conversation to one provider` feeds through a one-way hash for cache routing — but here it goes out as-is, purely as an attribution tag the provider can read. Enable it when incident response needs to trace activity to one conversation; leave it off otherwise."
    },
    "SEND_END_USER_ID": {
        "title": "Attach end-user ID",
        "group": "Usage & Status/Request Identity",
        "detail": "Sends the user's stable Open WebUI account id to OpenRouter on every request, so the provider can group all of one person's traffic together.\n\nOff by default; turn it on for provider-side abuse attribution. The id goes out both as the top-level `user` field — OpenRouter's end-user abuse identifier — and as `metadata.user_id`, and it is the same account GUID on every request that user ever sends, so the provider can re-identify and cluster an individual's activity over time.\n\nThis is the strongest privacy exposure in this group: unlike `Pin conversation to one provider`, whose key is an opaque one-way hash, here the real, re-identifiable identifier leaves the pipe. The pipe uses the authentic Open WebUI id and discards any `user` a client tries to supply, so it cannot be forged."
    },
    "SEND_MESSAGE_ID": {
        "title": "Attach message ID",
        "group": "Usage & Status/Request Identity",
        "detail": "Adds the raw Open WebUI message id to a request's `metadata.message_id`, tagging the single chat message that produced the request.\n\nOff by default, metadata-only, with no effect on routing or billing. This is the finest-grained identifier of the family: where the user, session, and conversation tags group many requests together, a message id pins one request to one specific message in the Open WebUI database.\n\nThat makes it the key for after-the-fact tracing — the pipe's encrypted session-log archives are named by user, chat, and message id, so a message id quoted in an OpenRouter abuse report leads straight to the stored record for that exact turn. The raw GUID is sent in clear text, so leave it off if per-message identity should not reach the provider."
    },
    "SEND_SESSION_ID": {
        "title": "Attach session ID",
        "group": "Usage & Status/Request Identity",
        "detail": "Adds the raw Open WebUI session id to a request's `metadata.session_id`, so the provider can group everything sent within one sign-in session.\n\nOff by default, and metadata-only — it never touches the top-level `session_id` (that field is always the cache-affinity pin from `Pin conversation to one provider`) and has no effect on routing or billing. The value is the actual Open WebUI session GUID in clear text, stable for the life of that session, so it re-identifies and clusters a burst of activity from the same login.\n\nEnable it, alongside the other identifier tags, when provider-side logs need to attribute abuse or trace an incident down to a single session; leave it off to keep the raw session id out of the provider's hands."
    },
    "SERVER_TIMEOUT_TEMPLATE": {
        "title": "Server timeout message",
        "group": "Error Messages/Templates",
        "detail": "The message users see when OpenRouter accepts a request but times out server-side (`408`) before replying.\n\nIt appears only for that server-side `408` timeout; other error codes render their own templates. Supports `{placeholder}` substitution — representative fields are `{error_id}`, `{openrouter_code}`, and `{openrouter_message}` — plus `{{#if placeholder}}...{{/if}}` blocks that render a line only when that field has a value. The `{support_email}` placeholder is filled from the `Support email` valve.\n\n**Warning:** Clearing this field shows a generic built-in provider-error notice instead of this timeout-specific message."
    },
    "SERVICE_ERROR_TEMPLATE": {
        "title": "Service outage message",
        "group": "Error Messages/Templates",
        "detail": "Markdown message shown to the user when OpenRouter returns a server-side (HTTP `5xx`) error instead of a valid response.\n\nCovers both streaming and non-streaming requests once a provider replies with status `500` or above; `4xx` failures fall to other templates. It supports `{placeholder}` substitution — representative fields are `{error_id}` (a random support-correlation code), `{status_code}`, and `{reason}` — plus `{{#if field}}...{{/if}}` blocks that render only when that field is set. Edit it for local wording or to point users at an in-house contact; the support line appears only when `Support email` is configured.\n\n**Tip:** A line whose placeholder resolves to empty is dropped entirely, so keep optional fields such as `{reason}` on their own line."
    },
    "SESSION_LOG_ASSEMBLER_BATCH_SIZE": {
        "title": "Archive assembly batch size",
        "group": "Logging/Session Logs",
        "detail": "How many finished message turns the background assembler packs into zip archives on each pass.\n\nThe cap is applied separately to freshly-completed turns and to crash-stranded ones, so a single pass can seal up to twice this many archives. It only matters while `Enable session log storage` is on. Set it too low under heavy logging and older turns wait for a later pass — timed by `Archive assembly interval` — so raise it to clear a backlog faster, at the cost of a larger burst of database reads and zip writes each pass.\n\n**Tip:** Rarely needs changing; raise it only if completed-turn archives visibly lag behind under sustained traffic."
    },
    "SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS": {
        "title": "Archive assembly interval",
        "group": "Logging/Session Logs",
        "detail": "How often the background assembler wakes to turn staged session-log segments into per-message zip archives.\n\nA random extra delay (up to `Archive assembly jitter`) is added to every wait, and each worker process runs its own assembler, so multiple workers don't scan the database in lockstep. It has effect only while `Enable session log storage` is on — otherwise every wake returns immediately without touching the database. Lower it so a completed turn's archive appears sooner, or raise it to reduce how often the database is polled.\n\n**Tip:** Rarely needs changing; shorten it only if finished-turn archives take too long to appear."
    },
    "SESSION_LOG_ASSEMBLER_JITTER_SECONDS": {
        "title": "Archive assembly jitter",
        "group": "Logging/Session Logs",
        "detail": "How much random slack the assembler adds around its scans so multiple worker processes don't hit the database in lockstep.\n\nIt waits a random amount up to this value before its very first scan, then adds another fresh random amount up to this value on top of `Archive assembly interval` before each later scan. Set it to `0` to remove the jitter and run exactly on the interval. It only matters when more than one worker process is running.\n\n**Tip:** Rarely needs changing; raise it only if several workers still spike the database at the same instant."
    },
    "SESSION_LOG_CLEANUP_INTERVAL_SECONDS": {
        "title": "Expired archive cleanup interval",
        "group": "Logging/Session Logs",
        "detail": "How often, in seconds, the background sweep runs that deletes archives past their retention age.\n\nThe default of one hour suits a retention window measured in days; the sweep removes any archive older than `Archive retention period`. Because deletion only happens on each pass, a longer interval mainly lets expired files sit up to one interval longer before removal — it does not change what counts as expired. Runs only while `Enable session log storage` is on, and a change is picked up on the next persisted request without a restart.\n\n**Tip:** Rarely needs changing; shorten it only to reclaim expired archives sooner."
    },
    "SESSION_LOG_DIR": {
        "title": "Session log directory",
        "group": "Logging/Session Logs",
        "detail": "Where on disk the pipe writes its encrypted session-log archives.\n\nBy default it is a bare relative name, so archives land under whatever working directory the Open WebUI process happens to start in — set an absolute path such as `/var/lib/open-webui/session_logs` to pin the location, and a leading `~` expands to the home directory. Archives are written as `<user_id>/<chat_id>/<message_id>.zip` beneath this root, with the tree created on first write. Because those archives hold whole conversations, restrict filesystem access to this directory even though each zip is encrypted. Leaving it blank while `Enable session log storage` is on skips all persistence and logs a warning; the background cleanup only ever removes `.zip` files and empty folders under this root.\n\n**Tip:** Use an absolute path in production so archives always land in the same place."
    },
    "SESSION_LOG_FORMAT": {
        "title": "Stored log format",
        "group": "Logging/Session Logs",
        "detail": "Which log files each stored session-log archive contains: machine-readable `jsonl`, human-readable `text`, or `both`.\n\nApplies only when `Enable session log storage` is on. A `logs.jsonl` file is always written regardless of this setting, so this choice governs only whether a plain-text `logs.txt` is added alongside it.\n\n- `jsonl` — archive holds only `logs.jsonl`, one JSON object per log record; the default.\n- `text` — adds a human-readable `logs.txt` next to the always-present `logs.jsonl`.\n- `both` — writes `logs.txt` and `logs.jsonl`; the same file set as `text`.\n\n**Tip:** `both` is equivalent to `text` here since `logs.jsonl` is always written; pick `jsonl` to keep archives smallest."
    },
    "SESSION_LOG_LOCK_STALE_SECONDS": {
        "title": "Assembly lock stale timeout",
        "group": "Logging/Session Logs",
        "detail": "How long an assembly lock may sit in the database before another worker treats it as abandoned and reclaims it.\n\nWhile the assembler zips one message turn's logs it briefly holds a database lock so no other worker writes the same archive at once. If a worker dies mid-write, that lock would otherwise keep the turn stuck forever; the default timeout sits far above a healthy run — which finishes in well under a second — so only a genuinely orphaned lock is reclaimed.\n\n**Tip:** Rarely needs changing; shorten it only if crashed workers leave archives stuck unassembled and their locks must free up sooner."
    },
    "SESSION_LOG_MAX_LINES": {
        "title": "Per-request log line limit",
        "group": "Logging/Session Logs",
        "detail": "Caps how many structured log records the pipe keeps in memory for each request before the oldest roll off.\n\nEach in-flight request keeps its own buffer, so this bounds memory per concurrent request and caps how many records a single request contributes to its archive — an archive that merges several request-segments for one message can hold more. The default comfortably holds a full request even at verbose log levels; set it too low and long, chatty requests get their earliest lines silently truncated, while very high values grow memory under heavy concurrency.\n\n**Tip:** Rarely needs changing unless verbose session logs are being cut off."
    },
    "SESSION_LOG_RETENTION_DAYS": {
        "title": "Archive retention period",
        "group": "Logging/Session Logs",
        "detail": "How old a stored archive may get before the cleanup sweep deletes it, measured in days from the file's last-modified time.\n\nThe default keeps roughly three months for audit or incident review; lower it to reclaim disk sooner, raise it to retain longer. This sets the age threshold only — how often the sweep actually runs is `Expired archive cleanup interval`, so an expired archive can linger up to one interval past its deadline. Takes effect only while `Enable session log storage` is on."
    },
    "SESSION_LOG_STALE_FINALIZE_SECONDS": {
        "title": "Crashed request finalize delay",
        "group": "Logging/Session Logs",
        "detail": "How long the pipe waits before sealing a session-log archive for a message turn that crashed without signalling it finished.\n\nA turn is normally sealed the instant its final segment is staged; a crashed or killed one never stages that terminal marker, so its partial logs would otherwise sit staged indefinitely. After this many idle seconds since the last segment, the assembler seals what it has and tags the archive finalized-incomplete, so the gap stays visible rather than lost. The long default keeps a slow-but-alive turn from being sealed early; it applies only while `Enable session log storage` is on. Lower it to reclaim stranded logs sooner.\n\n**Warning:** Values under `300` seconds (five minutes) are raised to `300` at runtime."
    },
    "SESSION_LOG_STORE_ENABLED": {
        "title": "Enable session log storage",
        "group": "Logging/Session Logs",
        "detail": "Records the full content of each request — the user's prompts, the model's replies, tool calls, and provider errors — into an encrypted archive on the server's disk.\n\nOff by default, storing nothing. When on, the pipe writes one encrypted zip per message turn under a folder tree keyed by user, chat, and message id, capturing the exchange at full debug depth regardless of the console log level — so the file holds sensitive conversation data at rest. Nothing is written until `Archive encryption passphrase` is set and `Session log directory` is non-empty; any request missing its user, chat, message, or request id is skipped entirely.\n\n**Warning:** Anyone with the passphrase and read access to the directory can recover whole conversations — restrict both, and bound how long copies live with `Archive retention period`."
    },
    "SESSION_LOG_ZIP_COMPRESSION": {
        "title": "Session log archive codec",
        "group": "Logging/Session Logs",
        "detail": "Which compression algorithm the pipe uses when writing encrypted session-log archives to disk.\n\nApplies only while session-log storage is enabled. `lzma` produces the smallest archives but costs the most CPU per write; choose a lighter codec such as `deflated` if archiving can't keep pace under heavy load.\n\n- `stored` — no compression; fastest writes, largest files on disk.\n- `deflated` — standard zip DEFLATE; moderate size and speed.\n- `bzip2` — smaller than `deflated` at higher CPU cost.\n- `lzma` — smallest archives, highest CPU.\n\n**Tip:** `Session log compression level` tunes only `deflated` and `bzip2`; it is ignored for `stored` and `lzma`."
    },
    "SESSION_LOG_ZIP_COMPRESSLEVEL": {
        "title": "Session log compression level",
        "group": "Logging/Session Logs",
        "detail": "How hard `deflated`/`bzip2` compression works on session-log archives — more CPU for smaller files.\n\nLeft unset (the default), each codec uses its own built-in level. The `stored` and `lzma` codecs ignore this entirely, and because `Session log archive codec` defaults to `lzma`, the setting has no effect until that codec is switched to `deflated` or `bzip2`.\n\n**Tip:** Rarely worth changing; switch the codec first, then tune this."
    },
    "SESSION_LOG_ZIP_PASSWORD": {
        "title": "Archive encryption passphrase",
        "group": "Logging/Session Logs",
        "detail": "The AES passphrase that encrypts each session-log archive — and gates whether archives are written at all.\n\nApplies only when `Enable session log storage` is on. While empty (the default), every archive is skipped with a warning, so storage stays off until a passphrase is set; there is no unencrypted fallback. Sensitive: shown write-only in the panel, and stored encrypted at rest when `WEBUI_SECRET_KEY` is set. Anyone holding this passphrase can decrypt every archive it wrote.\n\n**Warning:** Rotating it affects only new archives; older ones open only with their original passphrase."
    },
    "SHOW_FINAL_USAGE_STATUS": {
        "title": "Show cost and token usage",
        "group": "Usage & Status/Status Display",
        "detail": "When enabled, the final status line under each reply reports elapsed time, request cost, and token usage instead of just how long the model thought.\n\nOn by default; when off, that line falls back to a plain `Thought for N.N seconds` with no cost or token figures. Cost is shown only when it is above zero, and the token breakdown lists a total plus input and output counts, adding cached and reasoning counts when those apply. This is a site-wide default each user can override in their own Open WebUI settings, and their choice wins for their chats. `Usage summary label style` and `Usage summary icons` shape how the line looks while it is on."
    },
    "SHOW_TOOL_CARDS": {
        "title": "Show tool activity cards",
        "group": "Tools/Execution",
        "detail": "When enabled, tool activity appears as collapsible chat cards showing each tool's name, arguments, and result; left off by default, tools run silently with no visual trace.\n\nCards cover both the tools this pipe runs itself and OpenRouter's built-in server tools — web search, fetch, datetime, advisor, subagent — and appear in progress while a reply streams. The pipe-run cards show only when `Tool execution location` is `Pipeline`; in `Open-WebUI` mode those calls go to Open WebUI's native tool display instead, though the server-tool cards still appear. It is a per-user-overridable default, and a user's own setting wins for their chats."
    },
    "SSE_WORKERS_PER_REQUEST": {
        "title": "Stream decoder workers",
        "group": "Streaming & Performance/Concurrency",
        "detail": "How many worker tasks parse a single streaming reply's raw chunks in parallel.\n\nThey run only on replies served over OpenRouter's `/responses` API — the usual path, chosen by `Default API endpoint` — where one producer reads the network stream and these workers turn its raw chunks into parsed events. Replies routed to `/chat/completions`, and all non-streaming replies, use a single decode path and ignore this entirely. One worker funnels every chunk through one parser; more workers keep decoding from falling behind when chunks arrive faster than a single task can handle, and each concurrent stream spawns its own set.\n\n**Tip:** Rarely worth changing — raise it only if profiling shows chunk decoding lagging behind a fast, high-volume stream."
    },
    "STREAMING_CHUNK_QUEUE_MAXSIZE": {
        "title": "Raw chunk buffer limit",
        "group": "Streaming & Performance/Streaming",
        "detail": "How many raw, unparsed chunks the pipe buffers from OpenRouter before backpressure pauses further reading.\n\nThis is the first queue on the `/responses` streaming path — where a producer reads network chunks and decoder workers parse them; replies on `/chat/completions` and non-streaming replies never use it. The count is of chunks, not bytes. Setting `0` leaves the buffer unbounded, the recommended value, because it can't trigger the stall a small bound can: once a bounded buffer fills, the OpenRouter read blocks until the pipeline drains. A cap below roughly `500`, under tool-heavy loads or slow database writes, risks hanging the whole stream.\n\n**Tip:** Leave it unbounded unless a deliberate memory cap is required, and never set it below `500`."
    },
    "STREAMING_CHUNK_QUEUE_WARN_SIZE": {
        "title": "Chunk backlog warning threshold",
        "group": "Streaming & Performance/Streaming",
        "detail": "How far the raw-chunk buffer may back up before the pipe logs a backlog warning.\n\nThis is monitoring only — it writes a rate-limited log line and never throttles, drops, or reshapes the stream. It watches the `/responses` path's first queue, the one bounded by `Raw chunk buffer limit`, so it does nothing for replies on `/chat/completions` or non-streaming replies. Because that buffer is unbounded by default, a rising backlog here is often the only early sign that parsing has fallen behind the incoming stream.\n\n**Tip:** Rarely worth changing — raise it only if backlog warnings flood the log on healthy high-load streams."
    },
    "STREAMING_DELTA_CHAR_LIMIT": {
        "title": "Stream batching toggle",
        "group": "Streaming & Performance/Streaming",
        "detail": "An on/off switch for batching streamed text and reasoning output; any value above `0` turns batching on.\n\nDespite the name, the exact number does nothing — the pipe only checks whether it is above `0`, so `256` and `4096` behave identically, and `Minimum flush size` is what sets the smallest batch. With batching on, small streamed fragments are grouped and flushed in larger pieces whenever the browser lags, cutting UI-update events under load on either endpoint; it never changes the text itself. Setting it to `0` does not on its own restore unbatched output — that also needs `Idle flush delay` at `0`, or the idle timer keeps batching alive.\n\n**Tip:** Rarely worth changing — to force raw one-to-one streaming, set this and `Idle flush delay` both to `0`."
    },
    "STREAMING_EVENT_QUEUE_MAXSIZE": {
        "title": "Parsed event buffer limit",
        "group": "Streaming & Performance/Streaming",
        "detail": "How many parsed events the pipe buffers between the decoder workers and forwarding to Open WebUI.\n\nThis is the second queue on the `/responses` streaming path, holding events the workers have already parsed; replies on `/chat/completions` and non-streaming replies never use it. The count is of events, not bytes. Setting `0` keeps it unbounded — recommended, since it can neither block nor deadlock. A positive cap makes a full buffer apply backpressure; below about `500`, under tool-heavy loads or when the database or UI drains slowly, it can stall the whole reply, because a saturated event buffer also backs up the `Raw chunk buffer limit` queue ahead of it and halts the OpenRouter read.\n\n**Tip:** Leave it unbounded unless buffered events during very long streams grow enough to pressure process memory."
    },
    "STREAMING_EVENT_QUEUE_WARN_SIZE": {
        "title": "Event backlog warning threshold",
        "group": "Streaming & Performance/Streaming",
        "detail": "How many parsed events may pile up before the pipe logs a backlog warning.\n\nThe effect is purely diagnostic — crossing it emits a rate-limited `WARNING` log line and never alters the stream. It watches the `/responses` path's event queue, the one bounded by `Parsed event buffer limit`, so it has no effect on replies routed to `/chat/completions` or on non-streaming replies. It matters most when that buffer is left unbounded (the recommendation), because then a growing backlog is the only signal that downstream processing has fallen behind.\n\n**Tip:** Rarely worth changing — raise it only if healthy streams flood the log with backlog warnings."
    },
    "STREAMING_IDLE_FLUSH_MS": {
        "title": "Idle flush delay",
        "group": "Streaming & Performance/Streaming",
        "detail": "How long buffered streaming text waits during an upstream pause before it flushes to the interface, in milliseconds.\n\nIt applies only while streamed deltas are being batched. The default flushes stalled text quickly so a paused reply never looks frozen; raise it to batch longer, at the cost of text sitting buffered during a pause. Setting `0` removes the timer, so buffered deltas then flush only when the next chunk arrives or the reply ends — if the model pauses mid-buffer, that text can sit until it resumes. Batching stops altogether only when this and `Stream batching toggle` are both `0`.\n\n**Tip:** Rarely worth changing unless streamed replies stutter or lag during model pauses."
    },
    "STREAMING_NAGLE_MIN_FLUSH_CHARS": {
        "title": "Minimum flush size",
        "group": "Streaming & Performance/Streaming",
        "detail": "The fewest buffered characters that must build up before a batch is released at the end of a drain cycle.\n\nThe default absorbs single-character jitter so brief low-traffic bursts aren't emitted one character at a time. Set `1` for pure Nagle, where any buffered delta flushes each cycle, or a larger value to batch more aggressively and cut UI-update events on chatty streams. It governs only the end-of-drain flush — a switch between text and reasoning, a structural event, the `Idle flush delay` timeout, and the end of the stream all flush regardless, so content is never held past that idle interval. It has no effect when `Stream batching toggle` and `Idle flush delay` are both `0`, where deltas stream one-to-one.\n\n**Tip:** Rarely worth changing — raise it only if a stream emits too many tiny update events."
    },
    "STREAM_INTERRUPTED_TEMPLATE": {
        "title": "Interrupted response message",
        "group": "Error Messages/Templates",
        "detail": "Markdown appended to the assistant's reply when a streamed response ends before the model signals completion.\n\nAny text already streamed is preserved and this notice is added below it, so the partial answer is never lost. It supports `{placeholder}` substitution plus Handlebars-style `{{#if name}}...{{/if}}` blocks that render a section only when its value is set; representative placeholders are `{model}`, `{timestamp}`, and `{support_email}`. Support details come from the `Support email` and `Support link` valves, and the built-in default reports the interruption and suggests retrying.\n\n**Warning:** Clearing this field does not disable the notice; a generic error message is appended instead."
    },
    "SUPPORT_EMAIL": {
        "title": "Support email",
        "group": "Error Messages/Support",
        "detail": "A contact email the pipe adds to the error messages users see when a request fails.\n\nAccepts one free-form address such as `support@example.com`. The default is empty, which omits the contact line so failure notices (out-of-credits, rate-limit, timeout, connection, and service errors) carry no support address. Set it to a monitored inbox so users have somewhere to report failures. Works alongside `Support link`, which adds a support link the same way."
    },
    "SUPPORT_URL": {
        "title": "Support link",
        "group": "Error Messages/Support",
        "detail": "A support link the pipe adds to error messages, such as a ticket-system or Slack-channel URL for users to reach help.\n\nWhen set to a value like `https://help.example.com/tickets`, that link is rendered as a support line in error notices, including internal errors and interrupted streams. The default empty string omits the line entirely, so error output shows only the error ID and any configured `Support email`. It applies globally to every user's error messages.\n\n**Tip:** This link appears to all end users in error output, so point it at a destination they are allowed to reach."
    },
    "TASK_MODEL_REASONING_EFFORT": {
        "title": "Background task reasoning effort",
        "group": "Reasoning & Thinking/General",
        "detail": "How hard the pipe's models think when handling Open WebUI's background chores — chat titles, tags, follow-up suggestions — rather than a user's actual reply.\n\nThese jobs are short and run often, so they favor speed and low cost over depth; this is separate from `Default reasoning effort` and applies only to task jobs that target models this pipe serves. Because task output is brief, higher levels mostly add latency and billed tokens for little gain.\n\n- `none` — no reasoning is requested for tasks.\n- `minimal` — the least thinking and the fastest task runs.\n- `low` — a light amount that balances speed with quality.\n- `medium` — deeper background reasoning when task quality needs it.\n- `high` — more depth, at more latency and cost.\n- `xhigh` — the most thinking, honored only by models that support that depth."
    },
    "THINKING_OUTPUT_MODE": {
        "title": "Thinking display location",
        "group": "Reasoning & Thinking/General",
        "detail": "Where a model's live thinking shows up while it works — the dedicated Open WebUI reasoning box, transient status lines, or both at once.\n\nThis is the site-wide default; each user can pick their own in Open WebUI settings, and their choice wins. It only matters for models that are actually returning a reasoning trace (see `Request reasoning traces`). Keeping thinking in the reasoning box, out of the visible answer, suits most deployments.\n\n- `open_webui` — thinking streams into the collapsible reasoning box, kept separate from the final answer.\n- `status` — thinking appears only as short-lived status lines above the reply; the reasoning box stays empty.\n- `both` — sends thinking to the reasoning box and the status line together, repeating it in two places.\n\n**Tip:** Choose `both` only when maximum visibility is worth showing the same text twice."
    },
    "TIMING_LOG_FILE": {
        "title": "Timing log file path",
        "group": "Logging/General",
        "detail": "Where timing events are written — one JSON object per line — while function timing capture is on.\n\nIt is used only when `Enable function timing log` is on; with that toggle off (the default) the path sits unused. The file and any missing parent folders are created on first write, and new events are appended rather than overwriting earlier runs. A relative value resolves against wherever the Open WebUI process happens to start, so use an absolute path like `/var/log/openrouter/timing.jsonl` to pin output to a fixed location.\n\n**Warning:** The file is only ever appended — never rotated or size-capped — so it grows without bound while timing capture stays enabled."
    },
    "TOOL_BATCH_CAP": {
        "title": "Tool batch size",
        "group": "Tools/Execution",
        "detail": "How many calls to the same tool the pipe will fire off together in one parallel batch.\n\nOnly independent calls to an identical tool are grouped — calls that depend on each other or are marked sequential always run on their own. A larger cap fans out more identical calls at once, raising simultaneous load; `1` turns batching off so each call runs singly. Applies only when `Tool execution location` is `Pipeline`.\n\n**Tip:** A whole batch shares one `Tool batch timeout` window, so a high cap lets a single slow call time out every call grouped with it."
    },
    "TOOL_BATCH_TIMEOUT_SECONDS": {
        "title": "Tool batch timeout",
        "group": "Tools/Execution",
        "detail": "How long a whole parallel batch of tool calls may run before the pipe cancels the batch and marks every call in it failed.\n\nIt guards a group of tools running together, where `Tool call timeout` guards each call on its own. The default gives a batch room to finish without being cut off; raise it for slow integrations, or lower it to reclaim worker slots sooner. The window is never shorter than `Tool call timeout` — a value below that per-call limit is quietly raised to it. Applies only while `Tool execution location` is `Pipeline`."
    },
    "TOOL_CALLING_FILTER": {
        "title": "Tool-calling model filter",
        "group": "Models & Catalog/Catalog & Access",
        "detail": "Chooses whether the model list keeps only tool-capable models, hides them, or shows every model — judged by each model's tool-calling support.\n\nA model is treated as tool-capable when OpenRouter reports the `tools` or `tool_choice` parameter for it; one whose parameters are unknown is treated as not tool-capable. When set to `only` or `exclude` it stacks with `Free model visibility` and `Show only ZDR models`, so a model must clear every active filter to appear.\n\n- `all` — no capability filter; every imported model stays listed.\n- `only` — keeps only tool-capable models.\n- `exclude` — hides tool-capable models, leaving those without tool support.\n\n**Warning:** A model hidden here is also refused per request — a direct call to it returns the `Blocked model message`, not just a missing entry in the list."
    },
    "TOOL_EXECUTION_MODE": {
        "title": "Tool execution location",
        "group": "Tools/Execution",
        "detail": "Chooses who runs a chat's tool calls — this pipe's own executor, or Open WebUI itself.\n\nThis is the workspace default, but it's also a per-user setting, so anyone can switch their own chats to the other backend regardless of this default. Most deployments keep `Pipeline`.\n\n- `Pipeline`: the pipe drives the whole tool loop, applying its own batching, concurrency limits, circuit breakers, timeouts, and a bounded call-loop cap.\n- `Open-WebUI`: the pipe hands tool calls back so Open WebUI runs them and shows its native tool UI; the pipe's batching, timeouts, loop cap, and its `Keep tool results across turns` persistence all stand down, since Open WebUI then drives each round."
    },
    "TOOL_IDLE_TIMEOUT_SECONDS": {
        "title": "Tool queue idle timeout",
        "group": "Tools/Execution",
        "detail": "How long the pipe's tool queue may sit idle between tool executions before pending work is cancelled.\n\nApplies to `Pipeline` tool execution set via `Tool execution location`. The default `null` means unlimited idle time: workers wait indefinitely, so intermittent or slow tool use never fails. Set an integer of `1` or more to cap that wait; if no tool call or result arrives in time, waiting tools are cancelled with an idle-timeout error.\n\n**Tip:** Rarely needs changing; leave at `null` unless idle tool workers must be reclaimed on a fixed deadline."
    },
    "TOOL_OUTPUT_RETENTION_TURNS": {
        "title": "Tool output history depth",
        "group": "Tools/Execution",
        "detail": "How many of the most recent conversation turns keep their full tool results before older ones are shortened to save tokens.\n\nThere is no upper bound; the most recent turns keep their tool outputs verbatim, and older long results collapse to a short opening-and-closing excerpt. A turn runs from one user message through the assistant and tool replies until the next user message. Set to `0` to disable pruning and keep every tool output in full. Only sizeable, persisted results are trimmed, so this takes effect only when `Keep tool results across turns` is enabled.\n\n**Tip:** Lower it (for example `3`) to cut token cost on long, tool-heavy conversations."
    },
    "TOOL_SHUTDOWN_TIMEOUT_SECONDS": {
        "title": "Tool shutdown grace period",
        "group": "Tools/Execution",
        "detail": "How long request cleanup lets a request's tool workers finish before force-cancelling them.\n\nAccepts any value from `0` upward; the default `10` lets in-flight tool calls finish gracefully at the end of each request, while `0` skips the wait and cancels the workers immediately. Any workers still running are always cancelled once the limit passes.\n\n**Tip:** Rarely needs changing; leave the default `10` unless slow tools are being force-cancelled during request cleanup, then raise it."
    },
    "TOOL_TIMEOUT_SECONDS": {
        "title": "Tool call timeout",
        "group": "Tools/Execution",
        "detail": "How long the pipe waits for one tool call to finish before it gives up, cancels it, and reports the failure to the model.\n\nThe generous default keeps slow-but-normal tools — web requests, file parsing — from being cut off mid-run. A timed-out call is not retried, so it fails fast: lower this to free stuck worker slots sooner, or raise it for genuinely slow integrations. Takes effect only when `Tool execution location` is `Pipeline`, where the pipe runs the tools; `Open-WebUI` mode hands execution off and ignores it.\n\n**Tip:** The batch-wide `Tool batch timeout` is never allowed below this, so raising this past it lifts the batch window to match."
    },
    "UPDATE_MODEL_CAPABILITIES": {
        "title": "Sync model capabilities",
        "group": "Models & Catalog/Model Metadata",
        "detail": "Keeps each model's Open WebUI capability checkboxes — vision, file input, web search, and the like — in step with what OpenRouter reports for it.\n\nAt each catalog refresh the reported flags replace those checkboxes, so a box an admin ticked or cleared by hand reverts to OpenRouter's value next time; capabilities OpenRouter doesn't report are left untouched. On by default. To freeze one model's checkboxes while the others keep updating, add `disable_capability_updates` to that model's advanced parameters, or `disable_model_metadata_sync` to opt it out of every sync. Its siblings `Sync model descriptions` and `Sync model icons` cover the other card fields."
    },
    "UPDATE_MODEL_DESCRIPTIONS": {
        "title": "Sync model descriptions",
        "group": "Models & Catalog/Model Metadata",
        "detail": "Fills each model's Open WebUI description with the summary OpenRouter publishes for it.\n\nOff by default, and the only one of these syncs that is — `Sync model capabilities` and `Sync model icons` both start on. With it on, each catalog refresh overwrites the stored description, so hand-written text is replaced; leave it off to keep operator-curated wording. Even while on, `disable_description_updates` on a model's advanced parameters preserves that model's text, and `disable_model_metadata_sync` opts the model out of all sync."
    },
    "UPDATE_MODEL_IMAGES": {
        "title": "Sync model icons",
        "group": "Models & Catalog/Model Metadata",
        "detail": "Gives each model the profile icon OpenRouter publishes for it, shown on the model's Open WebUI card.\n\nIcons come from OpenRouter's catalog — falling back to the model maker's logo — and refresh as the catalog changes, replacing the stored icon whenever OpenRouter's differs, including one an admin set by hand. On by default. Because it downloads icon image data from OpenRouter's catalog and external maker hosts, it needs outbound network access and is worth disabling in locked-down deployments. To keep a hand-picked icon while the rest sync, add `disable_image_updates` — or `disable_model_metadata_sync` — to that model's advanced parameters."
    },
    "USAGE_STATUS_ICON_SET": {
        "title": "Usage summary icons",
        "group": "Usage & Status/Status Display",
        "detail": "Defines the comma-separated glyphs the final usage status line uses for each field, applied only when that line renders in icon mode.\n\nTakes effect only when `Usage summary label style` is set to `icons`; in text mode the value is ignored. The entries are positional: the first seven map, in fixed order, to time, cost, total tokens, input, output, cached, and reasoning, and any beyond the seventh are dropped. Each entry is trimmed of surrounding spaces. Change it to substitute glyphs a specific theme or font renders cleanly; clearing it restores every default.\n\n**Tip:** Because slots are positional, leave one empty (two adjacent commas) to keep that field's built-in glyph rather than shifting the rest."
    },
    "USER_PROVIDER_ROUTING_MODELS": {
        "title": "User-adjustable provider routing",
        "group": "Connection & Routing/Provider Routing",
        "detail": "Lists the models that get a per-chat provider-routing switch, letting each user choose which upstream provider serves their own requests to that model.\n\nGive a comma-separated list of exact catalog ids such as `meta-llama/llama-3.2-3b-instruct, openai/gpt-4o`; left empty, no switch is added. For each listed model a switch appears in the chat Integrations menu; a user turns it on, then a gear opens their own settings — provider order, allowed or avoided providers, a sort by price, throughput, or latency, quantization, performance floors, and price ceilings. The switch starts off, and each person's choices stay private to their own chats.\n\nUnlike `Admin-enforced provider routing`, these preferences are optional and user-owned rather than imposed; listing a model in both places gives each user an admin-seeded default they can still change or switch off. A provider order a user sets here also overrides the soft provider affinity from `Pin conversation to one provider`.\n\n**Warning:** Each slug must exactly match an OpenRouter catalog id, or no switch is created for that model."
    },
    "USE_MODEL_MAX_OUTPUT_TOKENS": {
        "title": "Apply model output ceiling",
        "group": "Models & Catalog/Model Metadata",
        "detail": "When enabled, the pipe adds the model's provider-reported maximum output-token limit to each request that doesn't already specify one.\n\nThe value is read from the model's catalog metadata, so a model that reports no limit is left unchanged. The default `false` omits the `max_output_tokens` parameter entirely, letting the provider apply its own default cap. Enable it to have requests explicitly ask for the model's full stated output ceiling."
    },
    "VARIANT_MODELS": {
        "title": "Model routing variants",
        "group": "Models & Catalog/Catalog & Access",
        "detail": "Adds extra virtual models to the picker, each a base model paired with an OpenRouter routing tag — one comma-separated entry per variant.\n\nAn entry takes the form `base_id:tag` — for example `openai/gpt-4o:nitro,anthropic/claude-sonnet-4.5:extended` — where the tag is one of OpenRouter's routing suffixes such as `nitro`, `exacto`, `online`, or `free` and is lowercased. The generated model reuses the base model's description, icon, and capabilities, shows the capitalized tag in its name (`GPT-4o Nitro`), and sends the full `base_id:tag` to OpenRouter so it applies that routing. A case-sensitive preset form, `base_id@preset/slug`, is also accepted for OpenRouter presets and shows as `Preset: <slug>`.\n\n**Warning:** An entry whose base model is absent from the catalog is skipped and never appears, so a mistyped or unavailable base silently yields no variant."
    },
    "VIDEO_DOWNLOAD_CHUNK_SIZE": {
        "title": "Video download buffer size",
        "group": "Files & Media/Video Generation",
        "detail": "How much data the pipe reads at a time while streaming a finished generated video from OpenRouter to temporary storage.\n\nAt the default, throughput and memory stay balanced. A larger chunk makes fewer read passes over a big video but holds more memory per active download; a smaller one trims memory on constrained hosts in exchange for more frequent reads. Overall download size has its own separate cap in `Maximum generated video size`.\n\n**Tip:** This rarely needs changing; adjust it only if a memory-tight host runs several video generations at once."
    },
    "VIDEO_FRAME_IMAGE_MAX_BYTES": {
        "title": "Maximum single frame size",
        "group": "Files & Media/Uploads & Limits",
        "detail": "How large a single image frame may be, in bytes, when the pipe encodes it and sends it into a video generation request.\n\nThe default covers typical source stills. Any frame whose decoded size passes this cap fails the whole video request, so too low a value turns away legitimate images. Raise it to allow larger source frames, or lower it to reject oversized uploads sooner.\n\n**Tip:** This limit applies to each frame on its own; a request with several frames can still be turned down by `Maximum combined frame size`, which limits the combined size of all frames."
    },
    "VIDEO_FRAME_IMAGE_MIME_ALLOWLIST": {
        "title": "Allowed frame image formats",
        "group": "Files & Media/Video Generation",
        "detail": "Which image MIME types may be sent as frame or reference images to OpenRouter video generation.\n\nThis is a comma-separated list; the default covers the formats video models normally accept. Each frame's detected MIME is matched case-insensitively, any `;` parameters are ignored, and a frame or reference image whose type is missing from the list fails the entire video request. Add a type only when the target video model genuinely accepts it, or drop one to reject a format a provider mishandles.\n\n**Warning:** An empty value permits no formats at all, so every attached frame is rejected and image-seeded video requests fail."
    },
    "VIDEO_FRAME_TOTAL_MAX_BYTES": {
        "title": "Maximum combined frame size",
        "group": "Files & Media/Uploads & Limits",
        "detail": "How large the combined decoded size of all image frames in a single video generation request may reach, in bytes.\n\nThe pipe adds up each frame's decoded size and, once the total passes this cap, rejects the whole request before it is dispatched. The default holds several high-resolution frames. Because this is the aggregate limit, frames that each clear `Maximum single frame size` can still trip it collectively.\n\n**Tip:** Raise it only for models that take many or large frames; the default suits most single- or dual-frame requests."
    },
    "VIDEO_INITIAL_POLL_DELAY_SECONDS": {
        "title": "Initial poll delay",
        "group": "Files & Media/Video Generation",
        "detail": "How long the pipe waits after submitting a video generation job before its first status poll, in seconds.\n\nIt applies once per job, ahead of that first status check only; every poll after that is spaced by `Base poll interval`. The default gives the job a brief head start so the opening poll isn't wasted on a video that has barely started. Set `0.0` to poll immediately after submission; higher values push back the earliest moment a finished video is detected.\n\n**Tip:** This rarely needs changing; shorten it only if the first poll routinely reports a not-ready job."
    },
    "VIDEO_INTENT_CONFIRM_MODE": {
        "title": "Reuse confirmation footer",
        "group": "Files & Media/Video Intent",
        "detail": "When to show the disclosure footer — a small thumbnail of the earlier video or image the classifier reused — beneath a generated video.\n\nIt never appears for a plain text-to-video request with no reused frames, and only while `Enable video intent classifier` is on. A site-wide default users can override for their own chats on each video model.\n\n- `always` — show it on every generation the classifier adds a source frame to, including a lone attached image.\n- `on_reference` (default) — show it only when a prior video's frame is reused, or more than one frame is combined; a single freshly attached image on its own does not trigger it.\n- `low_confidence` — show it only when the classifier marks its own choice low-confidence, so the user can sanity-check the guess.\n- `never` — suppress the footer entirely."
    },
    "VIDEO_INTENT_ENABLED": {
        "title": "Enable video intent classifier",
        "group": "Files & Media/Video Intent",
        "detail": "Turns on a classifier that reads recent chat turns and any attachments before each video request, then decides what the new video should reference.\n\nWith it on, a follow-up like \"make it black\" can reuse a frame from the earlier video, adopt an attached image, or start fresh — and the classifier asks one clarifying question when intent is genuinely ambiguous. Off, only the latest user message reaches the video model: no cross-turn context, no questions, no frame reuse. This is the master switch for the Video Intent group — the task model, timeout, clarification cap, confirmation footer, and per-chat and per-day cost caps all sit idle while it is off. Users can override this and three related settings for their own chats on each video model. On by default; turn it off for literal, one-shot requests."
    },
    "VIDEO_INTENT_FRAME_EXTRACTION_INDEX": {
        "title": "Reused frame position",
        "group": "Files & Media/Video Intent",
        "detail": "Which frame — `first` or `last` — the pipe substitutes when the classifier asks for a moment past an earlier video's end.\n\nThis is only the overshoot fallback: when a continuation reuses a prior clip and the requested timestamp runs longer than that clip, the pipe grabs this frame instead and notes the swap in the disclosure footer. It does not change extraction when the requested moment is in range.\n\n- `last` — the earlier video's final frame, so the new clip resumes where the old one stopped (default).\n- `first` — its opening frame, restarting the scene from the beginning.\n\nA site-wide default users can override for their own chats on each video model."
    },
    "VIDEO_INTENT_LOG_DECISIONS": {
        "title": "Log intent decisions",
        "group": "Files & Media/Video Intent",
        "detail": "Promotes the classifier's per-turn decision record from `DEBUG` up to `INFO`, so it appears in the pipe's normal shared logs.\n\nThe record is always built and written — this only raises its level — so at `INFO` it lands in the application logs any operator or log shipper can read. Each line is derived classification metadata: the chosen intent, confidence, classifier-detected language, frame counts, latency, and fallback/failure flags, plus a chat id that is SHA-256 hashed (stable, so still correlatable within a chat) rather than raw. It deliberately omits the user's verbatim prompt and the model's free-text reasoning, and turns where the classifier is bypassed log nothing at any level.\n\n**Tip:** Leave it off for normal running; enable it briefly only while diagnosing misclassification."
    },
    "VIDEO_INTENT_MAX_CALLS_PER_CHAT": {
        "title": "Intent calls per chat",
        "group": "Files & Media/Video Intent",
        "detail": "A per-chat ceiling on how many times the classifier's extra, billable Task Model call may run in one conversation.\n\nThe default `0` means unlimited. Set a positive number and, once a chat hits it, further video requests skip classification and generate from the latest message alone — the paid render still proceeds. It works alongside `Daily intent calls per user`; whichever ceiling is reached first applies. Each run bills at the Task Model's rate, so check its OpenRouter pricing.\n\n**Warning:** The tally is per worker process in memory and resets on restart, so a chat spread across workers can exceed the cap."
    },
    "VIDEO_INTENT_MAX_CALLS_PER_USER_DAY": {
        "title": "Daily intent calls per user",
        "group": "Files & Media/Video Intent",
        "detail": "A per-user daily ceiling on classifier calls, so one person cannot run up unlimited extra Task Model spend.\n\nThe default `0` means unlimited. A positive value caps how many times one user triggers the classifier per UTC day; beyond it, that user's video requests skip classification and generate from the latest message alone, while the paid render still proceeds. It pairs with `Intent calls per chat`; the first ceiling reached applies. Each run bills at the Task Model's rate — check its OpenRouter pricing.\n\n**Warning:** The daily tally is per worker process and resets at UTC midnight or on restart, so behind several workers the real limit is roughly this value times the worker count."
    },
    "VIDEO_INTENT_MAX_CLARIFICATIONS": {
        "title": "Maximum clarifying questions",
        "group": "Files & Media/Video Intent",
        "detail": "How many clarifying questions in a row the classifier may ask before it commits to its best guess.\n\nBy default it allows a single question per chat, then generation proceeds on the classifier's own interpretation. `0` turns the clarification loop off entirely, so ambiguous requests never pause to ask. Raise it where a quick follow-up question is worth more than a wrong, paid video. Users can override this for their own chats on each video model, and it has no effect while `Enable video intent classifier` is off."
    },
    "VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT": {
        "title": "Skip intent on empty chats",
        "group": "Files & Media/Video Intent",
        "detail": "Skips the classifier on a chat's very first turn when nothing has been uploaded for it to reference.\n\nAn opening prompt with no prior video and no attached image gives the classifier nothing to interpret, so the call would be wasted Task Model spend. Turn it off only to allow clarifying questions on context-free openers like `make it red`. On by default."
    },
    "VIDEO_INTENT_TASK_MODEL_FALLBACK": {
        "title": "Intent classifier fallback",
        "group": "Files & Media/Video Intent",
        "detail": "What the classifier does when its chosen Task Model fails to return a usable classification — stop there, or try the other one.\n\n`Intent classifier task model` picks which of the two global Task Models runs first; this valve decides the second attempt.\n\n- `none` — only the chosen Task Model is tried; if it errors, times out, or is unset, classification is skipped.\n- `other_task_model` — the pipe then tries Open WebUI's other global Task Model before giving up (default).\n\nIf every attempt fails, only the latest user message reaches the video model and the paid video generation still proceeds — the classifier never blocks a render.\n\n**Tip:** `other_task_model` helps only when the two Task Models differ; if the other is unset or identical, it behaves like `none` (duplicates are collapsed)."
    },
    "VIDEO_INTENT_TASK_MODEL_MODE": {
        "title": "Intent classifier task model",
        "group": "Files & Media/Video Intent",
        "detail": "Which of Open WebUI's two global Task Models runs the intent classifier — the one configured for external (API) models or the one for local models.\n\nThe classifier is a short structured call made before each video generation. This valve names only the primary model; `Intent classifier fallback` decides whether the other Task Model is tried when this one is unset or fails. If neither resolves, the classifier is quietly skipped and the video still generates, just without cross-turn intent analysis.\n\n- `external` — runs Open WebUI's `TASK_MODEL_EXTERNAL` setting; the default, and the safer pick when that model follows structured JSON instructions reliably.\n- `internal` — runs Open WebUI's `TASK_MODEL` setting instead.\n\n**Tip:** Set the chosen model under Open WebUI's admin Task Model settings; each classifier run is a billable call at that model's OpenRouter rate."
    },
    "VIDEO_INTENT_TIMEOUT_S": {
        "title": "Intent classifier timeout",
        "group": "Files & Media/Video Intent",
        "detail": "How long the pipe waits for the classifier's Task Model call to answer before abandoning it.\n\nThe default suits a fast task model. If the call times out or fails, the pipe degrades open — it sends only the latest user message and the paid video generation still proceeds, so the classifier never blocks a render. Set it too low and the classifier is cut off before it can reuse a prior frame or ask a clarifying question; raise it only for a slow task model.\n\n**Tip:** The limit applies per attempt, so pairing it with a fallback task model can roughly double the worst-case wait."
    },
    "VIDEO_MAX_POLL_TIME_SECONDS": {
        "title": "Video generation timeout",
        "group": "Files & Media/Video Generation",
        "detail": "How long the pipe keeps polling a video generation job, in wall-clock seconds, before giving up and recording a timeout.\n\nThe default suits most short clips. This is the ceiling on the whole poll loop rather than how often status is checked — that cadence is set separately by `Base poll interval` and `Maximum poll interval`. Raise it for longer or high-resolution renders that legitimately take minutes, or lower it to abandon stuck jobs sooner; there is no unlimited option.\n\n**Tip:** A job holds one generation slot for its entire wait, so a high value combined with a stalled job can starve others up to `Maximum concurrent video jobs`."
    },
    "VIDEO_MAX_SIZE_MB": {
        "title": "Maximum input video size",
        "group": "Files & Media/Uploads & Limits",
        "detail": "How large an inbound video the pipe will take in, in megabytes, before turning it away.\n\nIt bites in two places: a video pasted inline as a base64 `data:` URL is rejected with a `Video too large` error when its estimated decoded size goes over, and a stored Open WebUI video the pipe reads back to extract frames is skipped when it exceeds the cap. Plain `http(s)` and YouTube video links are forwarded to the provider unchanged and are never weighed against this limit.\n\n**Tip:** A higher value mainly lets larger base64 videos decode into process memory, so keep it near the size of clips actually in use."
    },
    "VIDEO_OUTPUT_MIME_ALLOWLIST": {
        "title": "Allowed video output formats",
        "group": "Files & Media/Video Generation",
        "detail": "Which MIME types the pipe accepts for a generated video downloaded from OpenRouter, checked once the download finishes.\n\nThis is a comma-separated list (for example `video/mp4,video/webm,video/quicktime`), with entries matched case-insensitively; the default covers the standard web-playable formats. The type is read from the response `Content-Type` header, falling back to magic-byte sniffing of the file's first bytes only when that header is absent or generic (`application/octet-stream`). If the result isn't listed, the video is discarded and generation fails with a download error.\n\n**Warning:** An empty value matches nothing and rejects every generated video; at least one type such as `video/mp4` must remain listed to keep video generation working."
    },
    "VIDEO_POLL_BACKOFF_FACTOR": {
        "title": "Poll backoff factor",
        "group": "Files & Media/Video Generation",
        "detail": "How much longer each status poll of a running OpenRouter video job waits than the one before it.\n\nThe default grows the interval modestly each poll, ramping from `Base poll interval` toward the ceiling `Maximum poll interval`. A value of `1.0` disables backoff and holds a constant interval, which gives the most frequent polling and the heaviest load on OpenRouter's status endpoint. The highest values reach the cap within one or two polls, cutting requests but noticing a finished job slightly later.\n\n**Tip:** This rarely needs changing; adjust it only if status polling is too heavy or finished videos take too long to appear."
    },
    "VIDEO_POLL_INTERVAL_MAX_SECONDS": {
        "title": "Maximum poll interval",
        "group": "Files & Media/Video Generation",
        "detail": "The longest the pipe waits between status checks on an OpenRouter video generation job, in seconds.\n\nPolling begins at the interval from `Base poll interval` and grows by `Poll backoff factor` after each check, but never past this ceiling. The default keeps late-stage checks reasonably spaced without hammering the API. Lower it for fresher progress on short clips at the cost of more requests, or raise it to cut API traffic on long renders.\n\n**Tip:** Keep this at or above `Base poll interval`; a lower value forces flat polling at this interval and disables the backoff growth."
    },
    "VIDEO_POLL_INTERVAL_SECONDS": {
        "title": "Base poll interval",
        "group": "Files & Media/Video Generation",
        "detail": "How long the pipe first waits between status checks on an in-progress OpenRouter video generation job, in seconds.\n\nThe default polls a fresh job promptly without flooding the API. After each poll the wait is multiplied by `Poll backoff factor` and capped by `Maximum poll interval`, so this value governs only the first few intervals before backoff takes over. Lower it to detect completion sooner at the cost of more status requests, or raise it to cut polling traffic on long renders.\n\n**Tip:** A value above `Maximum poll interval` is clamped to that ceiling, so keep it at or below the maximum interval."
    },
    "VIDEO_STATUS_POLL_MAX_ERRORS": {
        "title": "Poll error tolerance",
        "group": "Files & Media/Video Generation",
        "detail": "How many status-check failures in a row a video generation job tolerates before it is abandoned and reported as failed.\n\nOnly back-to-back polling errors count, and any successful status poll resets the tally to zero, so scattered hiccups never add up. The default rides out brief transient failures while still giving up on a genuinely stuck job; a value of `1` fails on the very first polling error. Each generation is also capped overall by `Video generation timeout`, which ends it on elapsed time regardless of this count.\n\n**Tip:** This rarely needs changing; raise it only if OpenRouter's status endpoint is intermittently returning errors on otherwise-healthy jobs."
    },
    "ZDR_ENFORCE": {
        "title": "Enforce ZDR routing",
        "group": "Models & Catalog/ZDR",
        "detail": "Forces every chat to route only to Zero Data Retention endpoints and refuses any model that has none.\n\nEach request carries `provider.zdr=true`, so OpenRouter itself keeps it on a no-retention endpoint, and a model with no ZDR endpoint is stopped with the `Blocked model message` before anything is sent. Enforcement is site-wide and overrides per-user choice: `Allow user ZDR opt-in` and each user's `Request ZDR` toggle are ignored while this is on.\n\n**Warning:** If OpenRouter's ZDR list can't be loaded, the request is rejected rather than sent unverified — unlike `Show only ZDR models`, this fails closed so an outage can't silently bypass ZDR."
    },
    "ZDR_MODELS_ONLY": {
        "title": "Show only ZDR models",
        "group": "Models & Catalog/ZDR",
        "detail": "Trims the model picker to only OpenRouter's Zero Data Retention models — those served by an endpoint that keeps no copy of the prompt or reply.\n\nLike any catalog filter, a hidden model also can't be reached directly: a request to one returns the `Blocked model message`. What it does not do is ask OpenRouter to route with ZDR — it curates the list but sends no `provider.zdr=true`, trusting the published ZDR list rather than enforcing per request. Use `Enforce ZDR routing` for that guarantee.\n\n**Warning:** If OpenRouter's ZDR list can't be loaded, filtering is skipped and every model stays visible and usable — it fails open, the opposite of `Enforce ZDR routing`."
    },
    "PIPE_DASHBOARD_ENABLE": {
        "title": "Enable Pipe Dashboard",
        "group": "Plugins/Pipe Dashboard",
        "detail": "Adds a virtual model named `Pipe Dashboard` to the model selector; opening a chat with it and sending `dashboard` opens a live multi-tab admin console — diagnostics plus this editable configuration panel — while `help` lists the commands.\n\nOff by default, that model stays unregistered and the console is unreachable. It also needs the plugin framework running: with `Enable plugin system` off, this does nothing. Who can open it is governed by Open WebUI's per-model access grants — a read grant views the dashboard, a write grant runs its operator actions — so treat enabling it as exposing an admin surface and grant access deliberately."
    },
    "PIPE_DASHBOARD_USAGE_COLLECT": {
        "title": "Collect usage records",
        "group": "Plugins/Pipe Dashboard",
        "detail": "When enabled, the pipe writes one metadata record per completed request into a dedicated `dashboard_`-prefixed database table, so the dashboard's `Usage` tab can chart activity over time.\n\nEach row identifies the acting user by both account id and display name (or email), alongside the chat and session ids, the model, a chat-vs-task marker, status, duration, token counts (input, output, reasoning, cached), tool success and failure counts, retries, and cost — but never the prompt or reply text. That makes it an identifiable per-user, per-chat audit trail retained on the server for `Usage record retention (days)`.\n\nOff by default, it writes nothing, so the tab has no history. Turning it on takes effect live, with no restart."
    },
    "PIPE_DASHBOARD_USAGE_RETENTION_DAYS": {
        "title": "Usage record retention (days)",
        "group": "Plugins/Pipe Dashboard",
        "detail": "Bounds how long each collected usage record lives before a background task purges it — and with it, how long the identifiable per-request history lingers on the server.\n\nApplies only while `Collect usage records` is on. A repeating purge (running a few times an hour, jittered, and coordinated across workers) drops every record older than the window, so a change here takes effect at the next pass rather than instantly. Shorten it to retain less identifiable data; lengthen it for deeper trend history at the cost of a larger table. The default keeps about a month."
    },
    "PIPE_DASHBOARD_UPDATE_ENABLE": {
        "title": "Enable the Update tab",
        "group": "Plugins/Pipe Dashboard",
        "detail": "Turns on the dashboard's `Update` tab: an admin sees the installed version next to the latest tagged GitHub release (with the release page's own changelog), and can self-update the pipe in one click — the new bundle is size-capped, sha256-verified against the release digest, exec-validated through Open WebUI's own loader, and only then written into the function row, with the previous version snapshotted for one-click restore.\n\nOff, the tab reports \"disabled by admin\" and every update action — including `Auto-update` — fails closed on the server, so nothing can change the installed code through this surface. The gate is server-side; hiding the tab is just the courtesy part.\n\nApply/restore/delete additionally require the acting account to hold the `admin` role — a write grant on the dashboard model alone is not enough for code-changing actions."
    },
    "PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP": {
        "title": "Update snapshots to keep",
        "group": "Plugins/Pipe Dashboard",
        "detail": "How many previous-version snapshots the updater keeps in Open WebUI's Files storage — each one is the complete function source taken immediately before an update or restore, and each is the rollback point behind the `Previous versions` table's Restore button.\n\nSnapshots live as a fixed set of dedicated file records whose metadata rides on the file itself, so they survive anything done to the function in the admin editor — hand-pastes and saves there can never erase the rollback list. When a new snapshot would exceed the limit, the oldest is pruned: its file record is removed first, then its stored blob (a failed blob delete only logs a warning — a stray blob has no record and is harmless). Retries don't burn slots: an attempt whose content matches any retained snapshot reuses that snapshot instead of writing a new one.\n\nSnapshots are public release code only — valve values live encrypted elsewhere and are never part of a snapshot."
    },
    "PIPE_DASHBOARD_UPDATE_REPO": {
        "title": "Update source repo (owner/repo)",
        "group": "Plugins/Pipe Dashboard",
        "detail": "Which GitHub repository the updater tracks for tagged releases — the default is the upstream project; point it at your fork (`owner/repo` on github.com) to self-update from your own builds. Forks inherit the release workflow, so the same bundle assets, automatic sha256 digests, and generated changelog all keep working; a fork that strips its CI simply shows an empty changelog and, if assets are missing, a \"no matching asset\" state.\n\nEvery check, changelog, and download is built server-side from this valve — the browser never supplies a repo or URL. A malformed value shows as a `bad repo` error in the tab; a well-formed repo that doesn't exist (renamed, private, or without releases) shows as `repo not found`. Neither can change the installed code.\n\nThe full validation gauntlet applies regardless of repo: the downloaded bundle must match the release digest and carry this pipe's own id before it is ever executed."
    },
    "PIPE_DASHBOARD_UPDATE_AUTO": {
        "title": "Auto-update",
        "group": "Plugins/Pipe Dashboard",
        "detail": "Lets the pipe update itself. In multi-worker deployments the workers elect ONE update leader through Open WebUI's own Redis lock: only the leader talks to GitHub (about every six hours), while every other worker just probes the lease hourly and takes over if the leader dies or restarts — worker count never multiplies GitHub traffic. Single-worker installs (no Redis) skip the election and check directly. An eligible release — one older than the `Auto-update quarantine (hours)` window — is applied through exactly the manual pipeline: snapshot first, digest and loader validation, same audit trail (actor `auto`).\n\nIt runs headless: as long as this and `Enable the Update tab` are on, updates continue even while the Pipe Dashboard model itself is switched off. Valve flips are re-read from the database each cycle, so turning it off takes effect within one cycle, no restart. GitHub rate limits and network failures back off and retry — a busy egress IP delays an update but never loses one.\n\nA release that fails validation or won't load is paused on that worker until a restart, a newer release, or a successful manual apply — the tab shows the pause and why. Off by default: unattended code swaps are a deliberate opt-in."
    },
    "PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS": {
        "title": "Auto-update quarantine (hours)",
        "group": "Plugins/Pipe Dashboard",
        "detail": "How old a release must be before `Auto-update` will touch it. The default 168 (7 days) exists so a bad release that gets yanked shortly after publishing never reaches a single auto-updater: the check re-resolves the latest release every cycle, so a deleted release vanishes from consideration and a fixed follow-up (say v2.7.1 over a bad v2.7.0) restarts the clock and supersedes the bad version entirely.\n\n`0` disables the quarantine — every new release is eligible the moment it is published. Raise the window if you'd rather trail the bleeding edge; the tab's `Auto-update` line always shows when the pending release becomes eligible.\n\nManual updates ignore this valve entirely — an admin clicking Update installs immediately."
    }
}
