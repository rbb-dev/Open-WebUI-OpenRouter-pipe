# Release notes pending

## Behaviour changes

- **Pipe dashboard, viewer payload** — the live dashboard payload no longer carries the data-dir path, so a viewer
  holding only a read grant no longer learns the server's filesystem layout from the System tab. The key was published
  by the system collector as `system.disk_path`, and every key in a payload reaches every socket in the viewers room.
  The Disk card is unchanged — it renders from `disk_free` and `disk_total`, both of which still ship — and the
  operator's own log still names the volume, latched to once per 300 s. An operator who wants the path already has it
  on the admin-only Config tab's `Session log directory` row.

- **Temporary chats, marker records** — three streaming records that named a temporary chat's browser socket id now
  name it `<not retained>`, as every other operator-visible record already did. They fired on the ordinary path for
  a temporary chat that reasons or calls a tool: two when the turn's committed artifact rows are addressed by hidden
  marker lines in the reply (a WARNING when the content was handed back before its markers were added, a DEBUG when
  they were added), and one when a cancelled turn leaves committed rows with no marker addressing them. Because
  `SessionLogger.get_logger` pins the package logger at DEBUG for the session archive, the id reached the
  process-lifetime log buffer — the dashboard's log view — on every such turn. The levels, the marker lists, the row
  counts and the reason strings are unchanged, so the records still say which turn lost rows and how many; only the
  chat id is gone. A saved or `channel:` chat's id is still named in full.

- **Provider errors** — a provider error body carrying a character no text encoding can hold (an unpaired
  surrogate, which `json.loads` accepts and `str.encode("utf-8")` refuses) now reaches the error card with
  that one character shown as `�`, the replacement character. The same body used to end the turn in an HTTP
  500 rather than a card, and to lose the operator's WARNING record and the turn's whole session-log archive,
  because the archive writer's zip write failed on it and lost `meta.json`, `logs.txt` and `logs.jsonl`
  together. Everything else the provider sent is unchanged and still reaches the card verbatim.
- **Tool pictures** — a faulted SSRF address pool now costs the picture instead of the reply. The
  address checks run on a private bounded thread pool, and that pool can refuse to take work: it is
  shut down when the worker closes, and re-made (with the old one shut down) whenever
  `MAX_CONCURRENT_REQUESTS` changes — so a dashboard save mid-turn can leave the next check unable
  to start. The tool-picture arm was the only forwarding path that let that fault escape: the
  person's own picture, file and video links each degrade to a refusal of their own, and the tool
  link instead took the whole turn down with an `Unexpected Error` card. One MCP tool's image link
  was enough to lose the reply. The check now costs that picture and nothing else: the turn is sent
  with the round's text, the link is not forwarded, and the person sees
  `Images: skipped 1 (could not be checked in time, so it was not sent).` — the same sentence the
  arm already used for a check that ran out of budget, because a pool that cannot run has reached
  no verdict, which is what that sentence already meant. The reader itself is unchanged and still
  raises, so a teardown race is never turned into a security refusal on any other path.

- **Temporary chats** — the reply hold no longer keeps a temporary chat's socket id. `ReplyMemory` keys a reply's
  held rounds and thinking on `(user id, chat id, message id)`, and for a temporary chat that chat-id slot is now
  blank, which is the shape the hand-back budget beside it has always used. Nothing observable changes for a reply:
  the same rounds are held for the same 15 minutes, under the same 64 MiB pool ceiling, for the same user, and are
  released at the same point — a reply still reads its own rows back by its own `message id`, which is the slot that
  was left alone and the only thing telling two replies of one chat apart. A saved chat's and a `channel:` chat's
  keys are untouched. What changes is who can read the id: a temporary chat's chat id is a credential Open WebUI's
  own session pool will authorise a request with, and it was the last process-lifetime structure in the pipe still
  holding one.

- **Video generation, machine callers** — a video turn that fails *before* OpenRouter answers the submission now
  reaches a caller with no chat as an HTTP error instead of a `200` with a Markdown card in it. A rejected job
  leaves with the status the pipe resolved on the status line and the same number in `error.code` (`502` when a
  proxy rewrote the `/videos` body); a fault the pipe owns leaves as `500` carrying `Video generation failed.`,
  with no Python class name in the body and the full message and traceback at ERROR in the session log as before.
  The gate is the same one the chat leg uses — no truthy `chat_id` **and** `message_id`, `stream: false`, never on
  an Anthropic Messages path — so a chat keeps its card, a streamed turn keeps its card in the stream chunks, and a
  successful job is unchanged. A plain API client that submits a video model with no chat can now branch on
  `status_code` the way it already could for a chat completion.

- **Provider routing panels** — a model's provider-routing filter is no longer detached from it when two model-list
  refreshes overlap. The pass that could not install the panel reported its verdict on a value shared with the pass that
  overlapped it, so a refusal to write could read as an answer and release the panel; the verdict now travels with the
  answer the pass obtained, and a pass that never obtained one leaves every model's filter list exactly as it was.
- **Media on the `/chat/completions` leg** — a turn resolved to the chat endpoint — by `DEFAULT_LLM_ENDPOINT=chat_completions`, by a model in `FORCE_CHAT_COMPLETIONS_MODELS`, or by the `AUTO_FALLBACK_CHAT_COMPLETIONS` fallback — now judges every media link its converter copies, by the same rule the `/responses` leg applies at ingress: the transport (`http://` only with `ALLOW_INSECURE_HTTP` **and** an allowlisted host), the scheme, a link naming this Open WebUI's own `/api/v1/files/` endpoint, and the inline size bound. Such a link used to be copied verbatim onto the wire, and the cleartext valve was never asked about it on this leg. A refused block is dropped from the turn and named on it as `[An attached item was not sent: …]`, in the request side's own words, beside whatever else survived — except an `input_file` that also carries a `file_id`, which keeps that id and drops only the refused link, exactly as on the other leg. An admin who allows cleartext HTTP sees the same link forwarded as before, and a `https://` link, a `data:` URL and raw base64 in `file_data` are untouched. One asymmetry is worth naming: a `data:` video sent to a chat-leg model is now held to the same size bound as on the other leg (`BASE64_MAX_SIZE_MB`, not `VIDEO_MAX_SIZE_MB`), so a clip over it is reported as not sent instead of forwarded.

- **Presets** — a request naming a preset in the spelling the pipe itself dispatches (`<base>@preset/<slug>`) is now served instead of being refused as blocked. The pipe publishes presets as `<base>:preset/<slug>` and dispatches them with the `@`; the model-restriction gate read the dispatched spelling as a model of its own, so an id copied out of a request log or a response body came back as the `Blocked model message`. The published row was already enforced under the picker spelling, so nothing new is admitted — the two spellings now name one model. A preset turn also stops being billed to the 128 000-token fallback and picks up its base model's real context window, so a tool result the base's window can hold is no longer trimmed away and one it cannot is no longer shipped whole.
- **image generation, a rewritten response body** — when something in front of OpenRouter answers the image endpoint with an error page or a WAF challenge instead of an OpenRouter response, the failure card now shows that text as a quoted code block instead of as ordinary message text. The provider's words are still on the card in full — nothing is shortened, rewritten or hidden — but they are contained, so a beacon embedded in the page no longer makes the reader's browser fetch it and a phishing link in it is no longer a live link. The two facts the card leads with, the endpoint that answered and the upstream `Content-Type`, are unchanged and remain ordinary prose. The fence is carried by the value rather than by a placeholder, so it holds on an installation that has never opened a template, and it is sized past the payload's own backtick run, so a page carrying its own fence cannot escape one. This affects every caller of a picture-only image model, on a surface that is on by default, and it is the same rule the text legs' error cards already applied.
- **Session log** — with `SESSION_LOG_STORE_ENABLED` off, which is the shipped default, the pipe no longer builds the
  DEBUG request and response payloads at all. They were redacted and serialised on every request and every
  non-delta SSE event, at any `LOG_LEVEL`, and then dropped by the console; the work is now skipped rather than
  done. What storage off now stops is those payload records, and nothing else: the in-memory session log is still
  filled with every other record the request makes, at any `LOG_LEVEL`, so the dashboard's **Log buffers (RAM)**
  counters count those records, not 0 — they are a live count of what the pipe is holding right now, not a
  session-logging indicator. `Log verbosity level` set to `DEBUG` still prints those payloads to the console,
  which never read the archive valve. With the valve on, nothing changes: the payloads are built and kept, as
  before.
- **Artifact retention sweep** — a cleanup pass whose Redis cache purge could not finish no longer deletes anything. Both purge
  arms returned on the first `delete` fault without telling the caller, so the sweep went straight on to the delete: rows the
  operator had been told were purged were gone while the cache entries naming them stayed readable, for a temporary chat under a
  key that embeds its socket id, until the cache TTL expired. The arms now report whether they completed and one gate acts on it,
  so a faulted cycle logs one warning naming the deferral, deletes no rows at all - retention and temporary-chat alike, since one
  arm's unread pages are the other's evidence - and both the rows and their entries go on the next interval instead. Nothing
  changes on a healthy pass, and a deployment with the Redis cache valve off still sweeps: no client means nothing to invalidate,
  which counts as complete.
- **Filter rows, two pipe copies** — in a deployment with two pipe copies, each copy now reads only its own filter rows: the Web Tools filter's stored per-user toggles and cost ceiling for internal Fusion's panel members, and the image-generation model. A row stamped with another copy's install record is no longer read as configuration or executed, whatever id it sits on. The isolation the documentation already promised, and the second copy gains its own image-generation filter row after the next model refresh, starting from the pipe's own drawing-model default rather than the first copy's admin's choice.

- **Timing profiler** — `ENABLE_TIMING_LOG` now writes only its file. The pipe used to keep a second,
  per-request in-memory copy of every timing event alongside `timing.jsonl`, bounded by a request count and a
  per-record character cap, and three help texts and the storage guide described that copy as something a
  running request could be read from. No production code ever read it: its two accessors
  (`get_timing_events`, `format_timing_jsonl`) had no caller outside the test suite, so the copy grew with every
  event the valve recorded and was bounded only by an eviction cap. That copy and both accessors are gone, along
  with the `clear_timing_events` release that existed to trim it. Nothing observable changes for anyone reading
  `timing.jsonl`: the file the valve exists to produce is byte-for-byte what it was, and with the valve off
  nothing was buffered either way.
- **Open-WebUI tool mode** — a call Open WebUI runs now shows its card from the moment the model names the tool, and the dashboard's live view now shows the running tool on those turns (it stayed empty before).
- **Pipe dashboard, deleted function row** — deleting the pipe's function row now releases the live dashboard on the worker that served the DELETE and lets
  the pipe finish its in-flight requests and close, instead of holding it, its session-log threads and its storage handle until a restart. The stand-down
  happens whether or not `PIPE_DASHBOARD_ENABLE` is on, and it is per worker: on a multi-worker deployment the other workers release their own generation at
  their next hot reload, exactly as Open WebUI keeps its own per-worker function cache. The action route and the socket gate still refuse once the row is
  gone; nothing about the authorization answer changes.
- **`/chat/completions`** — a streaming body the provider accepted and then lost is no longer re-sent.
  The `/responses` leg already stopped re-POSTing one; the fallback leg did not, so the same shape cost
  `1 + TRANSIENT_RETRY_MAX_ATTEMPTS` POSTs for a request OpenRouter had already accepted and begun
  generating for. It now costs one, whatever frames the stream had published — visible or not — and the
  fault crosses as `AcceptedResponseLostBody`, which every handler above the transport still reads as the
  connection-class fault it has always seen. The rule is about whether the **body finished arriving**, not
  about the exception class: a body that arrived whole and delivered nothing a reader could use is still
  retried on the full budget and keeps its own class, and a connection that never reached the provider is
  still retried. `DEFAULT_LLM_ENDPOINT` defaults to `responses`, so this is the fallback leg — reached by
  `AUTO_FALLBACK_CHAT_COMPLETIONS`, by `FORCE_CHAT_COMPLETIONS_MODELS`, and by any deployment whose
  `DEFAULT_LLM_ENDPOINT` is `chat_completions` — rather than the one most turns take.
- **Streaming** — a frame the drop valve discarded no longer closes the retry barrier. With
  `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` above `0`, a full buffer used to discard the delta the pipe had just
  published, and the streaming loop raised the retry barrier anyway — so a recoverable rejection right
  after it ended the turn on a card, and the discarded text was lost with nothing on the wire to say so.
  A frame the reader never received has not been published: the emit and the barrier now go through one
  helper that reads the emitter's own delivery verdict, so such a turn is handed back and retried. The
  direction that matters is unchanged — a frame the buffer *did* take still ends the turn, because a retry
  would splice two attempts into the reader's message — and with streaming off `body.stream` still governs
  the leg, since the non-streaming wrapper drops those frames before they leave the pipe.
- **Session log** — a turn the pipe refused before sending is now archived as `error` instead of `complete`.
  The seven pre-send refusals (Zero Data Retention routing in force with the model off the roster, the ZDR
  endpoint list unreadable, an endpoint-override conflict from a preset or from Direct Uploads, a Fusion model
  forced to `/chat/completions`, an attachment that would not load, and the operator's model restrictions)
  return a card instead of raising, so the job's future resolved cleanly and the archive recorded a turn whose
  chat holds nothing but a refusal card as a clean completion with no cause. The archived `reason` is the
  pipe's own sentence naming the control that refused it, through the operator's own valve labels — not the
  rendered card, which carries a generated error id and a timestamp that identify one message and nothing else.
  The card the person sees is unchanged, and a turn that answered, that a plugin answered, or that the person
  stopped is still archived as it was.
- **session log, per-record bound** — a record that carries a whole payload is now cut at 16,384 characters, and the
  record names the cut: `...(truncated: 1,234 characters omitted)...`. This covers all five payload sites — the
  streamed SSE event, the request headers, the request payload, the response payload and the error response — so one
  1.8 MB frame or one 200 KB reply can no longer become a single 1.8 MB record. The bound is on the serialised record
  after redaction, so a payload that is large because of many small fields is bounded as well as one large field, and
  a payload below the bound produces a record byte-identical to today's. It is the same 16,384 and the same marker
  the tool-result record already used, so there is now one number rather than two.

  This is what `SESSION_LOG_MAX_LINES`'s help text already promised and did not do; the text is now true rather than
  corrected. The cut is in the archive exactly as it is on stdout, and nothing else is cut by it: an exception block
  keeps its whole traceback, the tool-result bound is unchanged, and the error object a card is rendered from keeps
  its own values whole.

  One help text moves with it. `OPENROUTER_ERROR_TEMPLATE` said "the complete values stay on the error object and in
  the session log"; after this change no session-log record carries them for a payload above the bound (the error
  record's own excerpt is cut at 8,192), so the sentence now promises the error object alone and says where the
  session log's copy is cut.

- **artifact retention, retired tables** — a `response_items_*` table that a previous `ARTIFACT_ENCRYPTION_KEY` left behind
  is now swept on the same `ARTIFACT_CLEANUP_DAYS` window as the table in use. Nothing in the pipe addressed those tables at
  all: after a rotation every row it had written — reasoning traces, tool results, temporary-chat rows, and staged
  session-log segments carrying whole request and response content — stayed in the database past every window meant to
  bound it. The retired tables get the age filter and the temporary-chat delete, but no bookkeeping deny-list: no owner
  can address a retired table, so a staged segment there has nothing behind it but this window.

  The pass is scoped by name to this pipe's own fragment, each table is reflected and swept in its own session, one
  table's failure warns and skips without touching the others, and the live table's statements, order and single
  `Retention removed` line are unchanged. Each retired table gets its own INFO line naming it and the row counts.

  **It does not run while `ARTIFACT_ENCRYPTION_KEY` is set to a value the application secret can no longer open.** On that
  arm the worker's own table name is derived from the sentinel placeholder, so its live table is a third table and every
  other sibling it can see belongs to a key it does not hold — including the table of another configuration of the same
  pipe that has simply never set a key. Nothing in the name tells those apart, so the whole pass is skipped: an unreadable
  key costs retention until the key is re-entered, rather than deleting another installation's rows. The live table is
  swept as before throughout.

- **`SERVICE_ERROR_TEMPLATE`** — an admin who has customised this template now sees it on a picture-only image
  generation whose reply a proxy, CDN or WAF rewrote. That card previously carried its own fixed sentence. The fence
  and the withheld-on-a-channel behaviour come with it, unchanged; the video leg keeps its own card shape.
- **Video and image catalogs** — a failed catalogue fetch now backs off instead of waiting out the whole
  refresh interval. Both media lists already stamped their attempt clock on every outcome; what was missing is
  that a failure and a success were indistinguishable to the gate, so a thirty-second provider blip cost the
  picker an hour of missing media rows. The first retry now costs 5 seconds rather than the full hour, and
  consecutive failures double from there — 10, 20, 40, 80, 160 — capped by `MODEL_CATALOG_REFRESH_SECONDS`
  whenever that is set below 160, which is the chat model list's own ladder. The count is kept per OpenRouter
  account and per media catalog, so one key's outage never paces another's media list, and a successful fetch
  of either resets it. A fetch that returns no models is treated as an answer rather than a fault: it stamps the
  clock exactly as before and leaves the count alone, so a proxy that is merely quiet is not paced like one that
  is down. At the shipped default the healthy path is unchanged.
- **Update tab** — an apply or restore is now refused when the pipe's source was hand-edited inside
  the same second the tab was drawn. The tab's revision was Open WebUI's whole-second
  `Function.updated_at`, so a paste in Workspace ▸ Functions that landed in that second left the
  revision exactly where it was; the update then overwrote the paste and still reported success. The
  revision the tab sends and the guard checks is now `<updated_at>:<content digest>`, so the write
  that could not move the revision cannot slip past it. A client that sends only the stamp is still
  guarded on the stamp alone, so a tab drawn by a worker on the previous version keeps working, and
  the refusal is the Update tab's existing "this view was out of date" message. The window is narrow
  and it needs an edit made in the same second as the tab's own load, so this closes a race an
  administrator had to be holding Apply for; updates are not otherwise riskier — nothing else about
  the apply path changed.
- **Provider routing panels** — every installed routing panel's code is replaced on the next model-list refresh, and
  what changed in it is entirely in the log. A stored choice the panel cannot map (an `ORDER`, `ONLY` or `IGNORE`
  dropdown value with no entry in the routing map) now warns **once** at `WARNING` and repeats at `DEBUG` on every
  later turn, naming the field and the value; before, one such choice meant three WARNINGs on every request for as
  long as it stayed open, with no throttle of any kind. The choice itself still goes out unconstrained exactly as
  before — nothing about the request changes. This entry also covers the two valve-healing warnings in the same
  generated panel, whose latch declarations were renamed to the spelling the pipe's own warn-once inventory reads, so
  that they are reset between runs instead of staying armed for the life of the process.

- **inline video and the tool budget** — a turn carrying an inline video clip on `/responses` no longer shows the
  "this conversation needs about N tokens" warning, and no longer drops a tool result, because the context budget
  now rates the clip under the name the wire gives it. The two spellings of a video block — `video_url` before the
  responses rewrite, `input_video` after it — were priced from different rows of the rate table, and only the first
  had one, so every pass that ran after the rewrite charged the clip's base64 at its literal length, some 125× the
  clip. The turn then read as hopeless and the budget stopped trimming tool output at all. Reach is occasional:
  inline `data:` clips only, and only on `/responses`; a linked clip was never affected, and neither was a turn
  without a tool result to lose. What it cost was the model answering without a tool result it had been sent.

- **Update tab after a refused write** — the dashboard now follows the rebuilt instance on every surface, so the
  Update tab, the Usage tab and the Live card stop disagreeing about which generation is live. A refused write no
  longer leaves the Live card on the retired generation's sessions, and no longer leaves Update and Usage answering
  "unavailable" once the rebuilt instance itself retires: the panel's registrations are taken back down with the
  generation that made them, instead of being pinned for the life of the worker.
- **Config tab** — a save the database refuses to write now raises a durable banner instead of a toast alone.
  The banner names the fault, carries no Reload control, and leaves your staged edits in place, so nothing the
  refusal preserved can be discarded from the tab that preserved it. It clears on the next successful save or
  the next configuration load; today's toast is unchanged.
- **Auto-update** — the pause line now names *why* a release was paused, not just the code. The Auto-update
  row carries the version it stopped on and the refusal's own sentence, versions included; the
  `(apply manually or restart to re-arm)` hint is now shown only for a pause reason one of those can
  actually clear, and withheld where a manual apply re-runs the same guard.
- **frame reuse** — a frame the pipe extracts from a prior video is now bounded to 1920 on its **long edge**, not on
  its width alone. A portrait 4K phone video keeps its continuation frame instead of losing it with a ⚠️ note: a
  2160×3840 source now yields a 1080×1920 frame where it used to hand back a 1920×3414 one, past
  `VIDEO_FRAME_IMAGE_MAX_BYTES` at the shipped default, which dropped the frame with a note in the chat. Landscape
  sources and portrait sources at or under 1080×1920 are unchanged — the bound was already inside them.

  Detail quality on 9:16 sources is what this trades: 2160×3840 loses 3.2× its pixels, which is the point of the
  change and still leaves the anchor frame above 1080p-equivalent detail.

  Reach is common but content-dependent, not universal. A PNG's size follows entropy, not geometry, so the same
  shape keeps or loses its frame depending on the picture: busy, grainy, low-light or fine-detail 4K portrait clips
  lose it, clean or smooth ones keep it. What made it worth fixing is that the boundary is a knife edge — one stop
  brighter or one notch sharper flips it — and nothing beyond a ⚠️ line said so.

- **frame reuse, memory** — with the long-edge bound in place, the extractor asks no decoder for a frame larger than
  1920 on either axis, whatever the source, so its per-frame footprint is now bounded at about 3.7 Mpx and 10.6 MiB
  instead of scaling with the source's short side. The pixel-cap gate that used to refuse an over-cap portrait source
  from its header before decoding is no longer reachable — the bound keeps the geometry inside the cap — so a tall
  over-cap source such as 1920×20000 is now decoded rather than refused from a header read, and the pipe pays that
  decode. It already paid it for every wide over-cap source (10000×3000 is the documented rescue), so what is given
  up is only the tall-and-narrow half of an asymmetry with no principled basis. The gate and the decoded-size check
  behind it stay in the code for a header that lies about the source.

- **Pipe Dashboard, Config tab** — when two administrators save at the same moment on *different* workers, one
  save is now refused with "nothing was saved" instead of being silently lost. Until this, both saves were
  answered "Saved 1 setting": the per-pipe lock only ever serialised saves on one worker, and the revision it is
  checked against is a whole-second timestamp, so two saves inside one second are one revision apart by nothing.
  The refusal appears as the Config tab's own save-failure toast, names the other save as the cause, and leaves
  your edits staged. Installs without Redis (`WEBSOCKET_MANAGER` unset) keep today's behaviour exactly.
- **dashboard** — the System tab's **Readiness** panel now reports how much memory the session-log buffers hold, not only how many there are and how many records they keep. The `Log buffers (RAM)` row reads `2 buf / 40 000 events / 12.4 MB`, and a request whose records rolled off the top under the record cap says so on the same row. The cap counts records, so the counts alone cannot distinguish a quiet afternoon from a request holding a gigabyte of provider payloads in RAM.

  The figure is the number of characters the held records actually carry, and it is counted where a record is appended rather than recomputed by the panel — so opening the dashboard on a busy worker costs the same whatever the buffers hold. A record dropped by the cap is subtracted as it is dropped, a change to `Archive record cap` is recomputed from the records that survive it, and a request's figure goes when its records are released.

- **thinking boxes** — a reasoning box that grows in the middle of an answer no longer cuts that answer in two. The pipe used to flush the pending text into a message item of its own ahead of the box, so the reply a reader got back carried a line break the model never wrote, and a Fusion turn got a second copy of its answer. The answer is now one segment across a growth; only a **tool card** still ends a message item.

  This changes what the next turn is **replayed**: Open WebUI rebuilds the stored answer from the output array and hands that string to the model, so a turn that grew a box mid-answer now replays without the break that the flush used to add.

- **reasoning summaries** — a provider that fragments one `reasoning.summary` now keeps every fragment. The chat adapter kept a single scalar per summary key, so on a model that sends disjoint fragments under one `index` all but the last were overwritten before anything downstream saw them, and a cumulative snapshot was added on top of what had already been delivered. Every fragment now reaches the thinking box, the closing record and the replayed `reasoning_details`, exactly once and in arrival order.

- **fusion** — a Fusion panel member's tool file is no longer filed against the outer chat. The file is still
  stored and still rendered in the panel, but it no longer appears in the chat's file list: the tool executor
  re-checks `fusion_inner` before it hands `chat_id`/`message_id` to Open WebUI's upload, so a member's file
  cannot produce a `chat_file` row naming a real chat in a message that does not exist.
- **image contracts** — repointing `BASE_URL` now drops the shared published image contracts immediately, and not
  only when a contract sweep happens to run. The drop sat below the sweep's staleness gate, so a pass with nothing
  stale to do returned before it, and a gateway admin running with the four image filter valves off kept being
  served the previous gateway's aspect-ratio, resolution and seed limits until some later sweep ran. The identity
  is now adopted before the gate, so one stored value (`API_KEY`) has one reader and one answer.
- **provider error row** — on a failure reported inside a reply the pipe has already started, the **Provider error**
  row now names the provider's own error type wherever the body carries one, read from `error.metadata.raw.error.type`
  or from the failure's own `error.type` — the same two places the rejection path already reads — instead of an HTTP
  status number or the literal word `error`. Nothing else on the card moves: the resolved status, the typed kind, the
  template choice and the retry decision are unchanged for every shape, and a body that names no provider category
  still falls back to the numeric code and then to the stream marker. Custom templates using `{upstream_type}` are
  affected.
- **API key** — a stored `API_KEY` the key gate refuses now produces the authentication card on the streaming and the
  housekeeping-task legs too, instead of a 401 or an opaque "Unexpected error in streaming loop". Both of those legs
  read the stored field with a bare decrypt instead of through `Pipe._resolve_openrouter_api_key`, so an encrypted
  value that cannot be decrypted went out as an empty `Bearer ` and an encrypted non-`sk-` value the gate refuses went
  out working; a stored value with padding was sent with its padding. Both legs now go through the gate, so one
  misconfiguration has one answer on every leg.
- **model icons** — a model icon is now stored the way it displays. The icon sweep applies the orientation the
  source published before it writes the PNG, so a logo stored sideways is no longer stored sideways; an icon
  already stored keeps its pixels until its source URL changes, which is when it is downloaded again.
- **integer request fields** — a non-finite value in `seed`, `max_tokens`, `max_output_tokens`,
  `max_completion_tokens` or `top_logprobs` no longer ends the turn. `"inf"`, `"-inf"`, `"nan"`,
  `"1e400"`, `inf`, `-inf` and `nan` all raised `OverflowError` out of the field validator, which pydantic
  does not convert, so the request died before dispatch and the chat got an *Unexpected Error* card plus a
  failure-budget strike. The field is now simply sent unset, which is what the float fields beside it
  (`temperature`, `top_p`, `top_k` and the rest) have always done with the same spellings. A bool, a list, a
  dict and a non-numeric string are still refused with the same message: those are mistakes, not infinities.
- **per-model filter installs** — a single model's image or video panel failing to install is now named once
  per model per kind of failure, at WARNING, and repeats at DEBUG with the same message and traceback. A
  catalogue of a few hundred models against a database that is refusing writes cost a few hundred identical
  lines on *every* refresh, which buried the one line that names the model. A model that starts failing
  *differently* warns again at WARNING. The whole-pass lines (`OpenRouter Image filter ensure failed` and its
  siblings) are unchanged.
- **`ModelFamily.capabilities`** — removed. It was an accessor no production code called, kept only for tests;
  `spec["capabilities"]` is still written, still read by `list_models()`, and still merged into Open WebUI's
  `meta.capabilities`, so nothing about a model's checkboxes in the model editor changes. A third-party
  Open WebUI plugin importing the pipe and calling this one accessor would break; nothing inside this package
  does.
- **Log redaction no longer copies the payload it removes** — a record or a debug payload carrying a
  multi-megabyte picture now costs a few kilobytes to redact instead of a copy of the picture. The redaction
  answers the same way as before — a run of 1024 or more base64 characters is still cut to 64 characters
  with the same marker, and the prose around it is still left alone — but it reaches that answer by
  reading spans and indices instead of materialising what it is removing: the `sub` replacement was built
  from `match.group(0)`, which is the whole run; the guard asked `"data:" in text.lower()`, and `str.lower()`
  copies; and `url_scheme` handed a multi-megabyte string with no colon in it to `urlsplit`, which copies it
  to build a `path`. A record's redaction runs on whatever thread emitted it, so on a busy worker that copy
  landed inside whatever else that thread was doing. Nothing an operator sees changes.
- **One user's `Set-Cookie` is no longer replayed onto the next user's request** — every
  `aiohttp.ClientSession` this pipe builds is now given an `aiohttp.DummyCookieJar()`, so a
  `Set-Cookie` on any response is discarded rather than queued. This was the batch's only
  cross-user exposure, and it was live: `openrouter.ai` returns one `Set-Cookie` on both
  `/api/v1/models` and `/api/v1/key`, named `__cf_bm` — Cloudflare's bot-management cookie,
  measured against a test key on 2026-10-02, values not recorded. Every user's later requests
  therefore carried the same Cloudflare cookie and Cloudflare saw them as one client. It is
  **not** an authentication cookie, and no credential left one user and reached another through
  it. Behind a gateway at `BASE_URL` that sets a session cookie of its own (an auth proxy,
  `oauth2-proxy`, an nginx `auth_request`), one user's cookie could have reached another user's
  requests; whether your own gateway does this was not measured. The three affected sessions are
  the pooled per-event-loop session every concurrent request shares, the vetted transport's
  session (which fetches model icons, maker pages and admin-configured release assets — addresses
  somebody else chose), and the one `_ensure_async_subsystems_initialized` opens and hands to
  plugins as `ctx.pipe._http_session`. Nothing else changes: no valve, no timeout, and nothing
  in the package read a cookie before or reads one now. Operator-configured gateways and
  catalog icon hosts were not probed.
- **Temporary Chat error cards and records** — a Temporary Chat's socket id no longer appears in its error card or in
  the log records the fix names. Open WebUI mints a temporary chat's id by prefixing the browser's own socket id, and
  that id is what `/api/tasks/chat/{chat_id}` and `…/stop` resolve back to the socket and then to the owning user as
  their ownership check — so printing it on a card, or in the operator's own log line, handed out a live handle rather
  than a correlation string. `session_id` is now withheld on a `temporary:` or `local:` chat across every error path
  (the templated card, the provider-rejection card, the non-streamed auth card, the card returned to an API caller with
  no emitter, and the three queue-backlog warnings at both `WARNING` and the demoted `DEBUG` repeat), reduced to the
  empty string rather than hashed or truncated. `error_id` remains the correlation handle and `user_id` is not
  withheld — it is the account GUID, present in every chat. A saved chat and a `channel:` chat are byte-identical to
  before, and a channel card still withholds the ids while the operator's channel log line still carries them.
- **session log storage** — an archive written under the pre-digest path component is now found and merged by the
  next assembly pass for that turn, which keeps the outcome the older archive recorded and republishes the merged
  turn under the digest name. Only when the older file's `meta.json` names that turn's exact `ids`, so an archive
  belonging to another turn that used to share the bare stem is left alone, and the file that was read is left
  where it is for the retention sweep to reap.
- **pipe dashboard** — the usage retention window now covers every `dashboard_*` table this pipe published and a key
  rotation left behind, de-identifying its temporary-chat rows as it does the current table's, and the purge runs
  whether or not `Collect usage records` is on. A table whose fragment another installed function id also sanitizes
  to is skipped and named once in the log, and no table is ever dropped — only emptied.
- **Pipe Dashboard, action route** — the dashboard's `POST /api/pipe/dashboard/action` route now answers
  `401` for a request that Open WebUI itself would refuse. The route validated the bearer token, re-read the
  user row and checked the role, but never compared OWUI's `WEBUI_AUTH_TRUSTED_EMAIL_HEADER`: on a deployment
  behind a proxy that authenticates by trusted header, a request whose header named a **different** person than
  the token's own user was admitted, and the admin-only write actions behind it (`config_set`, `update_apply`)
  ran. It now compares exactly as OWUI's `get_current_user` does — the header value lowercased against the
  stored `user.email`, and only when the setting is on *and* the header actually carries a value — so the
  comparison is presence-optional and an API-token caller that sends no such header is unaffected. The
  property the route is built on is unchanged: the comparison reads a request header and never the session
  cookie, so the route stays CSRF-safe regardless of OWUI's CORS/SameSite settings. With the setting off —
  the default, and every deployment that does not run one — nothing changes at all. The header's *presence* is
  not required and an empty value is not a mismatch — both are skipped exactly as Open WebUI skips them, so a
  proxy that stamps the header on `/` but not on `/api/pipe/dashboard/action` does not lose the admin surface.
  The check is read from the header name Open WebUI itself bound at startup, so it names the header your
  deployment is actually using. The same branch now also refreshes the caller's "last active" time, which
  Open WebUI's does, so an operator reading that field to spot idle admins no longer sees dashboard-only
  admins as inactive. Both live in the route's own glue and do not follow a hot reload: they take effect at
  the next worker restart. The socket leg is
  untouched: OWUI performs no such comparison there either.
- **Video, prior-video frames** — the frame extracted from a prior video is now **read** as the person who
  asked for it, not as the pipe's storage service account. The read that copies a prior video out of Open
  WebUI storage to extract a frame was authorised with the storage account — an `admin` by default — so a
  turn whose user could not be resolved read any file in the deployment, including videos belonging to other
  people. Reads now run as the request's own resolved identity, which is what Open WebUI itself gates file
  content on; a turn with no signed-in user gets no prior frame and is told so (`⚠️ Previous video could not
  be loaded.`) rather than quietly sent one. The **upload** of the extracted frame is a write and is
  unchanged: the storage account still owns the file the pipe publishes. One consequence worth naming: a
  video turn driven by API automation, with no signed-in user, loses its prior-video frame. That is the
  intended outcome — Open WebUI gates file content on the requester and never on a service account — and the
  only alternative would be a pipe-owned service identity for prior-video reads.
- **Fallback storage account** — the service account the pipe auto-creates for the pictures a model generates on a
  user-less turn (API automations) is now created with **no linked identity**, the way Open WebUI itself creates a
  non-federated user. It used to be written with `oauth={"sub": "openrouter-pipe-storage:…"}` — the marker at the top
  level, where no OAuth provider key exists — so the row named no identity to Open WebUI's own readers while still
  satisfying SCIM's listing predicate, which accepts any non-null JSON in that column. The consequence an operator can
  see: the auto-created account **stops appearing in Open WebUI's SCIM listing**, and therefore in an IdP console
  that reads that listing. What does not change: the account is still found by `FALLBACK_STORAGE_EMAIL`, still owns the
  same pictures, and still serves `GET /api/v1/files/…` for them. **There is no migration.** An existing deployment
  keeps its malformed row until the account is recreated (change `FALLBACK_STORAGE_EMAIL`, or let it be recreated on a
  fresh install); the pipe cannot rewrite rows already in your database, and the old marker was a per-creation random
  value recorded nowhere, so it cannot tell which rows were its own. Note separately that with
  `OAUTH_MERGE_ACCOUNTS_BY_EMAIL` enabled a login whose IdP asserts this address still adopts the account — that arm is
  keyed on the email, and no change to the account's `oauth` column closes it.
