# Release notes pending

## Behaviour changes

- **Session log assembler, one offer per pass** — a turn the assembler already offered, or already failed, inside a
  pass is not offered again by that pass, and a turn whose assembly lock another pass holds is offered once per pass
  instead of once per re-listing round. The pass re-lists itself after meeting a lock-contended turn so the window
  keeps moving, but the exclusion it re-lists with was a snapshot taken before the loop, so a turn that failed in this
  pass — and a contended turn whose last failure was the stale-finalize seal, which the completed-turn listing's
  exemption had released — came back every round until the pass ran out its wall-clock budget, spending the whole
  `SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS` on SQL and never reaching the turns behind it. Nothing else changes: the
  backoff is still `SESSION_LOG_LOCK_STALE_SECONDS`, a turn whose only failure was the stale-finalize pass is still
  offered by the completed-turn listing straight away, and contention still books nothing.

- **Chat-message writes are gated on ownership** — a reply's error note, its Fusion snapshot and its turn metadata (`sources`, `annotations`, `reasoning_details`) are now written only to a saved chat the asker owns, or, for an admin, any saved chat. The gate matches Open WebUI's own rule for writing to an existing chat and fails closed: if the ownership check cannot be run, nothing is written and the operator log says why once per cooldown. Two consequences are visible: the Fusion panel and the loop-limit note are no longer persisted on a Temporary Chat or on a channel (both are still delivered live, and a temporary chat has no reload to restore from), and no turn writes into another user's saved chat. A `channel:` chat has no chat row of its own, so the write was already a no-op on a real installation.
- **Citation notices** — the "this response included a citation type the pipe can't render" notice now names the
  answer the reader is looking at. The notice used to be published the moment a delta carrying an unrenderable
  annotation went past, which is before the retry loop has decided anything: on a turn that was re-sent it
  described the answer that was thrown away, so the toast named a citation from a reply nobody was ever shown while
  the answer that replaced it got no notice at all. The types are now collected per attempt and published once, after
  the retry loop has closed. A turn whose surviving answer carries no such citation gets no notice, and a turn that
  ends in a failure card gets none either — it renders the card. The notice still arrives once per turn, and the
  types it names are the same set; only the attempt they are read off has changed.

- **Stalled accepted bodies** — a request OpenRouter has already accepted and whose body then stops arriving is one
  POST on every path, and it renders the connection card. The carve-out that names a body lost in transit keyed on two
  exception classes, `ClientPayloadError` and `ServerDisconnectedError`, so the three commonest ways a body stops
  arriving — an `Idle read timeout`, a reset socket, and an expired `Total request timeout` — were not in it. On the
  streaming `/responses` leg and on `/chat/completions` each of those spent the full retry budget on a call OpenRouter
  had already begun generating for, and OpenRouter bills on generation rather than on delivery: three POSTs on this
  leg meant three billed answers for one question, and the reader saw whichever attempt won. On the non-streaming legs
  the same faults were reported as a body that arrived and was not an OpenRouter response, which blamed the operator's
  proxy for a fault in the network path and, because that class is charged nothing, passed the breaker as a success.
  The discriminator is now whether the body **finished arriving**, on both streaming legs and both non-streaming ones.
  A body that arrived whole and delivered nothing still takes the full retry budget, a fault raised before the request
  reached OpenRouter is still retried, and an in-band or status-level provider error keeps its own class and its own
  budget. A mid-body timeout now renders the connection card rather than the network-timeout card; the timeout valve
  it was read from is unchanged, and so is every knob an operator can set.

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

- **Video generation, a temporary chat** — a video turn that runs the intent classifier no longer puts the browser's
  socket id into that classifier's task-model request. The classifier builds its own `metadata` rather than going out
  over the chat path, so the identifier valves never reached it, and it stamped the id unconditionally: on a stock
  install, with `SEND_CHAT_ID` at its default off, a Temporary Chat's id went out on every video turn that ran the
  classifier. The stamp now answers to the same two rules as the other two writers — a temporary chat's id is never
  sent, and `SEND_CHAT_ID` off means no chat id at all — so an admin's toggle now changes what the classifier sends as
  well. Nothing else about the request moved; the `task` key is unconditional, since it is what makes the call a task
  call. A saved chat keeps its id on the valve, and so does a `channel:` chat, which is not a temporary chat.

- **Video, channel resume** — a video resume in a `channel:` chat is now authorised against the channel before the stored turn is read, against Open WebUI's own predicate (membership for a `group`/`dm` channel with no admin exemption, a `read` grant or the `admin` role otherwise). A requester the channel does not admit reads nothing, so the turn **starts a new, separately billed job** instead of resuming someone else's stored turn. Open WebUI's own channel route refuses such a requester with a 403 before the pipe is called, so this fires only on a request shaped to bypass that route, and it cannot fire for a member resuming their own turn. Nothing from the other member's turn is read or shown. The identity the check uses is the resolved Open WebUI user, never a caller-supplied one.
- **Pipe Dashboard, a second administrator** — on an installation with Open WebUI's `ENABLE_ADMIN_WORKSPACE_CONTENT_ACCESS` off (so `BYPASS_ADMIN_ACCESS_CONTROL` is false), an administrator who is neither the owner of the `Pipe Dashboard` model row nor holds a grant on it now loses the dashboard: the view, the live-feed room and the `read`-permission actions, including `config_get`, which returns the whole stored valve set. The dashboard read gate asked Open WebUI's `check_model_access` and took a non-exception for an answer, and that function puts its entire grant decision inside `if user.role != 'admin':` — so it admitted every admin whatever the flags said, while the Open WebUI gates that decide whether a model is visible at all refused him. Administrators now answer the same question Open WebUI does: `BYPASS_MODEL_ACCESS_CONTROL`, or the admin flag, or owner, or a `read` grant; with the bypasses off a model row that cannot be read grants nobody, administrators included. Nothing changes on the shipped default (`BYPASS_ADMIN_ACCESS_CONTROL` is on), on the row's own owner, or for anyone holding a grant. Note that `whoami` is itself a `read` action, so a refused administrator cannot call the action that would have told them their status — the panel falls back to its access-denied card.
- **Dashboard Update tab, restore on a package or stub install** — Restore is now refused there with the same
  `package_mode` outcome Update already gave, instead of writing an older bundle snapshot over the install and
  turning a pinned install into a self-updating bundle one. The tab no longer shows the Restore button on a
  package or stub install either; its snapshot list and Delete action stay as they were.
- **`file_url` with no scheme** — an attached `file_url` written without a `scheme://` is now refused like any
  other link the provider could not dial, and the person sees the existing `Files: skipped N (served from a link
  that is neither http nor https, which is blocked by security policy).` status. A bare path (`/etc/passwd`), a
  relative reference (`../../etc/passwd`) and a `host/path` written without one (`169.254.169.254/latest/meta-data/`)
  all reached the provider ungated; they are now put to the address gate and refused on their scheme. The gate is
  the same one an `ftp://` link already met, and the refusal reuses that sentence unchanged — nothing is dropped
  silently. `ENABLE_SSRF_PROTECTION=False` does **not** restore these, because the scheme rule is sequenced ahead
  of that valve exactly as it is for `ftp://`. Raw base64 in `file_data`, a `data:` URL in either field and an
  Open WebUI file path are unchanged: they are inline payloads or become a `file_id` before this point, and are
  never checked. Open WebUI 0.11.4 emits no `file_url` of its own, so this only reaches a custom filter, agent or
  API caller that writes one by hand.

- **Multi-worker Redis deployments** — an admin save of the dashboard configuration, or an unattended update,
  no longer closes the Redis client that Open WebUI's own websocket bookkeeping shares. The pipe took its
  cross-worker lease over the cached client and then tore that client down when it released the lease, so on
  any install with `WEBSOCKET_MANAGER=redis` and `WEBSOCKET_REDIS_URL` set, every configuration save and
  every leader-election poll cut connections that `MODELS`, `SESSION_POOL`, `USAGE_POOL` and the two cleanup
  locks were using. The client survives — the cache still holds it and redis-py reconnects — so the visible
  effect was the `Session pool cleanup failed. Retrying.` and `Unknown receive error` lines around a save,
  plus reconnect churn, rather than an outright outage. The lease is still released exactly when it was
  taken, and a deployment without a distributed lock is unaffected: there was never a client to close.

- **Web-Tools filter valves, the default the pipe seeded** — turning *both* Web-Tools filter valves off
  (`AUTO_ATTACH_WEB_TOOLS_FILTER` **and** `AUTO_INSTALL_WEB_TOOLS_FILTER`) now also releases the default the
  pipe had seeded for the panel, on the same pass that takes the panel off `filterIds`. The family-off arm
  cleared the filter list and left the seeded `defaultFilterIds` entry behind against a panel that no longer
  ran, and the entry could not be removed by the pipe afterwards. The three sibling families already
  consulted their own family-off flag here. A blank panel id that means "the installer did not answer" is
  unchanged: nothing is released and the install is retried, and turning the family back on re-seeds the
  default. `AUTO_ATTACH_WEB_TOOLS_FILTER`'s own help text now says the release needs
  `AUTO_INSTALL_WEB_TOOLS_FILTER` off as well, which is what detachment has always keyed on.
- **Native image filters, a superseded panel** — a panel the pipe retires as superseded now leaves the model
  rows it was still attached to on that same pass: the id comes off `filterIds` **and** `defaultFilterIds`,
  with either image valve on. The sweep's verdict reached the filter list only on the branch that computes it
  itself, and never reached the default list on either branch, so a deactivated filter could stay attached and
  default-on. An id the pipe did not retire is untouched.
- **`meta.builtinTools` the pipe wrote from a File-context untick** — the pipe now writes
  `builtinTools.files = false` only on the media models whose own media rule produced the `file_context`
  untick this pass. On a text or vision model the only source of that untick was the operator's own tick in
  the model editor, and the pipe answered it with Open WebUI's own off-switch for `list_chat_files`,
  `query_chat_files`, `grep_chat_files` and `view_file` — the tools `file_context` off is there to turn *on*.
  An operator who unticked the box now keeps the chat-file tools. A media model the pipe really is correcting,
  and a Files default an admin ticked in the editor, are both unchanged.
  **Rows already carrying `builtinTools.files = false` from an earlier build keep it**: there is no valve to
  undo that write and no tick-back, so untick the box in the model editor, or wait for the marker work
  (B682/T971) that can tell a pipe-written entry from an admin's.

- **Pipe shutdown no longer drops a session-log write that was still in flight** — a turn cancelled by `close()` writes its terminal
  segment from its own cleanup, and that write is now given a budget of its own: it is registered when it starts and awaited before the
  artifact store is closed, so a write slower than the 5-second job-drain budget finishes instead of being dropped with the store closing
  under it. The drain's budget is unchanged, so `close()` stays bounded; on the normal path there is nothing to wait for. The log an
  operator most wants — the turn a reload or a shutdown killed mid-answer — is the one that was at risk.
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
- **Video attachments, the size the pipe measures** — a clip whose real video size is below the
  model's published pixel floor is now withheld with a notice naming that size, where before an `.avi` or ProRes
  `.mov` was measured wrong and sent for the model to refuse; and a clip whose cover art came first is no longer
  withheld, because the cover's size is no longer read as the video's. The declared geometry is now the picture
  stream's own, read off the field ffmpeg writes it in, so a cover-art stream is skipped and a sub-floor clip is
  left out with a notice a person can act on.
- **Open-WebUI tool mode** — a call Open WebUI runs now shows its card from the moment the model names the tool, and the dashboard's live view now shows the running tool on those turns (it stayed empty before).
- **Pipe dashboard, deleted function row** — deleting the pipe's function row now releases the live dashboard on the worker that served the DELETE and lets
  the pipe finish its in-flight requests and close, instead of holding it, its session-log threads and its storage handle until a restart. The stand-down
  happens whether or not `PIPE_DASHBOARD_ENABLE` is on, and it is per worker: on a multi-worker deployment the other workers release their own generation at
  their next hot reload, exactly as Open WebUI keeps its own per-worker function cache. The action route and the socket gate still refuse once the row is
  gone; nothing about the authorization answer changes.
- **Pipe dashboard, usage collection switched on after start** — an install whose dashboard model is off but whose usage
  collection was switched on after the worker booted now records usage, and runs the abandon sweep, without a restart.
  The reaper used to be armed only from the boot path and from the model-list pass *below* the dashboard-off return, so
  a collect-only install that was flipped on later never reached it: sessions were tracked, abandoned ones accumulated
  with nothing to finalize them, and the Usage tab's **Failed** count was short exactly the rows nobody was looking at.
  The reaper is now armed by the first request that starts tracking and by the model-list pass above that return, so it
  comes up within one chat turn or one model-list request, whether or not `PIPE_DASHBOARD_ENABLE` is on. No query is
  added to the request path: every one of those sites reads the valve copy Open WebUI rebinds before the call.
- **Pipe dashboard, session tracker bound** — the tracked-session registry is now bounded. At the bound the stalest entry
  is finalized as `failed` — one usage row, carrying that request's own cost and tokens — and a latched warning names
  the bound that was reached. The **Active** tile and the Live table are unchanged: the tile still describes the whole
  tracked population and the table still renders its 30 newest rows, and a tracker under the bound tracks everything it
  is asked to.
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
  The eight pre-send refusals (Zero Data Retention routing in force with the model off the roster, the ZDR
  endpoint list unreadable, Zero Data Retention requested for a model whose request format cannot carry the
  flag, an endpoint-override conflict from a preset or from Direct Uploads, a Fusion model
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
- **Provider routing** — the per-model endpoint sweep behind the routing dropdowns is now bounded as a whole at 45 s
  instead of running one per-read cap at a time. With 50 slugs listed against a slow or unresponsive OpenRouter
  endpoints host, the old fan held a metadata pass for roughly 25 minutes while it ran its legs in ten-wide waves,
  and the metadata-sync key stayed committed the whole time, so every later schedule was refused. The pass now comes
  back, logs how many of the listed models were still unfinished at `WARNING`, and carries on. An abandoned slug
  keeps its provider data from the last cycle and has its routing row left exactly as it is — for an admin-enforced
  routing filter, that is the difference between a stale dropdown and a setting the pipe switched off on its own.
  Nothing is announced as a spelling problem: a slug whose read *finished and failed* still is, exactly as before.
  The cost is freshness on a slow host, never correctness, and `HTTP_TOTAL_TIMEOUT_SECONDS` remains the lever for how
  long a single read may take.
- **provider routing** — the two WARNINGs a generated routing filter raises when it heals a
  stored row no longer name the stored value. They name the field and what the filter kept
  instead, because a valve row holds whatever was typed into it — a `data:` URL, a signed
  link, a paragraph of prose — and none of that belongs in a log. What the warning was for
  is unchanged: it still says which field went stale and what it was replaced with, and a
  value that is still on the option list still produces no record at all. Open WebUI's own
  filter warnings name the filter and never the payload, and these now match that.

  **Every installed routing filter is rewritten on the first model-list refresh after this
  update.** The two strings live in the generated Python source, and the pipe compares each
  row's stored source against freshly rendered source byte for byte, so the first refresh
  rewrites all of them — once, for this change, and silently. Nothing about the rows'
  scope, ownership or behaviour changes with it; the rendered code differs only in the two
  format strings. If you have hand-edited a routing row, that edit is reverted by this
  rewrite for the same reason any other regeneration reverts one.

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

- **address checking** — a request's `ADDRESS_CHECK_BUDGET_SECONDS` is now spent by address checks and by nothing else.
  A remote picture's **transfer** used to come off that budget, so a turn carrying a few slow pictures could leave a public
  video link or a file link behind them with no time to resolve, and the link was refused with the sentence for a check
  that never finished. The download's own address check still draws on the budget; the bytes it moves afterwards do not,
  and the valve help text, the dashboard entry, the atlas and the security and multimodal documents now say so in the
  sentence that already promised it.
- **pictures** — one request now resolves at most 16 addresses, whatever the picture count. `Maximum images per request` only ever counted
  what was **forwarded**, so a picture the address gate refused consumed no slot and a message carrying two hundred unfetchable remote
  pictures paid two hundred lookups on top of the one the pipe downloaded each. The ceiling applies to every forwarding arm — the
  cold picture, a picture reused from an earlier turn, the download's own check, a picture a tool result carries, a file link and a
  video link — and a link past it is refused without being looked up and without being downloaded, with the count named in the refusal.
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
- **Video turns on the dashboard** — a video turn now books its real usage on the dashboard, and a failed one counts as Failed rather than Completed. The video adapter reports its own terminal state, so a Usage row carries the job's real tokens and cost where it used to carry zeros, and a video that failed, stored no clip, or stalled no longer reads as a free success. A second request that attached to an already-running video job reports under its own id without that job's usage, so one billed clip is one row's spend.
- **model icons** — a model icon is now stored the way it displays. The icon sweep applies the orientation the
  source published before it writes the PNG, so a logo stored sideways is no longer stored sideways; an icon
  already stored keeps its pixels until its source URL changes, which is when it is downloaded again.
- **reasoning budget** — a `reasoning.max_tokens` written as a number in a string (`"2048"`, `" 2048 "`,
  `"2048.7"`) is now read as that number, on both endpoints: it is reserved against the request's own output
  cap like an integer budget, and on the Gemini 2.5 leg it overrides `GEMINI_THINKING_BUDGET` instead of being
  replaced by it. A value that expresses no number (`"abc"`, `""`, a list, a boolean, `"inf"`, `"1e400"`,
  `NaN`) used to reach OpenRouter verbatim as the budget and produce a failed turn; it now produces a turn with
  **no thinking budget**, which on a wide model is a slower, more expensive turn rather than an error. Nothing
  is logged for a dropped value, as the carried-field convention has always been.
- **responses streaming** — every SSE frame on `/responses` is parsed exactly once per turn now, by the
  reader that decodes it, and the workers forward that parsed frame instead of parsing it a second time. The
  pre-first-output buffer therefore holds parsed frames rather than byte strings, so a stream whose opening
  frames a fallback has to discard no longer keeps the raw bytes of those frames in memory.
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

- **Unencodable characters in a provider's reply** — a provider's unencodable characters reach the card and the
  log record as U+FFFD instead of losing them and the archive.
- **Fusion, Continue of a Fusion answer** — the terminal snapshot no longer replaces the stored answer with this
  generation's alone. It was written ungated, so on a Continue -- a turn carrying `assistant_message_id`, which
  Open WebUI sends only when continuing, and which on the hosted backend (`FUSION_BACKEND='openrouter'`) is how a
  person presses Continue on a Fusion reply -- the upsert cut the stored message back to the text this generation
  produced and dropped the half it was continuing from. The panel (`embeds`) is written either way, and a turn
  Open WebUI is *not* holding still writes `content`: on a first generation the Fusion answer travels as a native
  output item rather than as `delta.content`, so that write is the only record of it.
- **A refusal that opens a Continue** — the refusal now starts a block of its own, so the separator Open WebUI's
  join needs is emitted before it. A refusal-first Continue had none: the refusal arrived glued to whatever followed
  it, and when the stored reply ended on a hidden marker line the two fused into one line the marker parser no
  longer read -- the continuation's own segments were lost with it. A stored reply ending on ordinary text gets the
  blank line it needed for the same reason. Nothing changes on a fresh turn, and nothing changes on a Continue
  whose stored reply is the empty string: there is no stored line to separate from, and no separator is added.
- **Unrecognised content block, a link in any letter-case** — an opaque block type (`vendor_widget` and friends)
  is forwarded verbatim, and the one thing that stops being forwarded is a block that carries a *location*. The
  test for that was a prefix comparison and a substring search on the raw value, both case-sensitive in a way the
  URL grammar is not. `HTTPS://169.254.169.254/latest/meta-data/`, `FTP://…`, `Gopher://…`,
  `HTTPS://OWUI.EXAMPLE.COM/API/V1/FILES/abc/content` and a `//host/x` behind a leading newline are all the same
  links as their lower-case twins, and every one of them — credentials included — went to the provider inside a
  block the typed arms would have refused. The pipe's own URL parser now decides, at whatever nesting depth and
  under whatever key but a prose key, so a block that is refused today stays refused and the mixed-case spellings
  join it. Two shapes are pinned as staying dropped rather than left to a parser's accidents: a `data:` URL and a
  `javascript:` URL whose *payload* names the file endpoint, neither of which is a link the provider would dial
  and both of which were dropped before. One value is deliberately left alone, because it names no address and no
  endpoint: `Api/V1/Files/abc/content`, with no leading slash.
- **A tool's picture that names a host** — an absolute URL a tool result carries is now put to the
  plaintext and address valves like any other link. The pipe has always had a classifier that answers "is
  this one of my own file endpoints?" from the *path* alone, and the six forwarding gates were reading it
  as "and therefore this is not a link" — so a path a tool wrote into a URL skipped the checks: a
  cleartext `http://169.254.169.254/api/v1/files/abc` skipped the plaintext valve, an `https://` one on a
  link-local or private address skipped the address check entirely, and one written as
  `https://user:pw@host/api/v1/files/abc/content?token=SECRET` skipped both and reached OpenRouter with
  the tool's own credentials in it. A path is not an origin, and `ENABLE_SSRF_PROTECTION` and
  `ALLOW_INSECURE_HTTP` promise to hold on every forwarding site, so the gates now exempt only a
  **relative, host-less** reference whose parsed path *begins* with `/api/v1/files/` — the spelling Open
  WebUI itself mints, which stays exempt on every valve and is asked about zero times, as does any
  letter-case of it. Everything else with that path in it (`file:`, `//host/…`, `\\/\/host/…`,
  `https://host/…`) goes through the cleartext valve, the scheme gate, the address check and the verdict
  memo on every leg, and a refused picture is skipped with a status; the turn is not lost. **What this
  costs:** a deployment whose Open WebUI is on a private or link-local address loses an absolute self-link
  from a tool (`https://10.0.0.5:3000/api/v1/files/<uuid>/content`) — it is skipped rather than inlined,
  because the pipe has no configured Open WebUI origin to compare against and comparing paths would reopen
  the hole above; that is filed as its own item. Five relative spellings that hide the path behind a dot
  segment, a query value or a fragment (`./api/v1/files/abc`, `/proxy?path=/api/v1/files/abc/content`) are
  now refused `not a link the pipe can resolve into an image` on the tool gate; they still resolve on
  every attachment arm, where the host-agnostic classifier is what runs. `/chat/completions` is now part
  of the same promise: an `image_url` block naming a file reference is resolved from local storage before
  the request leaves instead of being forwarded as a link. Nothing about the attachment arms, the video
  arm or the host-agnostic resolve classifier changes.
- **Task-model failure notices in shared rooms** — when a task model (titles, tags, follow-ups) fails in a `channel:`
  chat or a saved chat several people share, each member now gets their own one-time notice. The notice used to be
  latched per conversation, so the first member's notice silenced everyone else in the room for the window. The DEBUG
  lines that record a suppressed or undelivered notice (`task-failure toast suppressed: …` and `task-failure toast not
  delivered; latch left open: …`) now name the user id beside the chat id for a saved or `channel:` chat; a temporary
  chat's line still names no chat.
- **error templates** — a stored operator row that fences a value on its own lines now renders a code block that
  closes on its own opening run. When the provider body carried a fence run long enough to widen the block, the
  template's own closer stopped being recognised and the card's closing guidance was swallowed into a block that ran
  to the end of the card; an indented opener row was rewritten from column 0 and emitted a run the closer did not
  match. No shipped template fences a value, so a card from one of the built-in templates is byte-identical.
- **attachments** — a content block whose conversion raises is now dropped and named in the `Files: skipped N (…)`
  status on your turn, as every other refused attachment already is, instead of being forwarded in the shape
  Open WebUI handed it over — a block the provider does not accept, carrying text the pipe had not finished
  preparing. The turn itself is unchanged: the model goes on with whatever else it carried, and a turn whose only
  block faulted still says so in-band. A block the pipe has no converter for is still passed through untouched.
