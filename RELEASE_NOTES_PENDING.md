# Release notes pending

## Behaviour changes

- **Gemini thinking budget, `0` is now a real off switch for the whole Gemini family** — setting `GEMINI_THINKING_BUDGET` to `0` used to switch thinking off on Gemini 2.5 only. On Gemini 3 it reached nothing at all: the request went out at the chat valve's effort with reasoning and the summary flag still on it, so thinking was on and nothing said so, while the Config tab promised the `0` switched thinking off. The `0` now reaches every Gemini model, Gemini 3 included; a mandatory-reasoning Gemini row (Gemini 3 Pro, like 2.5 Pro) still cannot stop thinking, so the refusal stands, the row's own floor is substituted and the notice goes out in a chat as before. Nothing else moves: the token budget is still Gemini 2.5's alone, because Gemini 3 is governed by Google's thinking-level API and a budget there buys no precise control, so any other value of the valve is ignored on Gemini 3 exactly as before.
- **A preset Fusion turn is no longer repaired by the chat fallback** — a Fusion turn served
  under its `@preset/` spelling (`openrouter/fusion@preset/<slug>`) whose `/responses` call
  fails used to be re-sent to `/chat/completions` by `AUTO_FALLBACK_CHAT_COMPLETIONS`. That
  retry was a *different* request: the `/chat/completions` converter strips the activating
  `{"id": "fusion"}` entry, because the entry cannot travel there, so OpenRouter answered it
  as an ordinary completion, the pipe had no panel to render, and the turn finished as a
  flattened transcript with the breaker strike already refunded. The fallback guard read the
  raw wire id and the converter read the normalised one — two questions, one shape, opposite
  answers — so the guard now asks the converter's question and the retry does not happen. The
  provider's own refusal is what reaches the person, the breaker strike stands, and the turn
  archives as `error` rather than `complete`. A non-Fusion model still falls back exactly as
  before, and a caller-supplied `enabled: false` Fusion entry is still the per-request opt-out
  it was. The same guard governs the streaming and the non-streaming leg.

- **A per-model advanced parameter now schedules the pass that honours it** — the nine `disable_*` flags Open WebUI's model editor writes are read only by the background metadata pass, and nothing in the pass's schedule observed them: an operator who ticked `disable_web_tools_default_on` on one model changed what the pipe *would* write and nothing told it to write it, so the row kept its old state until a catalogue fetch happened to move the content stamp or an admin moved some unrelated setting. A completed pass now records a digest of the flags it read, per model row, and the next model-list build schedules when that digest differs — so saving the model takes effect within one model-list build instead of waiting for something unrelated to change. With nothing changed the digest settles, so the pass still runs once per real change and not once per refresh, and the scheduler still reads no model rows of its own.

- **Switching the last sync valve off now detaches on the pass after the switch** — the gate that decides whether a metadata pass runs is a disjunction over the sync valves and the routing lists, so it can answer "on" but never "this one just went off": with the last term on, an admin switched something off and the pipe attached no pass, so the panels it had attached stayed attached until some unrelated tracked setting happened to move, which for a valve edit reads to the operator as never. The scheduler now also schedules on an on→off transition of any of those terms, and the pass that runs admits itself through the body gate so it actually detaches. Nothing changes for a deployment with a valve on: turning one on still schedules one pass and then settles, and with every term off the gate still declines on every refresh rather than syncing the whole catalogue each time.

- **Video, Continue Response after a Stop** — after Stop on a video turn, a Continue now picks the job already running and waits for it, rather than submitting a second one. Previously the Stop cancelled the request and that request's cleanup also dropped the in-process claim the still-polling job held, so the Continue started a second poller on the same `job_id`: the clip was downloaded and stored twice and the same generation was billed twice. The wait is the point — it is what makes the job bill once — so a Continue issued minutes after a Stop now waits for the render it already paid for instead of starting a fresh one. Shutdown reaches an abandoned job too, which it could not while its claim was gone.
- **Video turns and the circuit breaker / session log** — a video turn that renders a complete answer now resets the user's circuit breaker, and a video turn the pipe refuses before sending is archived as `error` with the control that refused it, rather than `complete` with no cause. Six of the video adapter's early returns handed back a card and recorded nothing: a stored clip or failure block already on the message, a clarification, the per-user video job cap on both the new-generation and the resume arm, and a message with no prompt. So a turn that succeeded left an open breaker open, and a chat holding nothing but a refusal card was filed as a clean turn. The refusal names the valve — `MAX_CONCURRENT_VIDEO_GENS_PER_USER` — or the missing prompt. A turn the person stopped still archives as `cancelled`, unchanged.
- **A turn's address checks now share one budget** — the `ENABLE_SSRF_PROTECTION` help text has always promised a
  single request-wide `ADDRESS_CHECK_BUDGET_SECONDS` ("that turn's other address checks draw on", "a repeated link is
  resolved once per request"), and the conversion leg — the one that rewrites a chat into a Responses `input` array —
  was the single phase that drew on a private budget and a private verdict memo of its own instead of the request's.
  So a request could spend up to twice the advertised budget (20 s in the transform plus 20 s in the tool round that
  followed it), and a link the transform had already resolved could be resolved a second time by that tool round,
  which is precisely what the text says does not happen. The transform now arms the request's own budget and publishes
  into the request's own verdict memo, as every sibling phase already did. **What a person can notice:** a
  picture-heavy turn can now starve its own tool round's picture check, where each of the two previously had 20 s to
  itself, and the person sees the existing `could not be checked in time, so it was not sent` wording rather than a
  new message. Nothing else about the refusals changes — a check that never answered is still named as unchecked
  rather than as a private address, a completed refusal still reads `could not be fetched, so it was not sent`, and a
  turn that runs with no request context installed at all (automation, the live rig) still draws a private budget of
  its own.

- **The `/responses` leg now gates the media it forwards** — the outbound filter that runs immediately before a
  `/responses` body is POSTed read no media at all: it dropped undocumented top-level keys and copied `input` through
  byte for byte, so every media value the conversion leg already refused for `/chat/completions` was forwarded
  verbatim on the other endpoint. It now puts each one to the same predicate — no cleartext `http://` link unless
  `ALLOW_INSECURE_HTTP` and `ALLOW_INSECURE_HTTP_HOSTS` permit it, no Open WebUI file path, no scheme a provider can
  dial, no non-base64 or over-bound `data:` payload, no `input_audio` URL — drops the block, and names the refusal.
  `ALLOW_INSECURE_HTTP`'s and `ENABLE_SSRF_PROTECTION`'s help text now say both legs; neither adds an address check,
  which still happens at ingress alone. The reach is not a caller's own `/responses` body: `CompletionsBody.messages`
  is required and ingress overwrites `input`, so what puts a media block inside `body.input` is a **filter function
  rewriting `messages`**, or a **`message` item the model itself authored** in a reply — stated here because it is
  narrower and stranger than the item's own text promised. The request sanitizer's `function_call_output`-only scope
  is deliberate and is not the gate for anything else; the outbound walker is. `input_audio` keeps its own rule
  (OpenRouter takes audio as base64 only) and is not widened into this gate. A value one of the two legs refuses is
  still reported as `Images: skipped N (<reason>).`, one line, never twice for the same block.

- **A tool's picture the inliner cannot name is skipped and named, not fatal** — when the file gateway resolves an
  Open WebUI file path to a file id, it inlines it and sends a `data:` URL, because no provider can fetch this
  deployment's own authenticated endpoint. Four spellings of that path yield no id at all — `/API/V1/FILES/abc/content`
  (the id pattern does not fold case), `/api/v1/files/internal.png` (the only segment is a filename),
  `/api/v1/files/%61bc/content` (percent-escaped) and `/api/v1/files/` (no segment). For all four the storage read was
  never attempted and the branch still fell through to `FileUnavailableError`, which is a terminal card: one tool
  picture the pipe cannot resolve cost the person the whole turn, and the sentence it stopped with claimed the file
  was gone — a claim about storage the pipe never made. Those four are now dropped from their block, the body is
  still dispatched, and the skip is named in the turn's usual `Images: skipped N (a picture naming … cannot be read
  from Open WebUI storage).` line. A reference whose id **does** read is unchanged: inlined when readable, terminal
  error card when the record is not there, because in that case the pipe did read the record. A file the *person*
  attached is a different fact and is unchanged, and so is the `input_file` arm. The two vocabularies are distinct on
  purpose: the gate's own causes (`insecure_http`, `unusable_link`, `unencoded_inline`, `oversized_inline`,
  `internal_reference`) describe a link the pipe chose not to forward, while `unreadable_internal_path` describes a
  reference it could not read at all, and the operator log says which is which by naming the cause on every line.

- **Per-user "How long to keep reasoning" now defaults to the whole conversation** — the
  per-user `PERSIST_REASONING_TOKENS` control used to default to `next_reply` while the
  administrator's site-wide valve of the same name defaulted to `conversation`, so a user who
  had never opened their own settings silently got the shorter window. It now defaults to
  `conversation` too: a user who chose a value keeps it, and a user who never set the field
  gets the administrator's site-wide value as before. A user whose stored row the pipe cannot
  read (a rotated `WEBUI_SECRET_KEY`) falls back to this per-user default, which is now
  `conversation` rather than `next_reply`. The cost is input tokens, not correctness: a longer
  chat replays more of the model's earlier reasoning back to it on every later turn. OpenAI's
  reasoning guide is the reason for the default — the GPT-5.6 model family supports
  `all_turns` and uses it by default, which needs the complete history replayed, and that is
  what `conversation` does where `next_reply` keeps only the previous turn's. `disabled` and
  `next_reply` themselves, their cleanups, and the administrator's default are unchanged.

- **Tools are no longer sent in strict mode by default** — `ENABLE_STRICT_TOOL_CALLING` now starts off, so a
  tool schema reaches the provider as its author wrote it, with an explicit `strict: false` rather than being
  converted to strict JSON Schema. OpenAI rejects a strict tool whose parameters contain a free-form object, and
  Open WebUI's built-in `ask_user` is exactly that shape (`questions` is an array of objects that declare no
  properties), so with strict mode on by default every request offering `ask_user` was refused by an OpenAI model
  before any tool ran. **Migration:** an admin who wants strict schemas back ticks the valve; the conversion
  itself is unchanged, so turning it on gives today's strict behaviour for every tool.

- **A background task on a picture-only or video-only model is skipped, not sent** — with no Task Model
  set, Open WebUI asks the pipe to title, tag and generate follow-ups using whatever model the chat used.
  On a model whose catalogue row lists no `"text"` among its output modalities, those requests cannot
  succeed: the pipe spent two attempts, logged an ERROR and toasted a task-failure card naming a fault that
  is really a property of the model (GitHub issue #61). The pipe now answers an empty task answer instead,
  before anything leaves, which is the value every Open WebUI task kind already reads as "no task ran" — the
  chat takes its first message as its title, exactly as it does with automatic titles off, and no tags and no
  follow-ups are written. The rule is one predicate on the resolved model's spec, so a variant or a virtual id
  resolves to its base; a model with no spec, or one that lists no output modalities, is treated as able to
  answer and keeps today's behaviour, and a multimodal model that emits text alongside pictures keeps its
  tasks. It is not an error: no breaker record, no card, nothing shown to the person. The first skip in a
  worker logs one WARNING naming the model, what it makes, and where to set a Task Model
  (**Admin Settings → Interface**); every later skip logs the same line at DEBUG.

- **A video model's own instructions now apply on "use my prompt verbatim" turns too** — the
  classifier writes the scene, but a turn where it was asked to hand the person's own words back
  untouched used to hand the video model those words *alone*: the deployment's system prompt, which
  the pipe prepends to every other video turn, was dropped from the wire, and the footer said
  nothing about it. The pipe now composes on both arms, so the model's standing instructions go
  ahead of the scene exactly once whichever arm a turn takes, and the classifier's own prompt no
  longer tells it to fold them in as well. The footer keeps showing the scene alone and adds one
  line saying the instructions were sent too — a system prompt is configuration, not something a
  person asked to read in their own chat. No valve and no configuration change.

- **Video generation, dual models** — a model the chat and video catalogues both list now publishes one verdict
  about pictures instead of two. `capabilities.vision` and `capabilities.file_upload` restate the row's merged
  feature set rather than keeping the chat twin's own answer, so a model whose chat twin takes no picture but whose
  video row takes a first frame is offered the `vision` box instead of being refused the attach it would have sent,
  and a twin that takes pictures keeps the box when the video row takes none. The other seven capability boxes are
  unchanged, so an image-generation answer the chat catalogue published still stands. The Direct Uploads decision
  is unaffected: it is read from the row's features, not from this dict.

- **A tool call the pipe gave up on is now counted as a failure on the dashboard** — a round could
  settle a call without ever reporting it: a call cut when the tool queue was still full when the
  batch ceiling ran out (it never started), and a call `TOOL_IDLE_TIMEOUT_SECONDS` gave up on,
  whether it had started or was still waiting for a worker. Both build the model's output exactly
  as before and both dispatch to plugins now, so a call the pipe answered the model about is a call
  the operator's Live row shows, and the Live row returns to `streaming` instead of being stranded
  on `status: "tool:<name>"`. Nothing the model sees changes: the output list for every one of those
  arms is byte-identical, and no valve moves — `TOOL_IDLE_TIMEOUT_SECONDS` is still unset by default
  and the plugin system is still off unless an admin turns it on. **Not comparable across this
  release:** persisted usage rows (`db_row` writes `tools_failed`, and every Usage window sums it)
  written before this fix count fewer failures than rows written after it, so a `tools_failed`
  figure that crosses this version is comparing two different things. Both give-ups are rare — the
  idle arm needs an idle limit an admin has set, the enqueue arm needs a round wider than the
  queue with no free worker for the whole batch ceiling — which is exactly why the shortfall it
  closes survived on the rows an operator is chasing.

- **Pipe dashboard, API-key callers** — the action route behind every dashboard tab now stands behind Open WebUI's own `get_verified_user` rather than a hand-maintained copy of `get_current_user`, so an `sk-` API key is admitted where it was previously decoded as a JWT and refused. Everything else about the route's identity decision is unchanged in substance and delegated wholesale: a request with no usable `Authorization: Bearer` credential is refused before Open WebUI sees it at all, so the session cookie and the `x-api-key` middleware fallback stay unreachable and the route remains CSRF-safe. What changes for an operator is the surface: a key holder Open WebUI permits can now drive the dashboard, bounded by the same read/write grant and admin-role checks as a session, and governed by Open WebUI's own API-key settings — `auth.enable_api_keys`, the `features.api_keys` permission for a non-admin key holder, and `auth.api_key.endpoint_restrictions`, which names `/api/pipe/dashboard/action` in its allow-list. Set that restriction if a deployment wants the old surface. A `401` now carries Open WebUI's own message rather than a bare `{"detail": null}`, so an admin on an expired session is told why instead of being shown "Could not load configuration".

- **Admin's Max Upload Size, no longer gated on the RAG bypass flag** — an install that turned
  `rag.bypass_embedding_and_retrieval` on stopped being held to the cap its admin last saved under
  **Admin → Settings → Documents → Max Upload Size**: the pipe read the bypass flag beside the stored
  size and treated "bypassed" as "no cap", handing that admin a `REMOTE_FILE_MAX_SIZE_MB`-sized download
  from an install whose own upload leg was refusing at the stored number. That flag switches off
  *embedding*, not the admin's ceiling — Open WebUI applies `rag.file.max_size` on its own upload leg
  with no reference to it — so the stored cap now binds either way, and the pipe asks the store for
  that one key rather than two. No valve default changed, and the smaller-wins arithmetic, the
  default-valve carve-out, the 500 MB ceiling and a cleared admin box all behave exactly as before on
  an install that does not use the flag.

- **Plaintext HTTP allowlist, a bare entry names one port** — an `ALLOW_INSECURE_HTTP_HOSTS` entry with no port now
  admits port `80` only, exactly as an explicit `host:80` entry does. It used to admit every cleartext port on that
  host, so an operator who listed an internal service by bare hostname also opened `:22`, `:6379`, `:9200` and the
  rest of its TCP ports to the pipe, which the valve's own "exact match only" help text does not describe.
  **Migration:** if a listed service listens on a port other than 80, add a `host:port` entry for it (for example
  `internal.svc:6379`) — a bare entry does not cover that port and the link is refused until the port is named.
  `https://` links are unaffected: TLS is the control there and the port is the provider's business. The valve's
  help text, the dashboard's `Plaintext HTTP host allowlist` detail and the five refusal messages a refused attachment
  or picture produces all carry the same rule, and the refusal log now names the port instead of blaming a host the
  operator did allowlist.

- **An SVG a person attaches, a tool hands over, or the model wrote is not sent to the provider** — Open WebUI asks two questions before it turns a file into a model image block, is it an `image/` type and is it not `image/svg+xml`, and asks both at both places it does so. The pipe asked only the first, so a vector the interface had declined to draw was typed, converted and forwarded to a third party. The pipe now answers the same two questions, on the resolved media type rather than the declaration, and reports the refusal the way it reports every other picture refusal: the person sees `Images: skipped N (not identifiable as an image).` on the turn that carried it, the picture is not sent, and it is not counted against `MAX_INPUT_IMAGES_PER_REQUEST`. Every leg a picture travels on now runs the same rule: an inline attachment, a remote attachment the pipe downloads, an Open WebUI file, a picture reused from an earlier turn, a live tool round, the request sanitizer, and the `/chat/completions` fallback — the last two reach a gate that checks the scheme, the cleartext valve and the size cap and never resolved a type, which is why an SVG survived them while the same picture was refused everywhere else. Everything else is unchanged: a payload of any other `image/*` type still goes, including the BMP and TIFF spellings this gate has always forwarded under their declaration, and PNG bytes that declare themselves an SVG are still sent as the PNG they are. The picture the pipe *generates* is untouched — a stored SVG is still stored as `image/svg+xml`, and Open WebUI still draws it in the chat. What changes is the next turn: an SVG the model wrote is no longer lifted back onto a later request as a picture. No valve and no configuration change.

- **A file's name can no longer carry the pipe's own transport markers** — every other free-text field the pipe hands the provider has its hidden marker lines removed before the request goes out, and a caller-supplied `input_file.filename` was the one that did not: it was forwarded exactly as it arrived, on both block spellings and on either transport. The pipe's framing is line-based, so a name carrying `[P:final_answer]: #` or a ULID marker line on a line of its own was not an odd string but a second thing — a marker the pipe's own replay then read as its own, which split a reply into several messages and could rebind a stored artifact to a turn the caller does not own. The same line filter as everywhere else now runs on the name, with the same rule that a marker-shaped line only counts on a line of its own, so `[2026] report.pdf`, `quarterly [P:final_answer]: # report.pdf` and a name with an ordinary newline all survive byte for byte, and nothing is renamed or trimmed. A name that is nothing but marker lines loses the `filename` field and the file still goes out on its `file_id`, so a person never loses an attachment to a name. No valve and no configuration change.

- **Pipe Dashboard write gate** — on an install with `BYPASS_ADMIN_ACCESS_CONTROL=false`, a second administrator who is
  neither the dashboard model's owner nor a write-grantee is no longer refused the operator actions; and an
  administrator whose `Pipe Dashboard` model row was never inserted is no longer locked out of a dashboard the picker
  still lists. The write gate now answers Open WebUI's own model-write formula (`routers/models.py:888`, `:956`,
  `:1077`, `:1126`), whose admin term carries no valve. The **read** gate is unchanged and still honours the valve, so
  whether a second admin can *open* the dashboard is still that administrator's setting; only the actions he can run
  once it is open follow Open WebUI now. A model row whose read raises is still undeterminable rather than a verdict.

- **Two installed copies of the pipe no longer share one dashboard** — every dashboard binding is now keyed by the
  pipe's own `Pipe.id`: the socket handler's, the action route's and the publisher's registrations, the socket.io
  viewers room, and the model row a subscribe or an action authorises against. A worker serving two copies of the
  pipe therefore resolves each request against the install that named it instead of whichever copy registered last,
  and one installed copy's off switch, config save or valve event no longer reaches the other's viewers. The action
  route takes the install's id in the request body (`pipe`) and refuses an id this worker has no registration for with
  `404 {"error": "unknown pipe"}`, before any authorization read. **A dashboard panel persisted in a chat message
  before this change carries no such id**, so its Config and Update tabs answer `unknown pipe` until the panel is
  reopened; the panel says so in words rather than showing a raw refusal.

- **Gemini 2.5 thinking, hidden reasoning traces** — a request that carries `reasoning.exclude` of `true` no
  longer defeats the `GEMINI_THINKING_BUDGET` valve. `exclude` is OpenRouter's display preference ("the model
  will still use reasoning, but it won't be returned in the response"), so with `GEMINI_THINKING_BUDGET=0` the
  valve's off now holds against such a request and goes out as `{"effort": "none"}`, and a request's own
  `reasoning.max_tokens` is no longer dropped on that shape at a nonzero budget. On Gemini 2.5 Pro, whose
  reasoning is mandatory, the refusal status line naming the model now fires where it did not, and on any
  mandatory row the repair consumes only the keys it reads as an off, so a chat that hid the trace is not
  shown it. Every path a chat or a background task takes is covered, Open WebUI's tool-execution mode included.

- **Video intent classifier, a retired Task Model is now named rather than left to guess** -- a Task Model the host no longer publishes is already skipped before it is called, and the operator reading the warning had no way to tell that case from a row nobody filled in: both said `no usable task model under task.model.default / task.model.external`, so an admin who had renamed the model had nothing to match the row against. The drop is now recorded at DEBUG by the resolver, naming the id it dropped, and the warning says which case it is and names the id -- `the host no longer publishes 'ext-4o'` -- while an unset row keeps the wording it always had. No user-facing text changed, the warning's per-chat latch is unchanged, and no valve or configuration change. The documentation follows the code: the sentence that still offered a retired id a branch is amended, and the `VIDEO_INTENT_TASK_MODEL_FALLBACK` row now covers a configured row whose model is gone, as the valve's own dashboard detail already did.

- **A housekeeping task that recovers is heard about again the next time it fails** -- the task-failure toast is latched on `(model, chat and user)` so a long outage warns once rather than on every dispatch, and nothing ever cleared a latch: a title, tag or follow-up that failed at 09:00, answered again at 09:05 and failed at 10:00 was told about once, and the second outage stayed invisible until 300 unrelated keys pushed the first record out of the window. A successful task now removes its own chat's record, so a recovery re-arms the toast and the next failure warns again. The release is one key and not the table, so one chat's recovery never silences another's, and consecutive failures with no recovery between them still toast exactly once. A temporary chat is unchanged: it keeps no record, is toasted on every failing turn, and a successful turn writes nothing for it. The latch's shape, its 300-entry window and the anti-storm property for two overlapping failing tasks are all as they were.

- **`logit_bias`, the text an admin typed, is decoded instead of destroying the request** -- Open WebUI writes `logit_bias` as the free text its Advanced Parameters box holds (`"1234:100, 5678:-50"`), and the pipe's own outbound body types the field as a mapping. The string was refused by that body, so a single `logit_bias` in an admin's Task Model Params row -- or in a chat's own advanced parameters -- raised a validation error before any request left the process: the title, tag or reply silently became `### <U+26A0> Unexpected Error` / `Error type: ValidationError` for as long as the setting stayed. A `logit_bias` that arrives as a string is now converted to the value Open WebUI's own `convert_logit_bias_input_to_json` (`utils/misc.py:1129-1144`) produces, with each bias clamped to `[-100, 100]`; a value that is not a string is carried through unchanged; and one that does not parse is dropped from the request rather than forwarded or re-raised, with a warning naming it so the ignored setting is diagnosable. The conversion is the pipe's own, not an import of Open WebUI's, and it runs before the typed body is built, so it covers the ordinary chat turn, every housekeeping task and the video intent classifier alike. On `/chat/completions` the decoded mapping is forwarded to OpenRouter; `/responses` still drops the field, which is deliberate. No valve and no configuration change.

- **Video intent classifier, retired Task Models** — a Task Model the host no longer publishes is now skipped instead of called. When `request.app.state.MODELS` is a non-empty mapping, an id it does not contain is not a candidate, which is the same membership check Open WebUI applies to a task model itself (`utils/task.py:16-27`, `routers/tasks.py:145-150`, and `generate_chat_completion` raises `Model not found` at `utils/chat.py:191-193`). A retired Task Model therefore no longer costs up to four billed attempts per video turn, every one of them answered `Model not found`, and no longer arms the per-user classifier breaker on a configuration state: the turn lands on the already-built `no_task_model_candidates` path, which logs once per chat and opens no outage. An empty, absent or unreadable model list is not evidence and filters nothing, so a host that has not synced its model list still calls the model its admin configured. No chat-model fallback was added, and no new failure code, log line or latch.

- **Tool breaker's out-of-service notice** — the one line that names a tool as out of service is no longer
  lost silently, and no longer costs a round more than that round's own limits allow. Its bound was opened on the
  round's clock, so a round whose per-call announcements had already spent `TOOL_BATCH_TIMEOUT_SECONDS` started the
  notice on an expired deadline and dropped it; and it knew nothing about `TOOL_IDLE_TIMEOUT_SECONDS`, so a browser
  slow enough to cost the notice more than the whole idle allowance pushed the round past a limit the operator set.
  The bound now runs from the moment the notice itself runs and is also capped by whatever the idle allowance leaves,
  so the notice is inside the round's documented totals rather than beside them. When either bound ends it, a
  `WARNING` names the request and the valve that cut it — the loss used to leave no record at all — and the model is
  still told the reason for every call that was skipped either way. Nothing else about a round changes: one line per
  tool per round, an unset `TOOL_IDLE_TIMEOUT_SECONDS` still leaves the round to its batch ceiling, and a notice that
  fits inside the allowance is still delivered.

- **A refused Continue no longer loses the answer it was continuing** — a chat turn refused at the pipe's admission
  gate while it is being streamed now publishes the frames the same refusal publishes when it is not streamed. The
  refusal is published from the stream generator's own cleanup, and that generator's body does not run until the caller
  starts iterating it — by which time the chat id and the continuation prefix the frame shape is decided from had both
  been reset. On a `channel:` chat the reader was left with a bare terminal frame and no error card; on a **Continue**
  the closing frame carried the refusal as `content`, which the browser assigns absolutely, so `Server busy (503)`
  replaced the answer the person was continuing. Both now take the shape every other refusal takes: the card is
  published, the closing frame carries no replacing content on a continuing turn, and the answer stays where the
  browser had it. A saved chat that is not continuing, an API caller with no chat, a temporary chat and a temporary
  answer are all unchanged.

- **A skipped oversized image names the kind of file, not the caller's path segment** — a stored
  Open WebUI picture that is too large to inline is skipped, and the record that says so used to
  carry the file id the caller wrote in the `image_url` as its subject:
  `Skipping an attached image (01JQ8ABCDEFGHJKMNPQRSTVWXYZ01): larger than the 1048576-byte inline
  limit [cause=oversized_inline]`. That segment is the caller's own string — Open WebUI's file-id
  pattern admits any run of letters, digits and hyphens, so a person's name spelled as a path
  segment is as legal as an id — and a subject is not shortened on its way to the log. The record
  now reads `Skipping an attached image (image): …`, naming the kind of attachment the way its
  sibling arms already name `audio` and `video`. Everything an operator triages on is unchanged:
  the kind, the cause, the byte limit, the single WARNING per cause, the
  `Images: skipped 1 (larger than the 1048576-byte inline limit).` status the person sees, and the
  answer. A file that is genuinely gone is untouched — that refusal still names the id, because
  there the sentence goes to the person, not to the log.

- **A file a tool returned is inlined, not forwarded** — a `function_call_output` whose `output` is a list of
  `input_text` / `input_image` / `input_file` parts is now resolved instead of being flattened to a JSON string
  before the request is dispatched, so an `input_file` part naming `/api/v1/files/<id>/content` no longer reaches
  OpenRouter with an Open WebUI path, a host, a userinfo and a `?token=` on it. Two passes were involved and both
  are fixed: the sanitizer decided on part type whether a list was media, and the inliner's `output` arm dispatched
  pictures only. A part type nothing recognises is still flattened, and a caller who writes an Open WebUI path as
  plain prose is quoting it, not handing it over — the claim here is about structured parts.

- **A cancelled attachment read reclaims its copy** — `materialize_owui_file_to_temp` now settles the copy it
  shielded and unlinks the private temp it produced when the call is cancelled, including when it is cancelled
  repeatedly. A Stop used to land the turn while a worker kept writing a full private copy of somebody's attachment
  whose name nothing held, so the file outlived the call and the pipe paid for a copy nobody would ever read. A
  returned path is untouched and an `OSError` out of the copy still reaches the caller unchanged (TODO T637 is
  unchanged and still open: the temp a failed copy wrote is still left behind).

- **Two refusal cards stop appearing for block spellings that carry a real source** — a file block naming its document under `id`/`url` rather than `file_id`/`file_url`, and an `input_image` naming a `file_id` instead of a url, are now sent rather than skipped. Both reached `Files: skipped 1 (has no readable source).` and `Images: skipped 1 (an image carried no picture data).` respectively, so an operator watching refusal counts will see both drop for these spellings even though nothing else was refused. The file block is read through the same address gate, plaintext policy and storage-path rules as any other link (an `id`/`url` naming a private address or a cleartext link is refused exactly as `file_url` is), and the image block's id is forwarded as a file reference to the provider rather than resolved out of Open WebUI storage — it is the provider's own id, and it does not count against the per-request image cap. Open WebUI 0.11.4 itself writes neither spelling into a message, so on a stock deployment these counts do not move; they move where an upstream filter or a direct `/v1/responses` caller sends them.
- **A remote picture whose download the cap aborted is no longer forwarded as its link** — the downloader returned
  the same empty answer for every failure, so a transfer that `REMOTE_FILE_MAX_SIZE_MB` (or the admin's stored Max
  Upload Size, whichever is smaller) aborted on the way in was indistinguishable from a 404, and the inline leg fell
  through to the same rule as any other failed fetch: the link went out, and OpenRouter pulled down the bytes the pipe
  had just refused. The downloader now says which failure it was, and both the inline and the reuse legs refuse a
  cap-aborted transfer in the pipe's own words, `Images: skipped N (could not be fetched, so it was not sent).`
  Nothing else moves: a transfer that failed for any other reason — a 404, a timeout, a slow host — is still forwarded
  as its link on both legs, a link the pipe's own address check refused is still never forwarded, and `BASE64_MAX_SIZE_MB`
  is unchanged, so a picture is still refused for its size once downloaded with its byte count named. The reuse leg's
  warm half is unchanged too: a picture the pipe already held and now declines to send still reports
  `N bytes, over the M-byte download limit, so it was not sent`, naming the count it is giving up. (T1283)

- **Secret valve values, a passphrase typed with the `encrypted:` prefix** — a secret an operator types beginning
  `encrypted:` that is not already a stored row is now sealed like any other, instead of being written into the
  valve row as typed. `encrypt` short-circuited on the bare prefix, so on an install that leaves Open WebUI's
  `ENABLE_VALVE_ENCRYPTION` at its default — which is where the row is plain JSON — that value was the artifact
  encryption key and the session-log archive passphrase, sitting in `Function.valves` in the clear, while two help
  texts promised it was stored encrypted at rest. Nothing about the secret changes: the same value is read back and
  both consumers use exactly the text the operator typed, so the artifact table keeps its name and previously
  written archives still open. What changes is the row, and a value already in one is left alone: a sealed row this
  server can open, a damaged one, and one sealed under a retired `WEBUI_SECRET_KEY` all survive a Config-tab save
  byte-identically, because re-sealing any of them would either rewrite the row on every save or hand the operator
  the damaged blob back as their plaintext — which is what disarms the guard that tells them to re-enter the key.
  Both refusals that a stored value can arm now name the remedy alongside the two causes: re-enter the secret
  without the `encrypted:` prefix. An operator who stored a prefixed passphrase before this change keeps working; the
  row is rewritten as a ciphertext the next time the value is saved through either Config tab, and nothing has to be
  re-entered to make that happen.

- **Session log assembler, one offer per pass** — a turn the assembler already offered, or already failed, inside a
  pass is not offered again by that pass, and a turn whose assembly lock another pass holds is offered once per pass
  instead of once per re-listing round. The pass re-lists itself after meeting a lock-contended turn so the window
  keeps moving, but the exclusion it re-lists with was a snapshot taken before the loop, so a turn that failed in this
  pass — and a contended turn whose last failure was the stale-finalize seal, which the completed-turn listing's
  exemption had released — came back every round until the pass ran out its wall-clock budget, spending the whole
  `SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS` on SQL and never reaching the turns behind it. Nothing else changes: the
  backoff is still `SESSION_LOG_LOCK_STALE_SECONDS`, a turn whose only failure was the stale-finalize pass is still
  offered by the completed-turn listing straight away, and contention still books nothing.

- **Artifact encryption on a schema-qualified deployment** — on a deployment with `DATABASE_SCHEMA` set, the artifact store now finds its
  encryption key and stops writing rows in plaintext. It was reading Open WebUI's `function` row by an unqualified name, could not
  reach the row at all, and read that failure as "there is no unreadable row here" — so after a `WEBUI_SECRET_KEY` rotation the guard
  never armed and every artifact (reasoning traces, tool results, tool arguments) was stored in the clear with nothing said to
  anyone. The row is now read from the same schema the pipe's own artifact table is built in. The existing key-guard WARNING may
  appear there for the first time; follow it (re-enter the key) rather than silence it.

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

- **Tool pictures, the sanitizer's leg** — a picture link a tool result carries is now refused on
  every leg that carries one, plain HTTP included. The address check behind `ENABLE_SSRF_PROTECTION`
  resolves a tool result's link once per request and memoises the verdict, but the request sanitizer
  read that memo for `https` alone — so with `ALLOW_INSECURE_HTTP` on and the host allowlisted, a
  cleartext link whose address the check had refused was forwarded unexamined. Cleartext links now
  go through the same memo, and a check that reached no verdict inside its budget is refused in its
  own words (`could not be checked in time`, cause `uncheckable_tool_picture`) rather than being
  reported as a failed download, which is what the valve's help text already promised. An `https`
  link that resolves to a public address is unchanged, and so is every link a request never resolved.
- **Tool calling, a caller's own `strict` on a Responses-shaped tool** — a `strict` the caller wrote
  is now forwarded instead of being dropped. A tool already in Responses shape reached
  `/responses` through a branch that only read a `strict` out of a nested `function` block, so the
  key went out with the entry missing it, and that endpoint's own default for an absent `strict` is
  `true`: a caller's opt-in arrived non-strict and a caller's opt-out arrived strict. A tool that
  arrived without a `strict` still goes out without one, and a Chat-shaped tool still states the
  endpoint's fallback explicitly; a tool that arrives in Responses shape now keeps exactly the keys
  the caller sent, which is what Open WebUI's own converter does.

- **Video generation, status-poll errors** — `VIDEO_STATUS_POLL_MAX_ERRORS` now bounds two things instead of
  one. It still reads as "how many status checks **in a row** may come back as an error", and an endpoint
  failing every time in a row is given up on after exactly that many, with the same card and the same single
  circuit-breaker charge. It now also bounds the errors spent **in total across one whole watch**: a second,
  larger budget of the configured value times five, counted however the errors are spaced. Nothing about the
  valve's own meaning changed, and a stored value of it is honoured as stored — but the pipe can now stop
  watching a job whose status endpoint is failing intermittently rather than only one that is failing
  outright, which it could not before: any successful poll resets the consecutive tally, so an endpoint
  answering every other call never ran one out and the job was watched until it rendered or the silence
  window ran out. Nothing is cancelled either way. The card is the resumable still-running one, the job's
  marker stays, Continue Response still picks the same job back up, and the person is told how many times the
  endpoint did not answer across the watch rather than in a row.
  That budget is a **transport** budget, and it now says so on both help surfaces as well as in `docs/`. Only
  connection failures, timeouts and provider-reported statuses count against it. A fault of the pipe's own code
  escaping the status check -- a bug, or the `RuntimeError("Session is closed")` aiohttp raises on a closed
  session -- used to be counted as a flaky poll, retried the full budget over roughly 27 s of the user's backoff,
  and then reported as "OpenRouter's status endpoint did not answer this video's job 5 times in a row": a card
  naming a party that was never asked anything. Worse, it reached the failure card, whose
  `### Video generation failed` heading the resume path reads as final, so a job the user had already paid for
  was never polled again. Such a fault is now neither counted nor retried. It ends the watching on the first
  attempt, is named in the server log as the pipe's own with its traceback rather than blamed on OpenRouter, and
  leaves the same resumable card the transport case leaves -- marker kept, resume offered wherever the marker can
  be stored. It is still charged once against the request breaker, like any other failure after the job was
  submitted. No valve changed, and no real status-endpoint outage behaves differently.

- **OpenRouter Web Tools filter id** — a recurring "the OpenRouter Web Tools function id is held by a row this pipe
  does not own" warning now re-warns once 300 other causes have been armed, instead of going permanently silent after
  its first emission. The latch that backs it carries no timestamps, so the `cooldown_s` it was given was never read;
  it is now a window-bounded latch like the two per-model install latches beside it, and the eviction is what the
  repeat is keyed on. Nothing else about the filter, the id, or the repair changes.

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
- **Usage collection, a rotated `WEBUI_SECRET_KEY`** — when the stored valve row cannot be read (a `WEBUI_SECRET_KEY`
  rotation with an older worker still serving), the usage writer now refuses and says so in one latched WARNING naming
  the read. Until now it emptied the Usage tab with no pipe-side line. The refusal itself is unchanged — nothing is
  written, and the batch is dropped rather than held — and a row that is merely unset, or plain JSON from an install
  that never turned Open WebUI's valve encryption on, is still read as an ordinary off and does not warn.

- **Video generation, machine callers** — a video turn that fails reaches a caller with no chat as an HTTP error
  instead of a `200` with a Markdown card in it, whether OpenRouter rejected the submission or accepted it and
  the job then failed. A rejected job
  leaves with the status the pipe resolved on the status line and the same number in `error.code` (`502` when a
  proxy rewrote the `/videos` body); a fault the pipe owns leaves as `500` carrying `Video generation failed.`,
  with no Python class name in the body and the full message and traceback at ERROR in the session log as before.
  The two refusals the pipe raises on its own terms — the per-user cap and a missing prompt — still hand that
  caller their cards, unchanged; the gate here is a failure the upstream path produced, not every refusal.
  This now covers the failures that arrive **after** the submission was accepted as well: a video job OpenRouter took
  and then failed on the poll — a status response a proxy, CDN or WAF rewrote instead of OpenRouter's — used to
  reach the same chatless caller as a `200` carrying a Markdown card that quoted the first 200 characters of that
  body's own words, which is the body a provider's operator could put there. It now leaves as the same `502` with
  the same endpoint and `Content-Type` and no excerpt, both on the turn that submitted the job and on one that
  joins a job already running. Nothing else about the accounting moved: a rewritten body is still charged nothing
  against the breaker and still marks the turn failed, because the envelope is built after that bookkeeping
  rather than before it. On a channel chat the same card now withholds the excerpt as the rest of the pipe's cards
  already did — the job is watched in a background task that does not inherit the request's chat id, so the card
  reads the one it was handed.

  **Two carve-outs stay, and they are the same two the submit arm already had.** A **streamed** chatless caller and
  a caller on an Anthropic Messages path (`/api/v1/messages`, `/api/message`) still receive the card with the
  fenced excerpt, on this arm as on every other. A begun stream cannot carry a status line, and Open WebUI's
  Anthropic Messages handler re-wraps any streaming response into a converter with no error branch, so an
  envelope on either path would arrive as an empty `end_turn`. So a machine caller that streams, or that speaks
  the Anthropic format, still sees the proxy's own words: the statement above is about the unstreamed
  `/chat/completions` envelope. A prompt a provider echoes back is not masked either — the pipe cannot tell which
  bytes of a reply are its own request — but it is already withheld on a channel.

- **Provider routing panels** — a model's provider-routing filter is no longer detached from it when two model-list
  refreshes overlap. The pass that could not install the panel reported its verdict on a value shared with the pass that
  overlapped it, so a refusal to write could read as an answer and release the panel; the verdict now travels with the
  answer the pass obtained, and a pass that never obtained one leaves every model's filter list exactly as it was.

- **A model-list refresh that outlives its generation no longer writes** — when a hot reload, a valve save or a deleted
  function row retires a Pipe while its own `/api/models` refresh is still running, the refresh used to go on installing
  and repairing filter rows, running the startup stale-id prune, scheduling the metadata sync and dispatching
  `on_models` after that generation's `close()` had already returned — so two generations could write the same rows, and a
  dashboard could be re-pointed at an instance nothing was serving any more. It already returned its own cached rows
  rather than an empty picker; it now also re-checks after the catalogue load, so a close landing underneath a refresh
  that has already started skips the whole write region instead of only the part before it. A refresh on a live
  instance is unaffected, and nothing a person sees in a chat changes.
- **Media on the `/chat/completions` leg** — a turn resolved to the chat endpoint — by `DEFAULT_LLM_ENDPOINT=chat_completions`, by a model in `FORCE_CHAT_COMPLETIONS_MODELS`, or by the `AUTO_FALLBACK_CHAT_COMPLETIONS` fallback — now judges every media link its converter copies, by the same rule the `/responses` leg applies at ingress: the transport (`http://` only with `ALLOW_INSECURE_HTTP` **and** an allowlisted host), the scheme, a link naming this Open WebUI's own `/api/v1/files/` endpoint, and the inline size bound. Such a link used to be copied verbatim onto the wire, and the cleartext valve was never asked about it on this leg. A refused block is dropped from the turn and named on it as `[An attached item was not sent: …]`, in the request side's own words, beside whatever else survived — except an `input_file` that also carries a `file_id`, which keeps that id and drops only the refused link, exactly as on the other leg. An admin who allows cleartext HTTP sees the same link forwarded as before, and a `https://` link, a `data:` URL and raw base64 in `file_data` are untouched. One asymmetry is worth naming: a `data:` video sent to a chat-leg model is now held to the same size bound as on the other leg (`BASE64_MAX_SIZE_MB`, not `VIDEO_MAX_SIZE_MB`), so a clip over it is reported as not sent instead of forwarded.

- **A refused tool picture on the unstreamed `/chat/completions` route** — a picture a tool result carried that the chat-leg conversion drops (over `BASE64_MAX_SIZE_MB`, a cleartext `http://` link, an unusable scheme) is now named to the person on the turn that carried it. The streamed chat leg already did; nothing in the chain above the unstreamed leaf carried an event emitter, so the refusal was collected and dropped. The sentence is the same `Images: skipped N (<reason>).` every other leg uses, so which leg dropped it is not something a reader can tell. A turn that loses no picture still reports nothing, and a headless caller with no emitter still sends its request.
- **A refused picture no longer re-reports itself on every later turn** — under the default `Image input reuse`, a picture an earlier turn's gate had already refused was put into the reuse pool without being measured, so the reuse arm re-derived that refusal and named it again on each later turn for as long as `Image reuse window` could reach the pool — for a picture over `Maximum base64 upload size`, four turns' worth. The pool is now gated on admission, so a refused picture never enters it, is not re-measured on a later turn, and is not re-reported there; the refusal is logged at the reuse arm's own wording and the person still hears about it once, on the turn that carried the picture. Nothing else moved: a picture within the cap is still measured on the same terms at admission and is still reused on the same window, and `user_turn_only`, which never reaches the reuse arm, is unchanged.
- **A refused file is named on the turn that carried it and on no later turn** — the third disjunct of the notice gate (`or status_files`) was not keyed on the turn, so a refusal filed on any turn satisfied the emit condition by itself, on every later request that replayed that history: measured over a six-turn history, one `Files: skipped 1 (…)` status per turn, unbounded. For a caller with no `chat_id`/`message_id` the same unbounded set was joined into `choices[0].message.content`, so pipe prose sat in the value an automation reads as the model's answer, once per historical turn. The gate now carries a turn-window bound (`_NOTICE_LOOKBACK_TURNS`, keyed on the turn index, not on a message index), and the aggregated `attachment_notices` list rides the same predicate, so the body carries the same bounded lines the status did. The operator decided (N25, 2026-10-04) that the window is `0`: a refusal is reported on its own turn only, which is what the sentence in `docs/multimodal_ingestion_pipeline.md` now says. Nothing else moved — the report is not deleted, the turn that carried the attachment still names it, and a turn that refused nothing still says nothing.

- **Presets** — a request naming a preset in the spelling the pipe itself dispatches (`<base>@preset/<slug>`) is now served instead of being refused as blocked. The pipe publishes presets as `<base>:preset/<slug>` and dispatches them with the `@`; the model-restriction gate read the dispatched spelling as a model of its own, so an id copied out of a request log or a response body came back as the `Blocked model message`. The published row was already enforced under the picker spelling, so nothing new is admitted — the two spellings now name one model. A preset turn also stops being billed to the 128 000-token fallback and picks up its base model's real context window, so a tool result the base's window can hold is no longer trimmed away and one it cannot is no longer shipped whole.
- **image generation, a rewritten response body** — when something in front of OpenRouter answers the image endpoint with an error page or a WAF challenge instead of an OpenRouter response, the failure card now shows that text as a quoted code block instead of as ordinary message text. The provider's words are still on the card in full — nothing is shortened, rewritten or hidden — but they are contained, so a beacon embedded in the page no longer makes the reader's browser fetch it and a phishing link in it is no longer a live link. The two facts the card leads with, the endpoint that answered and the upstream `Content-Type`, are unchanged and remain ordinary prose. The fence is carried by the value rather than by a placeholder, so it holds on an installation that has never opened a template, and it is sized past the payload's own backtick run, so a page carrying its own fence cannot escape one. This affects every caller of a picture-only image model, on a surface that is on by default, and it is the same rule the text legs' error cards already applied.
- **Image generation, caller with no chat** — an image request that fails on a provider rejection, or on a body a proxy rewrote, now answers a caller with no `chat_id` **and** `message_id` with an HTTP error envelope instead of a Markdown card, carrying the resolved upstream status on the status line and in `code` (`502` for a rewritten body, on both surfaces). A caller in a chat keeps its card on the same faults. A fault the pipe owns (a blank prompt, a refused caption, storage refusing the file) still reaches every caller as readable text. A streamed image turn now always delivers its card, so a chatless caller is never handed an empty stream; see [Error handling](docs/error_handling_and_user_experience.md).
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
- **Endpoint valves** — a `FORCE_CHAT_COMPLETIONS_MODELS` or `FORCE_RESPONSES_MODELS` entry that names a base model now also
  matches that model's suffixed spellings: the `:free`, `:thinking`, `:nitro`, `:exacto`, `:floor` and `:online` tags, dated
  forms, and any combination of the two. A pattern that names a tag or a date stamp still matches only that spelling, and a
  `:preset/...` tag is left alone. The valve previously built two candidate keys per request and matched both over the whole
  string, so a base entry silently missed its own `:free` twin; a widened pin means those rows are no longer served, or
  retried, on the endpoint the pin excluded.
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

- **Reasoning summary blocks** — on a `/chat/completions` turn carrying two or more `reasoning.summary`
  blocks in one round, the thinking box and the status line now read them as separate blocks separated by one
  space rather than as one glued word (`Name the tradeoff.Then weigh it.`). The blocks are joined the same way
  by every producer the loop folds together — both chat-completions arms, the non-streamed chat leg and the
  `output_item.done` snapshot — so the round still reaches the person once, as two blocks. A block's own text is
  untouched: a block that already ends in a space does not gain a second one, and internal whitespace is left
  exactly as the model wrote it. The stored reasoning row and the closing item are unchanged.
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
  instead of scaling with the source's short side. A geometry the pipe cannot measure at all — a video line carrying no
  size, a header read that fails, or zeros arriving from the probe rather than from a second read — is now treated as
  unbounded and bounded at the decoder anyway, instead of turning the scale filter off and decoding the whole frame at
  native size first; the filter is `min(1920, iw)`, which cannot invent pixels, so applying it to a source of unknown
  size cannot make that source look smaller than it is. The ffmpeg arm's own pre-child gate is gone: it compared the
  scaled size the header predicted against the 25-megapixel budget, a comparison that could never fire at the shipped
  values (the largest product it could reach is 1920×1920, 3.7 Mpx, against a 25 Mpx cap), so what it cost was a
  container-header read on every extraction whose probe had measured no geometry. The pre-decode refusal and the
  decoded-size checks behind it stay in the code for a header that lies about the source.

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

- **API callers, streamed** — a 502 that used to reach a streamed API caller as assistant text now reaches it
  as an in-band error frame. An API caller with no chat to write a card into used to receive the error card as
  the model's own answer on a streamed call: an HTTP 200 whose body was Markdown, indistinguishable from a
  reply, with no status to branch on. A streamed turn cannot carry an HTTP status at all — Open WebUI hands a
  pipe's non-2xx back inside a response with no `status_code`, so it would arrive as a bodyless 200 — so the
  failure is now delivered as a terminal `{"error": {"message": …, "code": …}}` frame carrying the upstream
  status, byte-for-byte the envelope the non-streamed leg already sent. Chats, temporary chats, internal Fusion
  members and both Anthropic Messages endpoints keep their card exactly as before.

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
  affected, and on a channel chat it is now withheld along with the rest of the provider's prose about the request: a
  card that reaches a room renders it as though empty, so a **Provider error** row guarded by `{{#if}}` is left out
  there, and an unguarded one omits its whole line. A saved chat, a temporary chat and an API caller are unchanged.
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
  *differently* warns again at WARNING. The whole-pass lines are a separate latch, and the image one has
  now moved with it: `OpenRouter Image filter ensure failed` warns once per cause — per exception class,
  for as long as the fault lasts — and every later failure of the same kind is recorded at `DEBUG`, where
  the Fusion, Web Tools, Image Gen and Direct Uploads lines already warned that way.
- **`Files` boxes** — the pipe's own untick of the `Files` box inside `Built-in tools` is now recorded on
  the model, under its `openrouter_pipe` metadata as `builtin_tool_defaults`, and it is unticked only on a
  pass where the pipe unticked `File context` itself. A `Files` box you unticked by hand is no longer changed
  as a side effect of that, and is never recorded as the pipe's. A model row written by an earlier release
  carries no such record and keeps the box it has; an older pipe ignores the unknown key.
- **temporary chats in the log** — a temporary chat's browser socket id is no longer written to the pipe's
  log on the Direct Uploads path or on the dropped-row path; both records now read `<not retained>` in its
  place, the same literal the video-intent path already used. The artifact-replay records join them, so the
  pipe has one spelling for a withheld chat id rather than two. A saved chat's id is still named in full on
  every one of them, and the record's counts and row fields are unchanged.
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
- **session log storage, an over-budget id with no task name** — a caller-supplied message id longer than
  the 64 characters the archive key is written to is now reduced on the **answer** archive too, not only on a
  task archive. The answer arm used to return the id verbatim, so on PostgreSQL — where that column is
  `String(64)` and the row is refused outright — the turn's staging write failed and fell through to the
  queued-zip fallback, which composes a still longer name, so the condition persisted for that turn. The
  reduction is the same one a task key already got: the stem the budget allows plus a short digest of the
  exact id, never a cut, so two ids that differ anywhere still reach two files. Every id Open WebUI mints is
  a `str(uuid4())` at 36 characters, so no browser-path archive changes; a caller-supplied id past the budget
  changes its key and its filename (and, until the exact id is carried in `meta.json`, its `ids.message_id`
  too), and an archive already written under the raw id is found and merged into the digest-named one.
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

- **Video attachments and the file host** — three changes to what a turn says and does when an attachment goes to a
   public file host. A **reference picture** over `MEDIA_FILE_HOST_MAX_SIZE_MB` — over the cap on its own, or over what
   one request may publish put together — is now left out with a notice naming it and the cap it broke, and the rest of
   the turn is generated as asked, instead of refusing the whole request; a **clip or a sound file** over the same cap
   still stops the request, naming the cap, because a clip the person attached is the thing being edited. The durable
   record's **"may have been uploaded"** line now speaks for each attachment that reached each host rather than being
   suppressed whenever another attachment of the same kind landed on the same host, so an ordinary two-clip turn records
   both, in one block; and it names that host's own retention, hedged on the file being there at all, instead of
   promising permanence on a host that deletes its own files. The **second-host failure card** no longer says "Nothing
   was uploaded" when an earlier attachment in the same turn already reached a host: it says that nothing further was
   uploaded, that an earlier attachment is already on a public host, and that sending the turn again will not take that
   one down. All four need the opt-ins below, which ship off.
- **Tool shutdown at `TOOL_SHUTDOWN_TIMEOUT_SECONDS=0`** — the per-request `WARNING` an operator who chose that
  documented quiet value saw on every turn stops appearing. A tool context is built for every request, whether or not
  the turn called a tool, and shut down at the end of the job, so the valve's own "there is no grace wait" path was
  the one path that always fell through into the wait-timed-out handler and wrote `Tool shutdown exceeded 0.0s;
  cancelling workers.` — a line describing a stall that could not have happened, on precisely the installations where
  cleanup was instant and deliberate. Nothing was waited for before either, and nothing changes now: the workers are
  still cancelled at the same instant, the reply is unchanged, and log-based alerting on `WARNING` stops firing on a
  configuration that is working as documented. The valve is now a positive guard rather than a synthesised timeout, so
  a wait that genuinely expires still warns exactly once and still names the valve's own value; and a DEBUG line
  records that the grace period was off, which is what tells an operator why the workers died instantly (genuinely
  abandoned tool tasks are left behind on this path).
- **Empty provider-error cards** — the card a reader gets when the operator's own template renders to nothing now
  names their configured support contact, on a `channel:` read as well as an ordinary one. The stub that replaces an
  empty card named the model and an error id but no way to reach a human, so on exactly the path where an admin's
  wording had been withheld from a channel reader, the card they were left holding had no support handle either. The
  two rows appear only when `Support email` or `Support link` is set, in the same guarded shape the shipped templates
  already use; with neither set — the shipped default — the card renders exactly as before, byte for byte, and the
  shipped 400 card is untouched. A non-streaming API caller reads the same text off `choices[0].message.content`, so
  they gain the same handle in the same string.

- **Model metadata, a stored grant the access field cannot spell** — on an Open WebUI whose `ModelForm` carries
  `access_control` instead of `access_grants`, a metadata refresh used to drop every stored grant whose principal type
  is neither `user` nor `group` (`anyone` today — "anyone, including people who are not logged in, may read this
  model"), and a wildcard read took every other grant in the set down with it on the way out: the converter answered
  the public wildcard before it read the rest, so `user:* read` plus `user:alice write` came back with neither. Both
  answers the dict can give move the row, which is why neither is used. `{}` replaces a shared model with a private
  one; `None` is Open WebUI's own public wildcard (`user:* read`), so a grant that conferred read on nobody would
  confer it on every signed-in session instead. The pipe now refuses the write and says so. The row keeps exactly the
  grants its admin set, its icon and its description stop being refreshed, one `WARNING` per row names the model and
  the principal type it could not carry, and on a sync pass the refusal is counted with the pass's other unwritten
  models — that count is the `WARNING` line to look at, with the per-model detail at `DEBUG`. Nothing changes on a
  stock install: Open WebUI 0.11.4's `ModelForm` carries `access_grants`, so the conversion never runs there. Where it
  does run, a wildcard read keeps the rest of its grant set instead of discarding it, and a set the dict can spell
  still round-trips unchanged.
- **Video intent, disclosure on long chats** — a chat past the classifier's window (24 conversation rows or 9 prior
  videos) no longer opens the Intent Disclosure Block on its own. Those two window notes change what the classifier
  *read*, not what was *sent*, and on a turn that lost nothing they were putting a full block on screen for a note
  about nothing that was dropped. People see fewer disclosure blocks on long chats, and the loss is disclosure: the
  note about the classifier reading only part of the chat still appears whenever the block shows for another reason —
  a rewritten prompt, a reused prior-video frame, or any recorded loss — and always under
  `VIDEO_INTENT_CONFIRM_MODE='always'`. A turn that *did* lose something opens the block exactly as before, and
  `VIDEO_INTENT_CONFIRM_MODE='never'` still renders nothing at all.

- **A `data:` header that is not a media type names nothing, on either reader** — both of the pipe's header readers took
  everything between `data:` and the URL's first comma, cut it at its first `;`, and printed up to 64 characters of it.
  A 64-character truncation is not a validation, and the header is the caller's own text: a remote server answering
  `Content-Type: Q4 Payroll - Margit Okonkwo` is enough to put that text in the header, because Open WebUI interpolates
  a remote server's raw `Content-Type` into the data URL it builds (`backend/open_webui/utils/files.py:96-97`). Both
  readers now answer through the pipe's own `media_type_or_empty` — the same predicate the read path already used, which
  accepts a header only if it really is `type/subtype` — so a header that names no media type contributes nothing
  rather than a prefix of itself. **This changes one thing on the chat leg, deliberately:** an inline `input_audio` block
  whose `data:` header is not a media type is no longer refused as "a format the pipe will not rename" (a refusal about
  the caller's spelling rather than about a format); it now declares nothing, so the block's own `format` decides, and
  a block with no `format` at all keeps the `mp3` default — which is what the tree already did for an *empty* header.
  A well-formed type is unchanged, and a media type that was echoed verbatim is now lower-cased (`data:IMAGE/PNG` reads
  `data:image/png`), which is RFC 2045 case-insensitivity and the same normalised form the read path and the free-text
  log leg already return. In a log a refusal subject for such a URL now reads `data:` — the scheme is known and named,
  the payload is not — and a file refusal with no other source reads `no source`, as it already does when nothing else
  can describe the link. No valve and no configuration change.

- **Image generation, the retired size spelling** — an image request that carries the output size under the pipe's own retired spelling `image_size` no longer fails with a 400 when the model's published settings cannot be read. `image_size` is kept alive for valve rows written by a filter that has since been deactivated; it was aliased onto `resolution`, which the OpenRouter schema documents as a closed enum of `512`, `1K`, `2K` and `4K`, so a pixel value written that way was refused by the whole generation. The retired spelling now meets the same gate as the modern one, `size`, whether the contract is readable, unreadable or a model that publishes a tier list of its own, and both spellings produce the same key, the same value and the same note. One behaviour does change and is worth stating: `image_size: "banana"` used to go out unchecked on an unreadable contract and is now refused as outside the contract, naming `size`, exactly as it already was where the contract could be read.
- **Open-WebUI-owned tools** — a round only Open WebUI can run is handed back to Open WebUI on every turn, whatever
  `MAX_FUNCTION_CALL_LOOPS` says. The registry entry under such a name is a marker recording who owns the tool, and
  the pipe read its presence as "the pipe can run this"; at the cap that made the pipe answer a call Open WebUI was about
  to run, and for a browser-run tool server, run it itself inside the pipe over Open WebUI's own Socket.IO bridge. The
  three provenances that reach the hand-back are a browser-run tool server, Open WebUI's own builtins — which carry a
  callable the pipe must not use — and a name the request withheld from the model before it ever saw it, under `ask`
  approval in a streamed saved chat or under `function_calling: "legacy"`. A reply that keeps asking for such a tool now
  goes back every time and is bounded by Open WebUI's own `CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS` — which means such a
  reply runs longer and consumes more of the operator's Open WebUI iteration budget than it used to. The skipped-call
  card the pipe emits for a tool Open WebUI owns now names the reason that is true of the request it came from: the
  spent-budget wording only where a hand-back budget really was charged and spent, which is the round that mixes an owned
  call with one the pipe can run, at the cap; and a wording that names no budget at all on a reply that could not be
  handed back, because it is not streamed, so no hand-back could have carried the round back and no turn of one was ever
  spent. A round that mixes an owned call with a call nobody can run still falls to the pipe's own cap, because a call
  with nothing behind it is not Open WebUI's to run either.
