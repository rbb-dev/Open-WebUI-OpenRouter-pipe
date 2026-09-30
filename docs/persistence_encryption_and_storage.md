# Persistence, Encryption & Storage

This document explains how the pipe persists response artifacts (reasoning payloads and tool outputs), how artifact replay works across turns, and which valves control encryption, compression, Redis caching, and retention.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Security](security_and_encryption.md) · [History/Replay](history_reconstruction_and_context.md) · [Concurrency](concurrency_controls_and_resilience.md)

---

## What gets persisted (high level)

During a chat, the pipe can persist structured “artifacts” so future turns can replay prior context without embedding large JSON blobs directly into the visible chat transcript.

Persisted artifacts include (at least):
- Reasoning items (as long as `PERSIST_REASONING_TOKENS` permits retention).
- The pipe's own copy of each tool round (see [History Reconstruction & Context Replay](history_reconstruction_and_context.md),
  which lists the rounds that are stored differently or not at all):
  the call and its output, with the full arguments and result, pictures included, whatever `PERSIST_TOOL_RESULTS`
  says. That setting decides what later turns receive; tool results are stored even while it is off. They are
  encrypted at rest only when `ARTIFACT_ENCRYPTION_KEY` is set and `ENCRYPT_ALL` is on at write time; a row written
  while it was on stays encrypted.

The pipe never stores a copy of a picture or file a person attaches, in any chat.

A temporary chat keeps none of its content in the pipe's storage. Open WebUI keeps it only in the browser, and the pipe stores none of it (database, disk or file storage): no tool round, no reasoning and no session log, no picture bytes in the worker's memory - a picture a temporary chat reuses is fetched again on each request that needs it rather than cached in the process - and no chat id in any process-lifetime structure. That last one is three once-per-chat warning latches — the failure toast, the no-usable-Task-Model warning and the housekeeping task-failure toast — a temporary chat is warned on every failing turn rather than once in any of the three, precisely because the latch that would silence the second turn is never written for it, and its `chat_key` reads `<not retained>` in the log lines that name it (see [Video Intent Classifier](openrouter_video_intent_classifier.md)). The hand-back budget is a second such structure, and it holds no chat id for a temporary chat either: the reply is charged and its budget spent under a key whose chat-id slot is blank, so the socket id is in neither the count map nor the last-seen map, while the budget itself is unchanged and still per-reply. The classifier's per-chat cost counter is inside the boundary too: with `VIDEO_INTENT_MAX_TURNS_PER_CHAT` above `0` — the per-chat cost cap, off by default — each admitted turn is charged under a one-way SHA-256 prefix of the chat id rather than the id itself, temporary chats included, and the tally keeps only the most recent 300 chats, so no chat id is held there and a temporary chat's share of the cap is charged exactly as a saved chat's is. When usage collection (`PIPE_DASHBOARD_USAGE_COLLECT`) or cost snapshots (`COSTS_REDIS_DUMP`) are switched on (both off by default), a temporary chat's records keep every figure but name no chat: its usage row leaves the chat and session ids empty, and its cost snapshot carries no chat or message id. Saved and channel chats keep those ids. The same boundary holds on the wire: a temporary chat's chat, session and message ids are never placed in the request `metadata` sent to OpenRouter, whatever the identifier valves are set to (see [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md)). While an admin has the dashboard open on a deployment that uses Redis, a temporary chat's live row passes through Redis (rewritten every 2 seconds, each entry expiring after 10) under an anonymous stand-in keyed by `WEBUI_SECRET_KEY`, never under its chat id, its socket id or the key sent to OpenRouter, and with no id at all when that secret is unset. Three things are kept the way Open WebUI keeps them:

- On the chat route, a picture a model generates is saved to Open WebUI's file storage and linked to the message in a saved chat; in a temporary, legacy-temporary or channel chat, which has no row to hold the link, it is kept inline in the message and nothing is stored. This is stricter than Open WebUI's own chat path, which uploads in a temporary chat and only declines the link (`.external/open-webui/backend/open_webui/routers/images.py:520`); a generated video still goes to the file store on the direct image and video routes, which have no inline form to fall back to.
- Pictures and audio an MCP tool returns are saved as files by Open WebUI's own tool handling, as it does in any chat. So is a *non-image* file a tool returns -- a PDF, a CSV, an archive, a clip -- in a saved chat, a channel and a plain API call with no chat at all: the pipe files it through Open WebUI's own storage and links it, so the person gets a document they can open and the next turn replays a link rather than the bytes. In a temporary chat or a `local:` chat that file is dropped instead, and the person is told once, because those chats keep their content in the browser and a server-side file would be one nothing links and nothing cleans up. This is stricter than Open WebUI, which emits a non-image tool file into the chat verbatim with no chat-id check at all (`middleware.py:3483-3489`); on no path does a `data:` payload reach a stored row, a `files` event or the request the provider receives.
- A streamed reply that may hand a call back -- in Open-WebUI mode, or in Pipeline mode for a tool the pipe cannot run -- holds its rounds and thinking in memory for that reply only, dropped when the pipe answers its last call back, or when the provider refuses a call-back the pipe was waiting for, or when the reply is stopped, or after 15 minutes unused. A reply that cannot hand one back holds nothing and writes nothing. The rounds and thinking of the reply being written are held in the pipe's memory, never on disk or in the database, until that reply ends, and only for the user who opened that reply, so those calls hand the model the same turn a saved chat's would. A reply left idle for 15 minutes is dropped. Once the held rounds pass 64 MiB in one Open WebUI worker, the longest-idle replies are dropped first, and for a single reply larger than that, the pipe drops everything held for it so far. For a dropped reply, the model continues without the thinking and rounds held for it, and whenever the pipe drops a reply under either of those two rules -- the single reply over the ceiling, or the pool's oldest-held reply once the total passes it -- the worker logs one warning naming the ceiling and how many replies it dropped, with repeats inside the next five minutes at debug level.

A call that carries no `chat_id` -- the plain API route, where Open WebUI supplies none -- is a separate case, not the temporary chat's twin. Its reasoning and tool rounds are held in the pipe's memory for the length of that one request, keyed on the request id, and are never written to the database; the request ends and they are dropped, so they never reach a later turn and no cleanup has anything to remove. The same 15-minute idle and 64 MiB limits apply, in a pool of their own, so a call cannot evict a chat user's in-flight held reply and vice versa, and the total memory ceiling is twice one pool. No marker line is added to the caller's response, so a program's bytes in and bytes out are unchanged. `API_CALL_ARTIFACT_MEMORY` turns the hold off, in which case the call simply stores nothing -- as it did before. A caller sending `parent_id: null` is given a real chat id by Open WebUI, so that shape is not one of these calls.

Rows an earlier release stored for a temporary chat are deleted at the next cleanup, whatever their age. Usage rows it wrote for a temporary chat are kept, with the chat and session ids cleared at the next usage purge, which runs only while usage collection is on; cost snapshots it wrote are not rewritten and expire on their own.

**Note:** Not every artifact type is replayed verbatim. The pipe filters certain tool artifact types to avoid wasting context window and to reduce provider-side errors.

---

## How replay works (ULID markers)

Instead of embedding full artifact payloads into the assistant’s visible text, the pipe appends hidden marker lines that reference persisted artifacts by ID.

Marker format:
- Each marker line contains a 20-character ULID-like identifier and a fixed suffix.
- Example marker line:

```text
[01J2VVZBDQTP1ZQJ8DSN]: #
```

Two other families of line are also hidden markers, and both are stripped from what the provider is sent:
`[P:<phase>]: #` (the phase label) and `[openrouter:v1:<kind>:<body>]: #` (the pipe's own transport lines, used by
the video intent and media relay disclosures). Only the ULID form above is an **artifact reference** — the one this
section is about, and the only one that is looked up and replayed. The kind form is a **transport line**: it carries
state within a single message and is never resolved against the artifact store.

On subsequent turns, the pipe scans prior assistant messages for marker lines, fetches the referenced artifacts in one batched pass per turn (from Redis cache if available, otherwise the database), and replays them into the next request in a structured form. Markers are emitted for chat turns only: a call that carries no `chat_id` adds none to its response, so a program calling the API sees exactly the bytes the model produced.

---

## Database storage model and table naming

### Table naming
The pipe stores artifacts in a per-pipe SQL table whose name includes:

- a sanitized fragment derived from the pipe/function ID, and
- a short hash derived from `(ARTIFACT_ENCRYPTION_KEY + pipe_identifier)`.

Pattern:

```text
response_items_{pipe_fragment}_{hash8}
```

Operational implications:
- Multiple pipe instances (different function IDs) do not collide.
- Each table carries a composite index on `(item_type, created_at)`, which is what the assembler’s stale and terminal passes query. Its name is derived from the table name, so it is unique per pipe and short enough for every supported dialect’s identifier limit.
- Changing `ARTIFACT_ENCRYPTION_KEY` causes the pipe to write to a different table name.
  - Existing artifacts are not “deleted” automatically, but they will not be read by the pipe unless you restore the prior key (and therefore the prior table name).
  - The cipher is rebuilt against the current key on every call, so a rotation never leaves the store using a retired one.
  - A row still buffered in the Redis pending queue when the key changes is dropped at the next flush with a warning naming its `item_type`, rather than being written into the new table where it could never be read.

### Stored columns (high level)
Persisted rows include:
- `id` (the ULID)
- Open WebUI identifiers such as `chat_id` and `message_id`
- `item_type`
- `payload` (plaintext JSON, or an encrypted wrapper when encryption is enabled; the wrapper carries the ciphertext and a version field, and the compression flag is inside the ciphertext, not in the column)
- `created_at` (UTC timestamp used for retention; refreshed on every read, database or cache)

---

## Encryption and compression behavior

Artifact encryption is controlled by these system valves:

- `ARTIFACT_ENCRYPTION_KEY` (enables encryption when non-empty)
- `ENCRYPT_ALL` (default `True`; controls “encrypt everything” vs “encrypt reasoning only”)
- `ENABLE_LZ4_COMPRESSION` and `MIN_COMPRESS_BYTES` (optional compression for persisted payloads)

### When encryption is active
- If `ARTIFACT_ENCRYPTION_KEY` is set (non-empty), the pipe encrypts payloads before persistence.
- When encrypted, the stored `payload` becomes a wrapper containing ciphertext (plus a version field). A row written by a build from before the wrapper existed holds the bare ciphertext as a plain string in that column; when such a row is written again, its ciphertext is kept as it is rather than re-encrypted, and it keeps reading back to the plaintext that went in.
- The seal is decided by the row's stored form, never by a key inside the payload. A row read back from the table is re-sealed for the replay cache on the strength of the `is_encrypted` flag the fetch carries out with it, and a payload carrying its own top-level `ciphertext` — a provider's opaque reasoning block, say, which the pipe passes through with unknown keys intact — is re-sealed like any other and replays to the same value.
- If compression is enabled and effective, the pipe compresses the JSON payload before encryption; the plaintext then begins with a one-byte flag saying whether the bytes after it are compressed. The flag lives inside the encrypted plaintext, not in the `payload` column. A payload written before the flag existed still decodes: a first byte of `0` or `1` is a flag, a JSON lead byte (`{`, `[`, space, tab, CR, LF) means the whole body is headerless, and any other first byte is refused as an unknown flag.

### When encryption is not active
- If `ARTIFACT_ENCRYPTION_KEY` is empty/unset, payloads are stored as normal JSON objects and `is_encrypted=False`. There are three ways to be in that state and they are told apart by the stored row, not by the valve: a site that never set a key has no ciphertext to read and keeps writing plaintext; a key that is set and readable encrypts; and a key that is set and **not** readable under the current `WEBUI_SECRET_KEY` (a rotation) is refused. In that last case the pipe cannot evidence "no artifact key is configured" from anything it was handed -- Open WebUI answers a valve row it cannot decrypt with an empty one -- so it reads the raw `Function.valves` column itself, and a row that will not open arms the guard: artifact content is written to a table of its own, nothing is stored in the clear, and the operator is warned once. A read that fails or a host with no such row leaves the guard unarmed, so a deployment that never set a key is never stopped by it.

**Operator note:** If you store secret valve values using Open WebUI encryption (`EncryptedStr`), configure `WEBUI_SECRET_KEY` (see [Security & Encryption](security_and_encryption.md)). If `WEBUI_SECRET_KEY` is changed later, encrypted valve values may decrypt differently, which effectively behaves like a key rotation for artifact storage; a stored row damaged after it was written is refused the same way, with the same warning. The cipher is rebuilt against the current key on every call, so such a rotation never leaves the store using a retired one. Rows still buffered in the Redis pending queue at that moment are discarded unreadable; draining the queue before rotating avoids it. A rotation of `WEBUI_SECRET_KEY` that leaves the stored `ARTIFACT_ENCRYPTION_KEY` unreadable also **stops artifact writes**: the pipe sees a row it cannot open, warns once, and drops what it would have written rather than storing it in the clear, until the correct `WEBUI_SECRET_KEY` or the key itself is restored. The coordination lock rows are still written, since they carry no content.

---

## Redis cache and write-behind (multi-worker deployments)

The pipe has an optional Redis-backed cache/write-behind path intended for multi-worker deployments.

### When Redis is used

`ENABLE_REDIS_CACHE` enables Redis support, but Redis is only used when the runtime environment indicates a multi-worker Open WebUI deployment and Redis tooling is available. The switch and the cache lifetime are re-read on every operation, so turning the switch off takes effect without a restart, and turning it on again does too: the request path reconnects to Redis and brings the cache back up on the next message, with no restart. A connection attempt that fails is retried on an interval rather than on every request: 300 seconds, a fixed constant and not a valve, and neither turning the switch off and on again nor a changed `REDIS_URL` makes the operator wait it out. In particular, the pipe requires:

- `UVICORN_WORKERS > 1` (multi-worker mode), and
- `REDIS_URL` is set, and
- `WEBSOCKET_MANAGER=redis` and `WEBSOCKET_REDIS_URL` are set (Open WebUI websocket/Redis configuration), and
- the Python Redis client dependency is available at runtime.

If these prerequisites are not met, the pipe runs without Redis and persists artifacts directly to the database (when persistence is enabled).

The master switch is read at the moment a Redis gate is reached, not frozen at start-up, so a change saved in the admin UI takes effect on the next request that reaches one without a restart. A change to `False` stops new connections immediately. A client that is already running keeps *serving reads* while the switch is off, and that is deliberate: a row still buffered in the pending queue has the cache as its only copy until the drain commits it, so a read gated on the switch would lose it from the model's context for good. Nothing is *written* to Redis while the switch is off — no artifact data reaches the socket, every data write path is behind the valve, so the switching turn's own row and any later turn's row go straight to the database. What the switch does not stop is the removal side of a delete, because a delete stores nothing: it removes a copy the pipe itself put there. A cleanup that runs while the switch is off still writes its delete marker and still deletes the cache entries of the rows it was told to forget, and so does a flush that finds a row it committed that a marker says was deleted while it was queued — that is the promise `PERSIST_REASONING_TOKENS="next_reply"` makes about the artifacts themselves, and a stale entry left behind is exactly what a later turn would read back through the cache. The retention sweep's invalidation of the cache entries of the rows it expires is the same kind of exception, for the same reason: it deletes a copy the pipe put there, and a stale entry is what a later turn would read back. The drain's own two keys, the pending queue and the flush lock, are the one exception: it touches them only to empty the queue. The read side is valve-blind on purpose; a "fix" that gates it on the switch loses buffered artifacts.

High-level behavior:
- When Redis caching is enabled and available, the pipe can enqueue persisted rows into Redis and flush them to the database asynchronously.
- When Redis is enabled, the pipe can also cache persisted artifacts for faster replay reads. A cached artifact keeps the form its table row has, so a cache read is never a weaker read than a database read — except for an artifact cached by a build before this one, which keeps its previous form until the entry expires within `REDIS_CACHE_TTL_SECONDS`, and except for a clear artifact whose own payload carries a top-level `ciphertext` field: a pure cache read can no longer mistake that field for a seal, but a read through the composite path still returns the artifact from the table. An artifact the table holds sealed is re-sealed for the cache from the flag the fetch carries, so a payload that carries its own `ciphertext` is sealed like any other and replays to the value it was written from. The table side has the same upgrade path for the encrypted form: a row stored before the ciphertext wrapper existed is preserved, not re-encrypted, when it is written again.
- When Redis write-behind is active, a row deleted while it is still queued is removed from the table and its cache entry, so it cannot be replayed from either. The marker carrying that decision is the one Redis lifetime deliberately not tied to `REDIS_CACHE_TTL_SECONDS`: it lives in Redis for its own fixed lifetime of 24 hours, because the write-behind queue it has to outlive has no deadline at all, and a row still queued when its marker expired would be written to the table and re-cached, losing the delete permanently — which is what a queue backlogged past that window means. The `{ns}:deleted:{row}` marker value is namespaced, and the reader decodes it rather than comparing it to the row's own `message_id`: a keep writes `owui:dropped:keep:{message_id}` and a cleanup that spared nothing writes `owui:dropped:all`. The key layout and the marker lifetime are unchanged. A `message_id` is a caller-supplied string, so an unnamespaced marker was ambiguous with it: a no-keep cleanup whose marker was a bare `"1"` spared every row the caller happened to own with that id, and the flush re-cached them, so the delete was lost permanently rather than delayed. A marker written before this change is still read: the bare `"1"` still drops every row, and a bare `message_id` still spares its own row. A worker from before this change reads the new values as before — `"owui:dropped:all"` as truthy (so it still drops, which is right) and `owui:dropped:keep:{id}` as a spared `message_id` that matches nothing (so it still drops that row) during a rolling deploy — a lost row, never a resurrected one.
- Redis keys are namespaced per pipe so multiple pipes can share the same Redis deployment.
- A worker that shuts down drains its pending queue before its Redis tasks are cancelled and before the client is closed, so the closing worker is not the one that strands a buffered row. A flush cancelled after it has already taken a batch out of the queue puts every entry it took back before the cancellation propagates, and a row that cannot be put back is reported at CRITICAL as `ARTIFACT LOSS` rather than dropped in silence. A backlog deeper than the drain bound stays in Redis, where another worker's flusher commits it.

Failure handling (operator-relevant):
- If Redis is unavailable, the pipe degrades to direct database writes and continues serving requests.
- If a user's database reads or writes keep failing, a breaker skips that user's database work for a while and shows a warning rather than letting the failures cascade. A delete skipped by the breaker keeps both the row and its cached copy, and the next turn retries it; the delete's Redis marker is written before the breaker is consulted, so a row still waiting in the queue is dropped by the next flush rather than reaching the table.
- A single refused write, below the breaker's limit, is announced to the person on the turn it happens: the toast says some stored items for that turn could not be written to the database and will be missing from later turns. The turn itself still completes and still shows its answer, so without that notice the loss is only discoverable on a later turn, as a model that no longer has a tool result it was given.
- A flush interrupted by a cancellation - a worker shutdown, an Open WebUI hot reload, or the Redis valve being flipped - puts its uncommitted batch back on the pending queue rather than dropping it, so a peer worker or the next flush commits it. A row the writer had already acknowledged is not returned, and a row whose re-queue does not finish in the shutdown budget is named in a `CRITICAL` line with its count.

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for Redis-related valves and their defaults.

---

## Retention and cleanup

Retention has multiple layers:

### Time-based cleanup
- A periodic cleanup worker deletes persisted rows older than `ARTIFACT_CLEANUP_DAYS` (as measured from `created_at`, which is refreshed on every read, database or cache); the Redis cache entries of those rows go with them, so a purged artifact is not replayed from the cache either.
- Each sweep also deletes every row a temporary chat left behind, whatever its age.
- Cleanup cadence is controlled by `ARTIFACT_CLEANUP_INTERVAL_HOURS` (with jitter).
- A sweep reports the id range it removed (`ulid_range=<lo>..<hi>`) and takes it as a SQL `min`/`max` aggregate, so it holds a constant amount of memory rather than one id per expired row. A sweep that matched nothing reports no id range and logs no `Retention removed` line; a sweep that only reaped temporary-chat rows still logs those separately.

### Reasoning retention policy (`PERSIST_REASONING_TOKENS`)
Reasoning retention controls whether replayed reasoning artifacts are deleted after use:
- `disabled`: no reasoning is retained.
- `next_reply`: reasoning is kept only until the next assistant reply finishes, then deleted. A reply still waiting on a tool result is not finished: a turn that hands its calls back keeps the rows, and when the later request that answers those results arrives it is the one that deletes them; if it never arrives, the rows wait for the periodic cleanup. A turn that was cancelled or errored keeps them too, since it produced no reply at all.
- `conversation`: reasoning is kept for the full chat history (until time-based cleanup removes it).

`PERSIST_REASONING_TOKENS` governs what is kept and what later turns replay. A stored reasoning item is replayed on both endpoints: on `/responses` as its own top-level `input` item, and on `/chat/completions` moved onto the assistant message that carries the round as `reasoning_details` — the shape Open WebUI itself sends to a chat-completions connection.

Tool-round copies do not follow this setting: they stay for the whole conversation, until time-based cleanup
removes them. Under `next_reply` the cleanup runs at the end of a reply-producing generation only. It spares the rows
of the message that generation is still writing, so continuing an answer does not delete the reasoning of the
generation it continues, and it spares the rows of a generation that was cancelled, errored, or handed its tool calls
back, none of which finished a reply. The
sparing holds on the Redis write-behind path too: a spared row still in the pending queue is committed and re-cached
rather than dropped, because the decision is read from the queued row's own message id.

### Tool output pruning
When an earlier turn's tool result is handed to the model again, `TOOL_OUTPUT_RETENTION_TURNS` decides how much of it goes: results from the most recent turns go in full, while a long result from an older turn is cut to its first and last few hundred characters with a note of how much was removed. OpenRouter's own advisor, subagent and model-search items go back whole. The stored row is not changed.

---

## Valves controlling persistence (common)

| Valve | Default (verified) | Notes |
| --- | --- | --- |
| `ARTIFACT_ENCRYPTION_KEY` | `(empty)` | Enables artifact encryption when set. Changing it changes the table name used for artifact storage. |
| `ENCRYPT_ALL` | `True` | When encryption is enabled, controls whether all artifacts are encrypted or only reasoning. It decides what is written; a row already stored encrypted stays encrypted, in the table and in the replay cache, whatever it is set to. |
| `ENABLE_LZ4_COMPRESSION` | `True` | Compresses some payloads before encryption (when `lz4` is available and compression is beneficial). |
| `MIN_COMPRESS_BYTES` | `0` | Compression threshold; `0` always attempts compression. |
| `ENABLE_REDIS_CACHE` | `True` | Enables Redis support when Redis is available and the deployment is a candidate for it. Re-read on every request, so it can be turned off and back on again without a restart; the request path reconnects when it is turned back on. A connection attempt that fails is retried on an interval rather than on every request: 300 seconds, a fixed constant and not a valve, cleared by the valve toggle and by a changed `REDIS_URL`. |
| `REDIS_CACHE_TTL_SECONDS` | `600` | TTL for cached artifacts in Redis, read live at each write, so a change applies to entries written from that moment on. |
| `ARTIFACT_CLEANUP_DAYS` | `90` | Time-based retention window. |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` | `1.0` | Cleanup cadence. |
| `DB_BATCH_SIZE` | `10` | DB transaction batching (also used for Redis flush batching). |
| `PERSIST_REASONING_TOKENS` | `conversation` | Reasoning retention policy (system default). |
| `TOOL_OUTPUT_RETENTION_TURNS` | `10` | How many recent turns hand the model their tool results in full; older long results are shortened. |

For the complete list (including breaker and tool execution settings), see [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
