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
  encrypted at rest only when `ARTIFACT_ENCRYPTION_KEY` is set and `ENCRYPT_ALL` is on.

The pipe never stores a copy of a picture or file a person attaches, in any chat.

A temporary chat keeps none of its content in the pipe's storage. Open WebUI keeps it only in the browser, and the pipe stores none of it (database, disk or file storage): no tool round, no reasoning and no session log, and no picture bytes in the worker's memory - a picture a temporary chat reuses is fetched again on each request that needs it rather than cached in the process. When usage collection (`PIPE_DASHBOARD_USAGE_COLLECT`) or cost snapshots (`COSTS_REDIS_DUMP`) are switched on (both off by default), a temporary chat's records keep every figure but name no chat: its usage row leaves the chat and session ids empty, and its cost snapshot carries no chat or message id. Saved and channel chats keep those ids. The same boundary holds on the wire: a temporary chat's chat, session and message ids are never placed in the request `metadata` sent to OpenRouter, whatever the identifier valves are set to (see [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md)). While an admin has the dashboard open on a deployment that uses Redis, a temporary chat's live row passes through Redis (rewritten every 2 seconds, each entry expiring after 10) under an anonymous stand-in keyed by `WEBUI_SECRET_KEY`, never under its chat id, its socket id or the key sent to OpenRouter, and with no id at all when that secret is unset. Three things are kept the way Open WebUI keeps them:

- Pictures and videos a model generates are saved to Open WebUI's file storage so the chat can show them, as Open WebUI does for pictures its own image generation makes.
- Pictures and audio an MCP tool returns are saved as files by Open WebUI's own tool handling, as it does in any chat.
- A streamed reply that may hand a call back -- in Open-WebUI mode, or in Pipeline mode for a tool the pipe cannot run -- holds its rounds and thinking in memory for that reply only, dropped when the pipe answers its last call back, or after 15 minutes unused. A reply that cannot hand one back holds nothing and writes nothing. The rounds and thinking of the reply being written are held in the pipe's memory, never on disk or in the database, until that reply ends, so those calls hand the model the same turn a saved chat's would. A reply left idle for 15 minutes is dropped. Once the held rounds pass 64 MiB in one Open WebUI worker, the longest-idle replies are dropped first, and for a single reply larger than that, the pipe drops everything held for it so far. For a dropped reply, the model continues without the thinking and rounds held for it.

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

On subsequent turns, the pipe scans prior assistant messages for marker lines, fetches the referenced artifacts (from Redis cache if available, otherwise the database), and replays them into the next request in a structured form. Markers are emitted for chat turns only: a call that carries no `chat_id` adds none to its response, so a program calling the API sees exactly the bytes the model produced.

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

### Stored columns (high level)
Persisted rows include:
- `id` (the ULID)
- Open WebUI identifiers such as `chat_id` and `message_id`
- `item_type`
- `payload` (plaintext JSON, or an encrypted wrapper when encryption is enabled)
- `created_at` (UTC timestamp used for retention; refreshed on DB reads)

---

## Encryption and compression behavior

Artifact encryption is controlled by these system valves:

- `ARTIFACT_ENCRYPTION_KEY` (enables encryption when non-empty)
- `ENCRYPT_ALL` (default `True`; controls “encrypt everything” vs “encrypt reasoning only”)
- `ENABLE_LZ4_COMPRESSION` and `MIN_COMPRESS_BYTES` (optional compression for persisted payloads)

### When encryption is active
- If `ARTIFACT_ENCRYPTION_KEY` is set (non-empty), the pipe encrypts payloads before persistence.
- When encrypted, the stored `payload` becomes a wrapper containing ciphertext (plus a version field).
- If compression is enabled and effective, the pipe compresses the JSON payload before encryption and stores a small header indicating whether the stored bytes were compressed.

### When encryption is not active
- If `ARTIFACT_ENCRYPTION_KEY` is empty/unset, payloads are stored as normal JSON objects and `is_encrypted=False`.

**Operator note:** If you store secret valve values using Open WebUI encryption (`EncryptedStr`), configure `WEBUI_SECRET_KEY` (see [Security & Encryption](security_and_encryption.md)). If `WEBUI_SECRET_KEY` is changed later, encrypted valve values may decrypt differently, which effectively behaves like a key rotation for artifact storage.

---

## Redis cache and write-behind (multi-worker deployments)

The pipe has an optional Redis-backed cache/write-behind path intended for multi-worker deployments.

### When Redis is used

`ENABLE_REDIS_CACHE` enables Redis support, but Redis is only used when the runtime environment indicates a multi-worker Open WebUI deployment and Redis tooling is available. The switch and the cache lifetime are re-read on every operation, so turning the switch off takes effect without a restart; turning it on again needs the worker to have connected at start-up, as it would after a restart. In particular, the pipe requires:

- `UVICORN_WORKERS > 1` (multi-worker mode), and
- `REDIS_URL` is set, and
- `WEBSOCKET_MANAGER=redis` and `WEBSOCKET_REDIS_URL` are set (Open WebUI websocket/Redis configuration), and
- the Python Redis client dependency is available at runtime.

If these prerequisites are not met, the pipe runs without Redis and persists artifacts directly to the database (when persistence is enabled).

The master switch is read at the moment a Redis gate is reached, not frozen at start-up, so a change saved in the admin UI takes effect on the next request that reaches one without a restart. A change to `False` stops new connections immediately; a client that is already running keeps serving its existing connections and keeps writing through them, until restart or `close()`.

High-level behavior:
- When Redis caching is enabled and available, the pipe can enqueue persisted rows into Redis and flush them to the database asynchronously.
- When Redis is enabled, the pipe can also cache persisted artifacts for faster replay reads.
- When Redis write-behind is active, a row deleted while it is still queued is removed from the table and its cache entry, so it cannot be replayed from either. The marker carrying that decision lives in Redis for its own fixed lifetime of one hour, not for `REDIS_CACHE_TTL_SECONDS`: a row still queued when its marker expires is written to the table and the delete is lost, which is what a queue backlogged past that window means. The `{ns}:deleted:{row}` marker value is the `message_id` a cleanup spared, or the sentinel `"1"` when it spared none; the key layout and the TTL are unchanged. A worker from before this change reads `"1"` the same way, but reads a spared `message_id` as truthy and will **drop that row** during a rolling deploy — a lost row, never a resurrected one. The no-keep path is byte-identical and does not diverge.
- Redis keys are namespaced per pipe so multiple pipes can share the same Redis deployment.

Failure handling (operator-relevant):
- If Redis is unavailable, the pipe degrades to direct database writes and continues serving requests.
- If a user's database reads or writes keep failing, a breaker skips that user's database work for a while and shows a warning rather than letting the failures cascade. A delete skipped by the breaker keeps both the row and its cached copy, and the next turn retries it; the delete's Redis marker is written before the breaker is consulted, so a row still waiting in the queue is dropped by the next flush rather than reaching the table.

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for Redis-related valves and their defaults.

---

## Retention and cleanup

Retention has multiple layers:

### Time-based cleanup
- A periodic cleanup worker deletes persisted rows older than `ARTIFACT_CLEANUP_DAYS` (as measured from `created_at`, which is refreshed on DB reads).
- Each sweep also deletes every row a temporary chat left behind, whatever its age.
- Cleanup cadence is controlled by `ARTIFACT_CLEANUP_INTERVAL_HOURS` (with jitter).

### Reasoning retention policy (`PERSIST_REASONING_TOKENS`)
Reasoning retention controls whether replayed reasoning artifacts are deleted after use:
- `disabled`: no reasoning is retained.
- `next_reply`: reasoning is kept only until the next assistant reply finishes, then deleted.
- `conversation`: reasoning is kept for the full chat history (until time-based cleanup removes it).

`PERSIST_REASONING_TOKENS` governs what is kept and what later turns replay. A stored reasoning item is replayed on both endpoints: on `/responses` as its own top-level `input` item, and on `/chat/completions` moved onto the assistant message that carries the round as `reasoning_details` — the shape Open WebUI itself sends to a chat-completions connection.

Tool-round copies do not follow this setting: they stay for the whole conversation, until time-based cleanup
removes them. Under `next_reply` the cleanup that runs at the end of a request spares the rows of the message that
request is still writing, so continuing an answer does not delete the reasoning of the generation it continues. The
sparing holds on the Redis write-behind path too: a spared row still in the pending queue is committed and re-cached
rather than dropped, because the decision is read from the queued row's own message id.

### Tool output pruning
When an earlier turn's tool result is handed to the model again, `TOOL_OUTPUT_RETENTION_TURNS` decides how much of it goes: results from the most recent turns go in full, while a long result from an older turn is cut to its first and last few hundred characters with a note of how much was removed. OpenRouter's own advisor, subagent and model-search items go back whole. The stored row is not changed.

---

## Valves controlling persistence (common)

| Valve | Default (verified) | Notes |
| --- | --- | --- |
| `ARTIFACT_ENCRYPTION_KEY` | `(empty)` | Enables artifact encryption when set. Changing it changes the table name used for artifact storage. |
| `ENCRYPT_ALL` | `True` | When encryption is enabled, controls whether all artifacts are encrypted or only reasoning. |
| `ENABLE_LZ4_COMPRESSION` | `True` | Compresses some payloads before encryption (when `lz4` is available and compression is beneficial). |
| `MIN_COMPRESS_BYTES` | `0` | Compression threshold; `0` always attempts compression. |
| `ENABLE_REDIS_CACHE` | `True` | Enables Redis support when Redis is available and the deployment is a candidate for it. |
| `REDIS_CACHE_TTL_SECONDS` | `600` | TTL for cached artifacts in Redis, read live at each write, so a change applies to entries written from that moment on. |
| `ARTIFACT_CLEANUP_DAYS` | `90` | Time-based retention window. |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` | `1.0` | Cleanup cadence. |
| `DB_BATCH_SIZE` | `10` | DB transaction batching (also used for Redis flush batching). |
| `PERSIST_REASONING_TOKENS` | `conversation` | Reasoning retention policy (system default). |
| `TOOL_OUTPUT_RETENTION_TURNS` | `10` | How many recent turns hand the model their tool results in full; older long results are shortened. |

For the complete list (including breaker and tool execution settings), see [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
