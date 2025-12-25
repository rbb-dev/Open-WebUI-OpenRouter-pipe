# Persistence, Encryption & Storage

This document explains how the pipe persists response artifacts (reasoning payloads and tool outputs), how artifact replay works across turns, and which valves control encryption, compression, Redis caching, and retention.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Security](security_and_encryption.md) · [History/Replay](history_reconstruction_and_context.md) · [Concurrency](concurrency_controls_and_resilience.md)

---

## What gets persisted (high level)

During a chat, the pipe can persist structured “artifacts” so future turns can replay prior context without embedding large JSON blobs directly into the visible chat transcript.

Persisted artifacts include (at least):
- Reasoning items (when enabled and retention permits).
- Tool execution artifacts (calls/outputs), subject to replay filtering rules and retention pruning.

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

On subsequent turns, the pipe scans prior assistant messages for marker lines, fetches the referenced artifacts (from Redis cache if available, otherwise the database), and replays them into the next request in a structured form.

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
- Changing `ARTIFACT_ENCRYPTION_KEY` causes the pipe to write to a different table name.
  - Existing artifacts are not “deleted” automatically, but they will not be read by the pipe unless you restore the prior key (and therefore the prior table name).

### Stored columns (high level)
Persisted rows include:
- `id` (the ULID)
- Open WebUI identifiers such as `chat_id` and `message_id`
- `item_type`
- `payload` (plaintext JSON, or an encrypted wrapper when encryption is enabled)
- `created_at`

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

`ENABLE_REDIS_CACHE` enables Redis support, but Redis is only used when the runtime environment indicates a multi-worker Open WebUI deployment and Redis tooling is available. In particular, the pipe requires:

- `UVICORN_WORKERS > 1` (multi-worker mode), and
- `REDIS_URL` is set, and
- `WEBSOCKET_MANAGER=redis` and `WEBSOCKET_REDIS_URL` are set (Open WebUI websocket/Redis configuration), and
- the Python Redis client dependency is available at runtime.

If these prerequisites are not met, the pipe runs without Redis and persists artifacts directly to the database (when persistence is enabled).

High-level behavior:
- When Redis caching is enabled and available, the pipe can enqueue persisted rows into Redis and flush them to the database asynchronously.
- When Redis is enabled, the pipe can also cache persisted artifacts for faster replay reads.
- Redis keys are namespaced per pipe so multiple pipes can share the same Redis deployment.

Failure handling (operator-relevant):
- If Redis is unavailable, the pipe degrades to direct database writes and continues serving requests.
- If database writes repeatedly fail for a user/session, a breaker can temporarily disable persistence and emit warnings rather than causing cascading failures.

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for Redis-related valves and their defaults.

---

## Retention and cleanup

Retention has multiple layers:

### Time-based cleanup
- A periodic cleanup worker deletes persisted rows older than `ARTIFACT_CLEANUP_DAYS`.
- Cleanup cadence is controlled by `ARTIFACT_CLEANUP_INTERVAL_HOURS` (with jitter).

### Reasoning retention policy (`PERSIST_REASONING_TOKENS`)
Reasoning retention controls whether replayed reasoning artifacts are deleted after use:
- `disabled`: no reasoning is retained.
- `next_reply`: reasoning is kept only until the next assistant reply finishes, then deleted.
- `conversation`: reasoning is kept for the full chat history (until time-based cleanup removes it).

### Tool output pruning
Tool artifacts are also subject to replay pruning based on `TOOL_OUTPUT_RETENTION_TURNS` (how far back tool results remain eligible for replay).

---

## Valves controlling persistence (common)

| Valve | Default (verified) | Notes |
| --- | --- | --- |
| `ARTIFACT_ENCRYPTION_KEY` | `(empty)` | Enables artifact encryption when set. Changing it changes the table name used for artifact storage. |
| `ENCRYPT_ALL` | `True` | When encryption is enabled, controls whether all artifacts are encrypted or only reasoning. |
| `ENABLE_LZ4_COMPRESSION` | `True` | Compresses some payloads before encryption (when `lz4` is available and compression is beneficial). |
| `MIN_COMPRESS_BYTES` | `0` | Compression threshold; `0` always attempts compression. |
| `ENABLE_REDIS_CACHE` | `True` | Enables Redis support when Redis is available and the deployment is a candidate for it. |
| `REDIS_CACHE_TTL_SECONDS` | `600` | TTL for cached artifacts in Redis. |
| `ARTIFACT_CLEANUP_DAYS` | `90` | Time-based retention window. |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` | `1.0` | Cleanup cadence. |
| `DB_BATCH_SIZE` | `10` | DB transaction batching (also used for Redis flush batching). |
| `PERSIST_REASONING_TOKENS` | `conversation` | Reasoning retention policy (system default). |
| `TOOL_OUTPUT_RETENTION_TURNS` | `10` | Limits how many turns back tool outputs remain eligible for replay. |

For the complete list (including breaker and tool execution settings), see [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
