# persistence, encryption, and storage

**file:** `docs/persistence_encryption_and_storage.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:3600-7200`, marker helpers at `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:8132+`

The OpenRouter manifold persists reasoning traces, tool inputs/outputs, and other structured artifacts so that regenerations can replay prior context without bloating chat transcripts. This document explains how the storage layer works, how encryption/compression are applied, and how Redis write-behind keeps multi-worker deployments consistent.

---

## 1. artifact lifecycle

1. **Creation** -- Whenever the streaming loop finishes a reasoning chunk or a tool execution completes, `_persist_artifacts` receives a list of dicts (`{"chat_id", "message_id", "payload", ...}`). Each entry receives a ULID (`generate_item_id`) if one was not provided and is stamped with timestamps + metadata.
2. **Encryption & compression** -- `_prepare_rows_for_storage()` inspects `self._encryption_key`, `self._encrypt_all`, and `self._compression_enabled`:
   * If no `ARTIFACT_ENCRYPTION_KEY` is configured, payloads are JSON blobs stored as-is.
   * When a key is present, a Fernet cipher is constructed lazily. Each payload is serialized, optionally compressed via LZ4 (flag stored in a 1-byte header), encrypted, and base64-encoded. Encryption now happens **before** any Redis write so queues/caches only ever contain ciphertext (`{"ciphertext": "...", "enc_v": 1}`).
   * `ENCRYPT_ALL=False` limits encryption to reasoning tokens while leaving other artifacts plaintext. Set it to `True` to encrypt everything.
3. **Write path** -- `_persist_artifacts` either writes directly via `_db_persist_direct` or funnels entries through Redis (`_redis_enqueue_rows`) based on the deployment. Direct writes happen through SQLAlchemy sessions running inside a `ThreadPoolExecutor` to keep the event loop non-blocking.
4. **Marker emission** -- After rows are stored, `_serialize_marker(ulid)` strings are appended to the assistant text so `_transform_messages_to_input` can recover them on the next turn.

```text
# example ULID + marker lines appended to assistant text
[01J2VVZBDQTP1ZQJ8DSN6BV4KQ]: #
[01J2VVZBGK3RH0JP7F7C7M7N5H]: #

# corresponding artifact payload (prior to optional encryption/compression)
{
  "id": "01J2VVZBDQTP1ZQJ8DSN6BV4KQ",
  "chat_id": "chat_Ms6u8pWk",
  "message_id": "msg_9Yf31",
  "type": "function_call_output",
  "call_id": "weather_lookup#1",
  "payload": {
    "type": "function_call_output",
    "output": "{\"location\":\"Berlin\",\"conditions\":\"Rain\"}",
    "call_id": "weather_lookup#1"
  },
  "created_at": "2025-01-07T18:22:13.912083Z"
}
```

> The ULID format is Crockford base32 with 16 characters of timestamp and 4 characters of randomness. You never handcraft these IDs--`generate_item_id()` does it automatically whenever `_persist_artifacts` is called.

```python
# code excerpt: generate_item_id() + marker serialization
def generate_item_id() -> str:
    timestamp = time.time_ns() & _ULID_TIME_MASK
    time_component = _encode_crockford(timestamp, ULID_TIME_LENGTH)
    random_bits = secrets.randbits(ULID_RANDOM_LENGTH * 5)
    random_component = _encode_crockford(random_bits, ULID_RANDOM_LENGTH)
    return f"{time_component}{random_component}"

def _serialize_marker(ulid: str) -> str:
    return f"[{ulid}{_MARKER_SUFFIX}"
```

The snippet above lives at `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:8132+`. When `_persist_artifacts()` stores rows, it appends `_serialize_marker(ulid)` lines to the assistant text; `_transform_messages_to_input()` later scans for those markers, fetches the referenced payloads, and replays them as structured Responses blocks.

---

## 2. database schema & naming

* **Table name pattern** -- `_init_artifact_store()` sanitizes `self.id` via `_sanitize_table_fragment()`, hashes `(ARTIFACT_ENCRYPTION_KEY + pipe_id)` with SHA-256, and emits `response_items_{fragment}_{hash[:8]}`. Example: `response_items_openrouter_6d9a1b2c`. This allows multiple copies of the pipe (different IDs or different encryption keys) to coexist without clobbering each other, and key rotation automatically writes to a new table so old ciphertext remains unreadable by design.
* **Self-healing creation** -- If table creation fails because of orphaned `ix_*` indexes (common after manual DB restores), `_maybe_heal_index_conflict()` drops those indexes and retries. Success is logged ("Dropped orphaned index(es)...") and persistence continues without operator work.
* **Model shape** -- `_item_model` is generated dynamically with columns for `id` (ULID PK), `chat_id`, `message_id`, `model_id`, `item_type`, `payload`, `is_encrypted`, and `created_at`. JSON payloads are stored plaintext or encrypted blobs depending on valves.
* **Engine/session reuse** -- The pipe reuses Open WebUI"s engine + `SessionLocal`, so artifacts share the same DB pool/config as the rest of the server. All blocking DB calls run inside `_db_executor` (a `ThreadPoolExecutor`) so async handlers stay responsive.
* **Bootstrap signature** -- `_artifact_store_signature = (table_fragment, encryption_key)` prevents redundant reinitialization. When either component changes (e.g., new key), the SQLAlchemy model and table are recreated automatically.

---

## 3. redis write-behind cache

Redis is optional but recommended for multi-worker deployments:

1. **Eligibility check** -- `_redis_candidate` requires `UVICORN_WORKERS>1`, `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, `WEBSOCKET_REDIS_URL`, `ENABLE_REDIS_CACHE=True`, and the `redis.asyncio` package. Otherwise the pipe logs why Redis stayed disabled.
2. **Queueing** -- `_redis_enqueue_rows` pushes serialized JSON rows onto `pending` (list) and caches each payload under `artifact:{chat_id}:{row_id}` with TTL `REDIS_CACHE_TTL_SECONDS`. When encryption is enabled, these rows already contain ciphertext so Redis never stores plaintext artifacts. A pub/sub message (`db-flush`) wakes other workers so they flush immediately.
3. **Flushing** -- `_redis_periodic_flusher` acquires a short-lived lock (`flush_lock`) to avoid duplicate flushes, pops up to `DB_BATCH_SIZE` entries, rehydrates them, and calls `_db_persist_direct`. Failures requeue the raw JSON and increment `REDIS_FLUSH_FAILURE_LIMIT`. Exceeding the limit disables Redis automatically and logs a critical alert.
4. **Reads** -- `_redis_fetch_rows` looks up cached artifacts when `_transform_messages_to_input` needs them. Cache hits save DB round-trips and keep multi-worker replay deterministic.
5. **Namespacing** -- Each pipe instance sets `_redis_namespace = (self.id or \"openrouter\").lower()`, and all keys derive from it: `pending` queue, cache prefix, and flush lock. Multiple pipes can therefore share the same Redis cluster without stepping on one another.
6. **Graceful degradation** -- If Redis fails mid-run, `_redis_enqueue_rows` logs a warning and falls back to direct DB writes, then retries enabling Redis later. No operator toggle is required.

---

## 4. cleanup & retention

* **Scheduled worker** -- `_artifact_cleanup_worker` wakes every `ARTIFACT_CLEANUP_INTERVAL_HOURS` (with jitter), deletes rows older than `ARTIFACT_CLEANUP_DAYS`, and logs the number of rows removed. Runs inside the DB executor.
* **Reasoning retention** -- `PERSIST_REASONING_TOKENS` decides whether reasoning artifacts are deleted immediately, after the next assistant reply, or never (until manual cleanup).
* **Tool retention** -- `TOOL_OUTPUT_RETENTION_TURNS` limits how many turns worth of tool results are replayed in full; older outputs are pruned inline via `_prune_tool_output`.
* **Self-healing breakers** -- `_db_breakers` watch for repeated DB errors per user. After `BREAKER_MAX_FAILURES` hits inside the `BREAKER_WINDOW_SECONDS` window (defaults: 5 / 60s), persistence is skipped (a status message warns the user). Once a write succeeds, the breaker drains automatically and normal persistence resumes.

---

## 5. valves controlling persistence

| Valve | Default | Notes |
| --- | --- | --- |
| `ARTIFACT_ENCRYPTION_KEY` | `""` | Minimum 16 chars recommended. Changing it rotates to a new table namespace. Requires `WEBUI_SECRET_KEY` so `EncryptedStr` can decrypt the valve. |
| `ENCRYPT_ALL` | `False` | Encrypt every artifact, not just reasoning payloads. |
| `ENABLE_LZ4_COMPRESSION` | `True` | Requires the optional `lz4` extra. When disabled, encrypted payloads remain uncompressed JSON. |
| `MIN_COMPRESS_BYTES` | `0` | Skip compression for small payloads by raising this threshold. |
| `ENABLE_REDIS_CACHE` | `True` | Auto-disabled unless the environment indicates a multi-worker Redis setup. |
| `REDIS_CACHE_TTL_SECONDS` | `600` | TTL for cached artifacts. Keep it long enough to cover the longest expected chat session. |
| `REDIS_PENDING_WARN_THRESHOLD` | `100` | Emits warnings when the pending queue grows beyond this size. |
| `REDIS_FLUSH_FAILURE_LIMIT` | `5` | Disables Redis after N consecutive flush failures. |
| `ARTIFACT_CLEANUP_DAYS` | `90` | How long artifacts remain in the DB before the cleanup job deletes them. |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` | `1.0` | Cadence for the cleanup worker. |
| `DB_BATCH_SIZE` | `10` | Number of rows written per DB transaction when draining Redis. |

---

## 6. failure modes & mitigations

| Scenario | Behavior | Operator action |
| --- | --- | --- |
| DB unreachable | `_db_persist_direct` raises → DB breaker trips → persistence disabled temporarily with user-visible warning. | Fix DB connectivity, then the breaker self-heals after one valve-defined window of successful operations. |
| Redis offline mid-run | `_redis_enqueue_rows` logs a warning and falls back to direct DB writes. A later success automatically re-enables the cache. | Inspect Redis logs; no manual recovery needed. |
| Encryption misconfigured | If `ARTIFACT_ENCRYPTION_KEY` is set but `WEBUI_SECRET_KEY` is missing, the pipe warns and treats the encrypted valve as plaintext. Artifacts remain unencrypted until the env var is provided. | Set `WEBUI_SECRET_KEY`, restart the pipe, and consider rotating the artifact key since earlier rows were stored plaintext. |
| Key rotation | Changing `ARTIFACT_ENCRYPTION_KEY` creates a new table. Existing chats cannot read the old ciphertext (intentional defense-in-depth). | Communicate rotations ahead of time; export/import artifacts if long-term retention is required. |

Persistence is the backbone of the manifold. If you touch any of the helpers mentioned here, update this doc and the production readiness audit so deployers understand the new guarantees.
