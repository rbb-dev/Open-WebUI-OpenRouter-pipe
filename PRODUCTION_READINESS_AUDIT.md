# Production Readiness Audit – OpenRouter Responses Pipe

**Last reviewed:** 2025‑01‑31  
**Auditor:** Codex (via GPT‑5)  
**Scope:** End‑to‑end readiness of the OpenRouter Responses manifold with emphasis on multimodal intake, persistence, and concurrency controls. References: `openrouter_responses_pipe/openrouter_responses_pipe.py`, Open WebUI sources at `C:\Work\Dev\open-webui`, and OpenRouter docs at `C:\Work\Dev\openrouter_docs\manual`.

---

## 1. Secrets & Key Material

| Secret | Location | Purpose / Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` (or valves `API_KEY`) | `.env` / OWUI UI | Required for every OpenRouter call. Stored as `EncryptedStr` in valves so it can live in config without plain‑text exposure. |
| `WEBUI_SECRET_KEY` | OWUI env (**required**) | Powers `EncryptedStr` runtime encryption/decryption. Without it, encrypted valve values (API keys, etc.) fall back to plain text and artifact encryption cannot function. |
| `ARTIFACT_ENCRYPTION_KEY` (≥16 chars) | Valve or env | Derives per‑pipe Fernet key. Changing it rotates the storage namespace and intentionally makes prior artifacts unreadable (defense‑in‑depth). |

Operational guidance:
- Manage keys via OWUI’s secured settings or deployment‑specific secret stores.
- Document rotation runbooks (rotate `ARTIFACT_ENCRYPTION_KEY` when retiring a cluster; rotate `OPENROUTER_API_KEY` on provider compromise).

---

## 2. Persistence, Encryption, and Redis

### Encrypted Artifacts
- `ARTIFACT_ENCRYPTION_KEY` seeds a Fernet key; `ENCRYPT_ALL` decides whether only reasoning tokens or every artifact is encrypted. **Important:** Fernet setup also requires `WEBUI_SECRET_KEY` so `EncryptedStr` can derive the runtime secret; if `WEBUI_SECRET_KEY` is missing, the pipe logs a warning, treats encrypted valves as plain text, and artifact encryption silently downgrades to **unencrypted JSON** (defense‑in‑depth is lost).
- Storage tables embed the key hash in their name: rotating the key creates a brand‑new table and leaves previous data inaccessible (expected behaviour).

### Compression
- Optional LZ4 compression (gated by `ENABLE_LZ4_COMPRESSION`) stores a one-byte header flag (`plain` vs `lz4`). By default the pipe always attempts compression and keeps it only when the LZ4 output is smaller, though operators can raise `MIN_COMPRESS_BYTES` to skip tiny payloads.

### Redis Write‑Behind
- When Open WebUI is configured for multi-worker Redis (`UVICORN_WORKERS>1`, `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, `WEBSOCKET_REDIS_URL`) and `ENABLE_REDIS_CACHE` is true, artifacts flow into a Redis pending list.
- A background worker flushes JSON blobs to the DB in batches of `DB_BATCH_SIZE`. Each entry is cached with TTL `REDIS_CACHE_TTL_SECONDS` to serve replays from memory.
- Flush failures now re‑queue the raw JSON entries and extend the queue TTL, so artifacts aren’t dropped. Repeated failures trip the `REDIS_FLUSH_FAILURE_LIMIT` breaker and force a fallback to direct DB writes.

### DB Breakers & Error Surfacing
- Per‑user DB breaker avoids hammering a failing database; when tripped, the pipe emits a warning status (“DB ops skipped due to repeated errors”) so users know persistence is temporarily disabled.

---

## 3. Multimodal Intake & Guardrails

### Remote Downloads
- `_download_remote_url` streams chunk‑by‑chunk via `httpx.AsyncClient.stream`, aborting as soon as either `Content-Length` or the running byte counter exceeds `REMOTE_FILE_MAX_SIZE_MB` (auto‑clamped to OWUI’s `FILE_MAX_SIZE` when RAG storage is enabled).
- SSRF protection resolves every IPv4+IPv6 address for a host, blocking private, loopback, link‑local, multicast, reserved, or unspecified ranges. DNS failures default to **unsafe**.

### Base64 Validation
- `_validate_base64_size` estimates decoded size (`len * 3 / 4`) and enforces `BASE64_MAX_SIZE_MB` before decoding. Violations raise warnings and surface user‑friendly status messages.

### Storage Ownership
- When the chat user is absent (API automations, system calls), uploads use a dedicated fallback identity derived from `FALLBACK_STORAGE_*` valves. Default role is low‑privilege `pending`, and the code warns if a privileged role is configured. A random `oauth_sub` is attached to the auto‑created user so it cannot log in interactively.

---

## 4. Concurrency & Tooling Pipeline

### Request Admission Control
- `MAX_CONCURRENT_REQUESTS` (default 200) is enforced via a global semaphore to cap live OpenRouter calls.
- `_QUEUE_MAXSIZE = 500` bounds the work queue; when full, new requests immediately return 503 instead of overwhelming worker tasks.

### Tool Execution
- Per‑request tool queues enforce:
  - Global (`MAX_PARALLEL_TOOLS_GLOBAL`) and per‑request (`MAX_PARALLEL_TOOLS_PER_REQUEST`) semaphores.
  - Timeouts (`TOOL_TIMEOUT_SECONDS`, `TOOL_BATCH_TIMEOUT_SECONDS`, optional idle timeout).
  - Batch execution when tool calls share the same function and have no argument dependencies.
  - Per‑user tool breakers that temporarily disable tool types after consecutive failures and emit UX status updates.

---

## 5. Streaming Pipeline

- SSE producer pumps raw chunks into a bounded queue; multiple worker tasks parse them and enforce in‑order delivery via sequence IDs.
- Text deltas are batched up to `STREAMING_UPDATE_CHAR_LIMIT` characters (or until `STREAMING_IDLE_FLUSH_MS` elapses) to balance latency vs. event spam.
- A surrogate‑pair normalizer ensures UTF‑16 pairs aren’t split across updates, preventing Unicode corruption during streaming.

---

## 6. Outstanding Items / Watchlist

| Area | Status | Notes |
| --- | --- | --- |
| Tests for streaming download path | Missing | Would require mocking `httpx.AsyncClient.stream` and the SSRF guard; currently manual testing only. |
| Integration tests for Redis requeue | Missing | Requeue logic recently added; consider adding a fake Redis or contract test. |
| Documentation | ✅ | `VALVES_REFERENCE.md`, `MULTIMODAL_IMPLEMENTATION.md`, and this audit now describe current behaviour. |

---

## 7. Startup & Catalog Resilience

- `_maybe_start_startup_checks()` performs an asynchronous warmup: it waits for an API key, pings OpenRouter with retries (`_ping_openrouter`), and caches the model catalog up front so the first user request doesn’t pay the entire cold-start penalty. Failures set `_warmup_failed`, which short‑circuits future requests with a clear status message until the operator fixes the configuration.
- `OpenRouterModelRegistry.ensure_loaded()` guards every request; it pulls `/models`, caches it for `MODEL_CATALOG_REFRESH_SECONDS`, and tolerates transient failures by serving stale data. Consecutive errors trigger exponential backoff (`_record_refresh_failure`) so we don’t hammer the API when upstream is down.
- Model selectors sanitise IDs, strip pipe prefixes, and respect OpenRouter capability metadata so features (reasoning, tool support, etc.) are keyed off actual provider declarations rather than user input.

---

## 8. Logging & Observability

- `SessionLogger` attaches contextvars to every log record (session ID, user ID, per-request log level) and streams through an async queue so slow sinks can’t stall request handlers. Each request accumulates a rotating in-memory buffer (2k entries) that can be cited back to the user when errors occur (`show_error_log_citation=True` in `_emit_error`).
- `LOG_LEVEL` valves can be overridden globally or per user, making it straightforward to capture DEBUG logs for a single tenant without flooding production logs.
- Status updates (`_emit_status`, `_emit_notification`, `_emit_completion`) ensure UI feedback is continuous: long tool runs and reasoning phases emit progress, and final status strings summarize elapsed time, cost, token usage, and tokens/sec when available.

---

## 9. Failure Surfacing & UX Backpressure

- Multiple breakers exist: user-level (`_breaker_allows`), tool-type (`_tool_type_allows`), and DB-level (`_db_breaker_allows`). When they trip, the pipe emits status messages (“Skipping … due to repeated failures”, “DB ops skipped due to repeated errors”) so operators and users know why a request degraded.
- Request admission control is explicit: `_QUEUE_MAXSIZE` is capped at 500, and a global semaphore enforces `MAX_CONCURRENT_REQUESTS`. If the queue fills the user receives a prompt 503 with a visible status rather than a timeout.
- Streaming SSE worker pool reorders events by sequence and batches text updates, but still guarantees delivery of reasoning deltas, citations, and final usage stats—even when the downstream client disconnects mid-stream we still flush pending artifacts (`_flush_pending`) before tearing down the job.

## 10. Testing Notes

- The pytest plugin `openrouter_responses_pipe.pytest_bootstrap` lives at the repo root, so developers must set `PYTHONPATH=.` (or install the repo with `pip install -e .`) before running `pytest`. Without that, Python cannot import the plugin and collection fails with `ModuleNotFoundError: openrouter_responses_pipe`.
- `tests/conftest.py` provides lightweight stubs for Open WebUI, FastAPI, SQLAlchemy, pydantic-core, and tenacity. No external services are required; just activate the `.venv` (Python 3.12) and run the suite.

---

**Conclusion:** With size limits, SSRF defenses, encryption, and the reworked Redis durability, the manifold meets the production requirements outlined above. Remaining gaps are in automated coverage rather than runtime safeguards. Add the noted tests when bandwidth allows, but the operational story is now consistent and documented.
