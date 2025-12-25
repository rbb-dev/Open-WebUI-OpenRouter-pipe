# Production readiness report (OpenRouter Responses Pipe)

**Last reviewed:** 2025-12-25  
**Scope:** Unified production readiness guidance for deploying this pipe in Open WebUI (single-worker and multi-worker), covering security, operations, and failure modes. This document is guidance only and does not guarantee compliance outcomes.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Security](security_and_encryption.md) · [Errors](error_handling_and_user_experience.md) · [Telemetry](openrouter_integrations_and_telemetry.md)

---

## 1. Version and source of truth

This report is written against:

- `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py` (pipe manifest header: `required_open_webui_version: 0.6.28`, manifest `version: 1.0.13`)
- `tests/` (behavioral checks for key safety and integration paths)

When there is any conflict between documentation and runtime behavior, the code and tests are the source of truth.

---

## 2. Executive summary (what to decide before production)

Before enabling this pipe for end users, operators should decide and document:

1. **Network egress posture** (especially remote downloads; SSRF risk).
2. **Persistence posture** (what is stored, retention windows, and who can access it).
3. **Key management posture** (`OPENROUTER_API_KEY`, `WEBUI_SECRET_KEY`, `ARTIFACT_ENCRYPTION_KEY`, session log zip password).
4. **Concurrency posture** (request admission limits, tool concurrency, streaming queue sizing).
5. **Telemetry posture** (whether to export cost snapshots to Redis; whether to send request identifiers).
6. **Incident response posture** (templates, session logs, and attribution identifiers).

All of these are controlled via valves and environment configuration; see [Valves & Configuration Atlas](valves_and_configuration_atlas.md).

---

## 3. Deployment modes (single-worker vs multi-worker)

### Single-worker (no Redis)

This is the simplest production mode:

- Requests are executed per-process with bounded admission control and semaphores.
- Persistence (if enabled) writes directly to the database.
- Redis features are not used.

### Multi-worker (Redis-enabled candidate)

The pipe only uses Redis when the runtime environment indicates a multi-worker Open WebUI deployment and Redis prerequisites are met:

- `UVICORN_WORKERS > 1`
- `REDIS_URL` is set
- `WEBSOCKET_MANAGER=redis` and `WEBSOCKET_REDIS_URL` are set
- `ENABLE_REDIS_CACHE=true`
- Redis client dependency is available at runtime

If any prerequisite is missing, the pipe logs warnings and runs without Redis.

Related docs: [Persistence, Encryption & Storage](persistence_encryption_and_storage.md).

---

## 4. Security posture

### 4.1 Secrets and key material

| Secret | Purpose | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` (or valve `API_KEY`) | Provider authentication | Required for provider requests and catalog refresh. |
| `WEBUI_SECRET_KEY` | Protects encrypted valve values (`EncryptedStr`) | Recommended in production if storing secrets in valves. |
| `ARTIFACT_ENCRYPTION_KEY` | Enables artifact encryption at rest (when set) | Changing this changes the artifact table namespace (hash-suffixed table name). |
| `SESSION_LOG_ZIP_PASSWORD` | Encrypts session log zip archives (when enabled) | Required to write session log archives; treat as a high-value secret. |

Operational notes (code-aligned):

- `WEBUI_SECRET_KEY` protects *stored secret valve values* (encryption/decryption via `EncryptedStr`).
- If `WEBUI_SECRET_KEY` is missing/mismatched relative to how values were stored, those encrypted values may not decrypt to the intended plaintext at runtime, causing authentication failures or unexpected table namespaces/passwords.

Related docs: [Security & Encryption](security_and_encryption.md).

### 4.2 Remote downloads and SSRF

Remote downloads are security-sensitive. The pipe’s download subsystem:

- permits `http://` and `https://` URLs (it is not HTTPS-only),
- supports SSRF protection via `ENABLE_SSRF_PROTECTION` (default `True`),
- enforces size limits and retry/time budgets for downloads.

Operator guidance:

- Keep SSRF protection enabled.
- Enforce egress restrictions in your network (proxy allowlists, firewall rules).
- If you require HTTPS-only remote retrieval, enforce it via egress policy (do not rely on the pipe alone).

Related docs: [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md), [Security & Encryption](security_and_encryption.md).

### 4.3 Identifiers and privacy

The pipe can emit Open WebUI identifiers to OpenRouter using valve-gated controls:

- top-level `user` and `session_id`
- `metadata.user_id/session_id/chat_id/message_id`

Operator guidance:

- Treat these as pseudonymous identifiers (they can be used to correlate activity).
- Enable only what you need for operations/abuse attribution.

Related docs: [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).

---

## 5. Persistence and retention posture

### 5.1 Artifact persistence and encryption

The pipe can persist response artifacts (reasoning and tool outputs) to a per-pipe SQL table.

Key properties (code-aligned):

- The table name includes a sanitized pipe id fragment and a short hash of `(ARTIFACT_ENCRYPTION_KEY + pipe_identifier)`.
- Artifact encryption is enabled when `ARTIFACT_ENCRYPTION_KEY` is set; `ENCRYPT_ALL` controls whether all artifacts or reasoning-only artifacts are encrypted.
- Optional LZ4 compression is applied before encryption when enabled and beneficial.

Related docs: [Persistence, Encryption & Storage](persistence_encryption_and_storage.md), [History Reconstruction & Context Replay](history_reconstruction_and_context.md).

### 5.2 Cleanup

The pipe includes:

- a periodic artifact cleanup worker (time-based retention via `ARTIFACT_CLEANUP_DAYS` and interval via `ARTIFACT_CLEANUP_INTERVAL_HOURS`, with jitter),
- replay pruning rules (for example, tool output retention window in turns).

Operator guidance:

- Set retention windows deliberately (especially in multi-tenant deployments).
- Validate that your database backups and access controls match your retention and privacy posture.

---

## 6. Concurrency and resilience posture

### 6.1 Request admission controls

The pipe uses:

- a bounded request queue (`_QUEUE_MAXSIZE = 500`), and
- a per-process concurrency semaphore controlled by `MAX_CONCURRENT_REQUESTS` (default `200`).

Notes:

- Increasing limits can take effect at runtime; decreasing some global limits may require a restart to fully take effect (the pipe logs when a restart is required).

Related docs: [Concurrency Controls & Resilience](concurrency_controls_and_resilience.md).

### 6.2 Tool execution controls

Tools are executed between Responses calls (after a run completes and `function_call` items are returned):

- per-request tool queue is bounded (maxsize `50`),
- per-request concurrency is limited by `MAX_PARALLEL_TOOLS_PER_REQUEST` (default `5`),
- global tool concurrency is limited by `MAX_PARALLEL_TOOLS_GLOBAL` (default `200`),
- per-call timeout by `TOOL_TIMEOUT_SECONDS`, batch timeout by `TOOL_BATCH_TIMEOUT_SECONDS`, optional idle timeout by `TOOL_IDLE_TIMEOUT_SECONDS`,
- limited retries for tool calls are applied.

Related docs: [Tools, plugins, and integrations](tooling_and_integrations.md).

### 6.3 Breakers and degradation

The pipe maintains breaker windows for repeated failures:

- per-user request breaker (temporary rejection with a user-facing message),
- per-user/per-tool-type breaker (skips tools after repeated failures),
- per-user DB breaker (temporarily suppresses persistence work after repeated DB failures).

Operator guidance:

- Treat breaker trips as a symptom to investigate, not as a “normal steady state”.
- Ensure your error templates and logs make it easy for users/operators to understand degraded behavior.

Related docs: [Error Handling & User Experience](error_handling_and_user_experience.md), [Concurrency Controls & Resilience](concurrency_controls_and_resilience.md).

---

## 7. Streaming posture

Provider streaming uses an SSE ingestion pipeline with worker fan-out and ordered emission to Open WebUI. Queue sizing valves exist for streaming chunks and parsed events.

Operator guidance:

- Default streaming queue sizes are unbounded; bounding them too aggressively can cause stalls under tool-heavy load.
- If you change streaming queue valves, test under realistic concurrency before rollout.

Related docs: [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md).

---

## 8. Telemetry posture (optional)

Optional operator-visible and downstream telemetry includes:

- user-visible final usage banners (valve-gated),
- optional Redis export of cost snapshots (valve-gated; see `COSTS_REDIS_DUMP`),
- request identifier emission for attribution (valve-gated; see identifier valves).

Related docs: [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md), [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).

---

## 9. Session logs (optional incident response feature)

When enabled (`SESSION_LOG_STORE_ENABLED=true`) and configured (zip password set and `pyzipper` available), the pipe can write encrypted per-request log archives to disk via a bounded writer queue.

Operator guidance:

- Treat session log archives as sensitive; restrict filesystem access and define retention policies.
- Verify that zip passwords are managed and rotated intentionally.

Related docs: [Encrypted session log storage (optional)](session_log_storage.md).

---

## 10. Pre-deploy checklist (unified)

Minimum functional:

- Open WebUI meets `required_open_webui_version` (0.6.28+).
- `OPENROUTER_API_KEY` (or valve `API_KEY`) is configured.
- `BASE_URL` is correct for your environment (direct OpenRouter or gateway/proxy).

Recommended security:

- `WEBUI_SECRET_KEY` is configured (so secret valves can be stored encrypted and decrypted reliably).
- SSRF protection is enabled and network egress policy is in place.
- Persistence posture is decided (encryption key set or intentionally left empty; retention windows set).

Multi-worker readiness (only if applicable):

- Redis prerequisites are met and validated if you expect Redis caching/write-behind to activate.

---

## 11. Post-deploy validation (unified)

Run a minimal validation in a staging environment:

1. **Catalog load**: models appear in Open WebUI; if not, validate connectivity/credentials and check backend logs for catalog refresh errors.
2. **Basic chat**: normal completion succeeds with streaming.
3. **Tool call**: enable a simple tool and confirm `function_call` → tool execution → continued completion works.
4. **Multimodal**: upload an image/file and confirm it is accepted and forwarded on a capable model.
5. **Persistence/replay** (if enabled): two-turn flow verifies that prior artifacts can be replayed.
6. **Error templates**: simulate invalid API key / rate limit / provider error and confirm the UI renders actionable templates.
7. **Session logs** (if enabled): confirm encrypted zip archives are written only for requests that have all required IDs and a configured password.

Related docs: [Testing, bootstrap, and operational playbook](testing_bootstrap_and_operations.md).

---

## 12. Compliance guidance (non-guarantee)

This project can help implement technical controls (encryption, retention, logging, SSRF mitigation), but it does not guarantee GDPR/HIPAA/SOC2/PCI compliance by itself. Compliance is an end-to-end system property (data flows, access control, retention, incident response, audit readiness) and must be assessed in your environment.
