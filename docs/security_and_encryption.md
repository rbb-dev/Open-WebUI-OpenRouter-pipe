# Security & Encryption

This document provides security guidance for production deployments of the OpenRouter Responses pipe, covering secrets handling, artifact encryption at rest, SSRF protections for remote downloads, and operational controls for multi-user environments.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Persistence](persistence_encryption_and_storage.md) · [Multimodal](multimodal_ingestion_pipeline.md) · [Identifiers](request_identifiers_and_abuse_attribution.md)

---

## Overview (what the pipe does for security)

Security-relevant mechanisms implemented by the pipe include:

1. **Secret valve handling** via `EncryptedStr` (optionally encrypts secret valve values at rest when `WEBUI_SECRET_KEY` is set).
2. **Artifact encryption at rest** for persisted response artifacts when `ARTIFACT_ENCRYPTION_KEY` is configured.
3. **SSRF protection** for remote URL downloads when `ENABLE_SSRF_PROTECTION=True`.
4. **Multi-user isolation primitives** (request-scoped `contextvars`, per-pipe database tables and Redis namespaces).
5. **Size/time guardrails** for remote downloads and base64 payloads to limit resource exhaustion.
6. **Optional encrypted session log archives** (zip encryption) when session log storage is enabled.

This pipe is one component of your system. Your overall security posture also depends on Open WebUI configuration, your database/storage security, and your network controls.

---

## Secrets management

### OpenRouter API key

You can provide the OpenRouter API key via:
- Environment variable: `OPENROUTER_API_KEY`, or
- Valve: `API_KEY` (type `EncryptedStr`).

Operational guidance:
- Prefer environment injection for infrastructure-managed secrets.
- If you store secrets in valves, configure `WEBUI_SECRET_KEY` so Open WebUI can store `EncryptedStr` values encrypted at rest.

### Protecting secret valve values at rest (`WEBUI_SECRET_KEY`)

The pipe defines an `EncryptedStr` wrapper used by sensitive valves such as:
- `API_KEY`
- `ARTIFACT_ENCRYPTION_KEY`
- `SESSION_LOG_ZIP_PASSWORD`

`EncryptedStr` encrypts/decrypts values using a key derived from the `WEBUI_SECRET_KEY` environment variable:

- If `WEBUI_SECRET_KEY` is **set**, `EncryptedStr.encrypt()` can store values prefixed with `encrypted:` and `EncryptedStr.decrypt()` returns the plaintext at runtime.
- If `WEBUI_SECRET_KEY` is **not set**, `EncryptedStr` behaves like a normal string: values remain plaintext and `decrypt()` returns the original value.

Recommended operator action:
- Set a strong `WEBUI_SECRET_KEY` in production deployments where valves may contain secrets.

Example:

```bash
export WEBUI_SECRET_KEY="$(openssl rand -base64 32)"
```

**Important:** `WEBUI_SECRET_KEY` protects *secret valve storage*. It does not, by itself, enable or disable artifact encryption. Artifact encryption is controlled by `ARTIFACT_ENCRYPTION_KEY` and related valves (next section).

**Warning:** If a secret valve value is stored with the `encrypted:` prefix but `WEBUI_SECRET_KEY` is missing or does not match the key used when the value was stored, Open WebUI will be unable to decrypt it at runtime. In that situation, `EncryptedStr.decrypt()` returns a different string than the intended plaintext. Operationally, this can cause:

- Provider authentication failures (if `API_KEY` cannot be recovered).
- A different artifact storage table namespace (if `ARTIFACT_ENCRYPTION_KEY` cannot be recovered), making previously persisted artifacts appear “missing” until the correct `WEBUI_SECRET_KEY` is restored.
- Session log archives being written with an unintended zip password (if `SESSION_LOG_ZIP_PASSWORD` cannot be recovered), complicating incident response.

---

## Artifact encryption at rest (database persistence)

The pipe can persist response artifacts (reasoning payloads, tool results, and related structured items) to the Open WebUI database. Pipe-level encryption of persisted artifacts is controlled by:

- `ARTIFACT_ENCRYPTION_KEY` (enables encryption when non-empty)
- `ENCRYPT_ALL` (default `True`)
- `ENABLE_LZ4_COMPRESSION` and `MIN_COMPRESS_BYTES` (optional compression for stored payloads)

See also: [Persistence, Encryption & Storage](persistence_encryption_and_storage.md).

### When encryption is active

- If `ARTIFACT_ENCRYPTION_KEY` is empty/unset, persisted artifacts are stored as plaintext JSON.
- If `ARTIFACT_ENCRYPTION_KEY` is set (non-empty), the pipe encrypts artifacts before persistence.
  - When `ENCRYPT_ALL=True`, *all* persisted artifact types are encrypted.
  - When `ENCRYPT_ALL=False`, only reasoning artifacts are encrypted; other artifacts remain plaintext.

### Table naming and key rotation implications

The artifact table name includes a short hash derived from `(ARTIFACT_ENCRYPTION_KEY + pipe_identifier)`, which means:

- Rotating `ARTIFACT_ENCRYPTION_KEY` results in a different table name.
- Old artifacts remain in the database, but the pipe will read/write using the table corresponding to the currently configured key.

Operational impact:
- Key rotation can intentionally reduce historical artifact replay (older marker references will not resolve unless you restore the prior key).
- Plan rotations as an operational change and communicate the impact to users if you rely on long-lived artifact replay.

### Recommended configurations (security vs operational cost)

| Mode | `ARTIFACT_ENCRYPTION_KEY` | `ENCRYPT_ALL` | Intended use |
| --- | --- | --- | --- |
| No pipe-level artifact encryption | empty | n/a | Development and low-risk deployments where DB/storage is already strongly protected and artifact sensitivity is low. |
| Reasoning-only encryption | set | `False` | Reduce sensitivity exposure while limiting encryption overhead to reasoning payloads. |
| Full artifact encryption | set | `True` | Multi-tenant deployments and environments where persisted tool outputs/reasoning may contain sensitive data. |

---

## SSRF protection for remote downloads

Remote file/image/video URLs are security-sensitive because they can be used for SSRF (Server-Side Request Forgery).

### Supported URL schemes

The pipe’s remote download subsystem only accepts:
- `http://`
- `https://`

Other schemes are rejected.

### SSRF guard behavior

When `ENABLE_SSRF_PROTECTION=True` (default):
- The pipe blocks remote fetches to private/internal address ranges (loopback, RFC1918, link-local, multicast/reserved ranges, etc.).
- Downloads that fail SSRF checks are rejected and logged; the pipe proceeds without crashing the request.

When `ENABLE_SSRF_PROTECTION=False`:
- The pipe may attempt to fetch internal URLs reachable from your Open WebUI environment. Only disable SSRF protection with a clear threat model and compensating controls.

### Additional mitigations for downloads

Even when a URL passes SSRF checks, downloads are constrained by:
- `REMOTE_FILE_MAX_SIZE_MB` (and optional Open WebUI RAG upload caps)
- `REMOTE_DOWNLOAD_*` retry/time budget valves
- `BASE64_MAX_SIZE_MB` and `VIDEO_MAX_SIZE_MB` for certain inline/base64 payloads

Recommended operator action:
- Keep SSRF protection enabled.
- Apply outbound egress controls at the network layer (proxy allowlists, egress firewall rules).
- If you must forbid plaintext `http://` downloads, enforce that via egress policy; the pipe permits both `http://` and `https://`.

---

## Session log storage security (optional)

When `SESSION_LOG_STORE_ENABLED=True`, the pipe can persist per-request session logs to encrypted zip archives on disk.

Security considerations:
- Archives are encrypted using `SESSION_LOG_ZIP_PASSWORD` (treat as a secret).
- Archives are written under `SESSION_LOG_DIR` with a predictable hierarchy (use filesystem permissions accordingly).
- Retention and cleanup are controlled by `SESSION_LOG_RETENTION_DAYS` and the cleanup interval valve.

See: [Session Log Storage](session_log_storage.md).

---

## Multi-tenant considerations

### Isolation and identifiers

The pipe uses request-scoped `contextvars` to keep per-request state isolated across concurrent requests.

If you run a shared deployment, you may enable OpenRouter attribution identifiers (for example `user` and `session_id`) and/or attach Open WebUI identifiers into OpenRouter `metadata` via valves. This helps incident response and abuse attribution but can increase privacy risk if you forward unnecessary identifiers.

See: [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md).

### Storage and persistence boundaries

- Artifact persistence is stored in per-pipe database tables (keyed by pipe ID and encryption key hash).
- Redis caching (when enabled) uses per-pipe namespaces for keys.

Operator guidance:
- Treat database backups and Redis access as sensitive.
- Limit operator access to Open WebUI admin features that can reveal stored artifacts or logs.

---

## Compliance guidance (non-guarantee)

This project can help implement controls (encryption, retention, logging, SSRF mitigation), but it does not by itself guarantee GDPR/HIPAA/SOC2/PCI compliance. Treat compliance as an end-to-end system property and validate in your environment (data flows, access controls, retention, incident response, and audit readiness).

---

## Quick reference (recommended baseline)

For a typical production deployment:

1. Set `WEBUI_SECRET_KEY` so secret valve values can be encrypted at rest.
2. Configure `OPENROUTER_API_KEY` (env) or `API_KEY` (valve).
3. If you persist artifacts, set `ARTIFACT_ENCRYPTION_KEY` and keep `ENCRYPT_ALL=True` unless you have a clear reason to encrypt reasoning only.
4. Keep `ENABLE_SSRF_PROTECTION=True` and enforce outbound egress policy.
5. Review retention (`ARTIFACT_CLEANUP_DAYS`, session log retention) and validate it matches your operational requirements.
