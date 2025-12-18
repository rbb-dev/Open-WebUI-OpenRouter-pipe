# Security & Encryption

**file:** `docs/security_and_encryption.md`
**related docs:**
- [Persistence, Encryption & Storage](persistence_encryption_and_storage.md) – Database schema and encryption implementation
- [Production Readiness Report](production_readiness_report.md#secrets-management) – Security audit and compliance
- [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md#ssrf-protection) – SSRF protection details

This document provides comprehensive security guidance for production deployments of the OpenRouter Responses pipe, covering encryption requirements, key rotation procedures, SSRF protection, secret management, and multi-tenant isolation.

---

## Overview

The OpenRouter Responses pipe implements multiple layers of security:

1. **Encryption at rest** - Fernet encryption for sensitive artifacts (reasoning tokens, tool outputs)
2. **SSRF protection** - Blocks malicious URLs in remote file downloads
3. **Secret valve handling** - Custom `EncryptedStr` type for encrypted storage of API keys and sensitive configuration
4. **Session isolation** - Request-scoped contextvars prevent data leakage between users
5. **Redis security** - Optional encrypted cache for multi-worker deployments
6. **Size limits** - Configurable MB caps prevent resource exhaustion

---

## Encryption Requirements

### Critical: WEBUI_SECRET_KEY is Required

**`WEBUI_SECRET_KEY`** (environment variable) is the **master encryption key** for all security features. Without it:

- ❌ All artifacts are stored **unencrypted** in the database (reasoning tokens, tool outputs, citations)
- ❌ Valve secrets (`API_KEY`, `ARTIFACT_ENCRYPTION_KEY`) are stored as **plaintext**
- ❌ The pipe logs warnings but continues operating in degraded security mode

**Why this matters:**
- The pipe implements a custom `EncryptedStr` type that uses `WEBUI_SECRET_KEY` to encrypt/decrypt valve values
- The pipe uses `EncryptedStr.decrypt()` to unwrap `ARTIFACT_ENCRYPTION_KEY` at runtime
- If the environment variable is missing, `EncryptedStr.decrypt()` returns plaintext silently

**Setting WEBUI_SECRET_KEY:**

```bash
# Generate a strong secret (recommended: 32+ characters)
openssl rand -base64 32

# Set in your Open WebUI environment
export WEBUI_SECRET_KEY="your-generated-secret-here"

# Or add to docker-compose.yml / systemd service
WEBUI_SECRET_KEY=your-generated-secret-here
```

### ARTIFACT_ENCRYPTION_KEY Configuration

**`ARTIFACT_ENCRYPTION_KEY`** (valve, ≥16 characters) enables Fernet encryption for stored artifacts.

**Location:** Admin → Functions → OpenRouter Responses Pipe → Valves → `ARTIFACT_ENCRYPTION_KEY`

**Format:** String, minimum 16 characters recommended (no maximum)

**Example values:**
```
production-encryption-key-2024
my-secure-artifact-key-v2
abc123def456ghi789jkl012mno345pqr
```

**How it works:**
1. The pipe derives a Fernet key using: `SHA256(ARTIFACT_ENCRYPTION_KEY + pipe_id)`
2. All reasoning tokens are encrypted before database storage
3. When `ENCRYPT_ALL=True` (the default once an encryption key is configured), tool outputs and citations are also encrypted
4. Table name includes key hash (first 8 chars): `response_items_openrouter_6d9a1b2c`

**Key derivation code reference:** See [Persistence, Encryption & Storage](persistence_encryption_and_storage.md#encryption-mechanics) for implementation details.

---

## Encryption Modes

### Mode 1: No Encryption (Default)

**Configuration:**
- `WEBUI_SECRET_KEY`: Not set or empty
- `ARTIFACT_ENCRYPTION_KEY`: Not set or empty

**Behavior:**
- All artifacts stored as plaintext JSON in database
- Suitable for: Development, testing, single-user deployments with no sensitive data

**Logs:**
```
WARNING: WEBUI_SECRET_KEY not configured; encrypted valves will be stored as plaintext
INFO: Artifact table ready: response_items_openrouter_default (no encryption)
```

### Mode 2: Reasoning-Only Encryption (Optional)

**Configuration:**
- `WEBUI_SECRET_KEY`: ✅ Set
- `ARTIFACT_ENCRYPTION_KEY`: ✅ Set (≥16 chars)
- `ENCRYPT_ALL`: ❌ False (explicit override – default is True)

**Behavior:**
- Reasoning tokens are encrypted (can be 100K+ tokens, high sensitivity)
- Tool outputs, citations, and metadata stored as plaintext JSON
- Balances security with performance

**Logs:**
```
INFO: Artifact table ready: response_items_openrouter_6d9a1b2c (key hash: 6d9a1b2c). Changing ARTIFACT_ENCRYPTION_KEY creates a new table; old artifacts become inaccessible.
```

**Why this mode:**
- Reasoning tokens contain internal model thought processes (potentially sensitive)
- OpenRouter charges for reasoning tokens separately; hiding them prevents reverse-engineering prompts
- Tool outputs are typically less sensitive (web search results, function returns)

### Mode 3: Full Encryption (Maximum Security, Default when keys are set)

**Configuration:**
- `WEBUI_SECRET_KEY`: ✅ Set
- `ARTIFACT_ENCRYPTION_KEY`: ✅ Set (≥16 chars)
- `ENCRYPT_ALL`: ✅ True (default)

**Behavior:**
- Everything encrypted: reasoning tokens, tool outputs, citations, metadata
- Maximum security for multi-tenant or regulated deployments
- Minimal performance impact on modern hardware: ~0.1ms per artifact (tested with 5KB payloads)

**Use cases:**
- Healthcare (HIPAA compliance)
- Financial services (PCI-DSS compliance)
- Government deployments
- Multi-tenant SaaS with strict data isolation requirements

---

## Key Rotation Procedures

### When to Rotate Keys

**Rotate `ARTIFACT_ENCRYPTION_KEY` when:**
- Employee with key access leaves organization
- Key is accidentally committed to version control
- Migrating between environments (dev → staging → prod)
- Decommissioning a cluster (intentionally strand old data)
- Suspected compromise or security incident

**Rotate `WEBUI_SECRET_KEY` when:**
- Database backup is compromised (encrypted valves are compromised)
- Open WebUI admin account is compromised
- Upgrading Open WebUI major versions (check release notes)

### Key Rotation Impact

**Rotating `ARTIFACT_ENCRYPTION_KEY`:**
- ✅ Creates new table: `response_items_openrouter_<new_hash>`
- ❌ Old table remains: `response_items_openrouter_<old_hash>`
- ❌ Existing chats **cannot** access old reasoning/tool artifacts (intentional)
- ✅ New conversations work normally with new encryption key

**Rotating `WEBUI_SECRET_KEY`:**
- ❌ All valve secrets must be re-entered (API keys, encryption keys, passwords)
- ❌ Open WebUI sessions are invalidated (users must re-login)
- ❌ All pipes lose access to encrypted configuration (must reconfigure)

### Rotation Procedure: ARTIFACT_ENCRYPTION_KEY

**Step 1: Announce maintenance window**
```
Subject: Encryption key rotation scheduled
Duration: 5 minutes downtime
Impact: Existing chat history will lose access to reasoning/tool outputs.
       New conversations will work normally.
```

**Step 2: Export critical conversations (optional)**
```bash
# If you need to preserve specific conversations, export them first
# (Manual process via Open WebUI UI: Chat → Export)
```

**Step 3: Update the key**
1. Open WebUI: Admin → Functions → OpenRouter Responses Pipe → Valves
2. Change `ARTIFACT_ENCRYPTION_KEY` to new value (≥16 chars)
3. Save

**Step 4: Restart workers**
```bash
# Docker Compose
docker compose restart open-webui

# Standalone Docker container
docker restart open-webui

# Systemd
systemctl restart open-webui

# Kubernetes
kubectl rollout restart deployment/open-webui
```

**Step 5: Verify new table**
```bash
# Check logs for new table name
grep "Artifact table ready" /var/log/openwebui/*.log

# Expected output:
# INFO: Artifact table ready: response_items_openrouter_a1b2c3d4 (key hash: a1b2c3d4)
```

**Step 6: Clean up old table (after retention period)**
```sql
-- IMPORTANT: This step is MANUAL and OPTIONAL
-- The automatic cleanup worker (ARTIFACT_CLEANUP_INTERVAL_HOURS) only prunes old
-- artifact ROWS within the current table based on TOOL_OUTPUT_RETENTION_TURNS.
-- It does NOT drop entire tables after key rotation.

-- Wait 30-90 days (your retention policy) to ensure all old conversations are archived.
-- The old table remains accessible for conversation history until you drop it.
-- Once dropped, any chat that references artifacts in the old table will lose
-- access to reasoning tokens and tool outputs from before the key rotation.

-- To reclaim disk space after your retention period:
DROP TABLE IF EXISTS response_items_openrouter_<old_hash>;

-- Example: If old table was response_items_openrouter_6d9a1b2c
DROP TABLE IF EXISTS response_items_openrouter_6d9a1b2c;
```

### Rotation Procedure: WEBUI_SECRET_KEY

**⚠️ WARNING:** This is a destructive operation. All encrypted valves become inaccessible.

**Step 1: Document all encrypted valves**
```bash
# Export valve configuration before rotation
# Admins must manually record:
# - OPENROUTER_API_KEY
# - ARTIFACT_ENCRYPTION_KEY
# - Any custom encrypted valves
```

**Step 2: Generate new key**
```bash
NEW_SECRET=$(openssl rand -base64 32)
echo "New WEBUI_SECRET_KEY: $NEW_SECRET"
```

**Step 3: Update environment**
```bash
# Update docker-compose.yml or systemd service
WEBUI_SECRET_KEY=$NEW_SECRET

# Restart Open WebUI
docker compose restart open-webui
```

**Step 4: Re-enter all valve secrets**
1. Admin → Functions → OpenRouter Responses Pipe → Valves
2. Re-enter `OPENROUTER_API_KEY`
3. Re-enter `ARTIFACT_ENCRYPTION_KEY` (triggers new table creation)
4. Re-enter any other encrypted valves
5. Save

**Step 5: Test functionality**
```bash
# Send a test message to verify encryption is working
# Check logs for "Artifact table ready" confirmation
```

---

## SSRF Protection

### Remote File Download Security

The pipe downloads remote files from URLs in multimodal messages (images, documents, audio, video). **SSRF protection** prevents attackers from:

- Accessing internal services (e.g., `http://localhost:6379` → Redis)
- Reading cloud metadata endpoints (e.g., `http://169.254.169.254/latest/meta-data/`)
- Port scanning internal networks
- Exfiltrating data via DNS queries

**Implementation:** See [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md#ssrf-protection) for complete SSRF protection details.

### Blocked URL Patterns

**Private IP ranges** (RFC 1918):
- `10.0.0.0/8`
- `172.16.0.0/12`
- `192.168.0.0/16`

**Loopback addresses:**
- `127.0.0.0/8` (IPv4)
- `::1` (IPv6)

**Link-local addresses:**
- `169.254.0.0/16` (AWS metadata endpoint)
- `fe80::/10` (IPv6 link-local)

**Special-use addresses:**
- `0.0.0.0/8`
- `224.0.0.0/4` (multicast)
- `240.0.0.0/4` (reserved)

### Additional Protections

**1. Scheme validation:**
```python
# Only HTTPS allowed (HTTP blocked for security)
if not url.startswith("https://"):
    logger.warning(f"Blocked non-HTTPS URL: {url}")
    return None
```

**2. Redirect following:**
- Maximum 5 redirects allowed (`max_redirects=5`)
- Each redirect re-validated against SSRF rules
- Prevents redirect-based SSRF bypasses

**3. Size limits:**
- `REMOTE_FILE_MAX_SIZE_MB` valve (default: 50MB)
- Automatically clamped to Open WebUI's `FILE_MAX_SIZE` when RAG uploads are enabled
- Streaming validation: aborts download if `Content-Length` exceeds limit

**4. Timeout enforcement:**
- Per-request timeout: `HTTP_TOTAL_TIMEOUT_SECONDS` (default: 120s)
- Prevents slowloris attacks and resource exhaustion

**5. User-Agent header:**
```python
"User-Agent": "OpenWebUI-OpenRouter-Pipe/1.0 (SSRF-Protected)"
```

### Customization (Advanced)

**Allowing internal URLs (not recommended):**
```python
# If you need to allow specific internal services (e.g., internal S3)
# Edit _download_remote_url() to whitelist specific hosts:

if url.startswith("https://internal-s3.company.local/"):
    # Skip SSRF check for trusted internal service
    pass
else:
    # Apply normal SSRF validation
    if _is_ssrf_attempt(parsed_url):
        return None
```

---

## Secret Valve Management

### EncryptedStr Type

The pipe implements a custom `EncryptedStr` type for sensitive valves:

**Encrypted valves in this pipe:**
- `OPENROUTER_API_KEY` (system valve)
- `ARTIFACT_ENCRYPTION_KEY` (system valve)

**How it works:**
1. Admin enters plaintext value in Open WebUI UI
2. The pipe's `EncryptedStr.encrypt()` encrypts it using `WEBUI_SECRET_KEY` before storage
3. Pipe calls `EncryptedStr.decrypt()` at runtime to get plaintext value
4. If `WEBUI_SECRET_KEY` is missing, encryption/decryption is skipped (degrades gracefully)

**Code reference:** See [Persistence, Encryption & Storage](persistence_encryption_and_storage.md#encryptedstr-handling) for `EncryptedStr` implementation details.

### Best Practices

**1. Never log decrypted secrets:**
```python
# ❌ WRONG
logger.info(f"Using API key: {EncryptedStr.decrypt(self.valves.API_KEY)}")

# ✅ CORRECT
logger.info("Using API key: [REDACTED]")
```

**2. Validate secret strength:**
```python
api_key = EncryptedStr.decrypt(self.valves.OPENROUTER_API_KEY)
if not api_key or len(api_key) < 32:
    raise ValueError("OPENROUTER_API_KEY must be at least 32 characters")
```

**3. Rotate secrets regularly:**
- API keys: Every 90 days (or per provider policy)
- Encryption keys: Every 180 days (or when personnel changes)

**4. Use environment variables for CI/CD:**
```bash
# Set secrets via environment instead of valves
export OPENROUTER_API_KEY="sk-or-v1-..."
export ARTIFACT_ENCRYPTION_KEY="production-key-2024"
```

---

## Multi-Tenant Isolation

### Session-Scoped ContextVars

The pipe uses Python's `contextvars` to isolate requests:

**Implementation:** See [Concurrency Controls & Resilience](concurrency_controls_and_resilience.md#session-isolation) for complete details on request isolation.

**Isolated per request:**
- Session ID (`SessionLogger.session_id`)
- User ID (`SessionLogger.user_id`)
- Request start time
- Valve snapshots

**Why this matters:**
- Concurrent requests from different users never share state
- Logging includes session/user context for audit trails
- Breaker windows are per-user (one user's failures don't affect others)

### Database Isolation

**Artifact storage:**
- Each pipe instance gets its own table: `response_items_<pipe_id>_<key_hash>`
- Multiple tenants can run separate pipe instances with different IDs
- No cross-contamination between tenants

**User ID enforcement:**
- All artifacts tagged with `user_id` from `__user__["id"]`
- Open WebUI enforces user access control before pipe invocation
- Pipe respects Open WebUI's authentication layer

### Redis Isolation (Multi-Worker)

When Redis is enabled (`UVICORN_WORKERS>1`):

**Namespace isolation:**
```python
cache_key = f"{self.id}:catalog:{model_id}"
# Example: "openrouter-responses:catalog:anthropic/claude-3-opus"
```

**Why this matters:**
- Multiple pipe instances share same Redis server without conflicts
- Model catalog cache is per-pipe (different API keys = different catalogs)
- TTL-based eviction prevents stale data

---

## File Upload Security

### Open WebUI Storage Integration

The pipe uses Open WebUI's existing file storage system:

**Storage paths:**
```
/app/backend/data/uploads/<user_id>/<file_id>
```

**Permissions:**
- Files owned by `user_id` (from `__user__["id"]`)
- Open WebUI's access control layer enforces read permissions
- Fallback storage user: `FALLBACK_STORAGE_EMAIL` (default: `openrouter-pipe@system.local`)

**Code reference:** See [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md#file-upload-storage) for complete file upload implementation.

### Size Limits

**Remote file downloads:**
- `REMOTE_FILE_MAX_SIZE_MB` valve (default: 50MB)
- Auto-clamped to Open WebUI's `FILE_MAX_SIZE` when RAG is enabled
- See [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md#size-guards) for size validation logic

**Base64 inline payloads:**
- `BASE64_MAX_SIZE_MB` valve (default: 50MB)
- Validates size before decoding to prevent memory exhaustion

**Video files:**
- `VIDEO_MAX_SIZE_MB` valve (default: 100MB)
- Separate limit for video content (typically larger than images)

### MIME Type Validation

**Allowed image types:**
```python
ALLOWED_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif"
}
```

**Blocked types:**
- Executable formats (`.exe`, `.sh`, `.bat`)
- Archives (`.zip`, `.tar`, `.gz`) - prevents zip bombs
- HTML files (prevents JavaScript injection)

---

## Operational Security

### Logging & Audit Trails

**What's logged:**
- All requests: session ID, user ID, model, elapsed time
- Encryption events: key derivation, table creation, encryption failures
- SSRF blocks: attempted URL, reason, user ID
- Error events: exception type, error ID, session context

**Log format:**
```
INFO [sess_abc123] [user_789] Starting request for model: anthropic/claude-3-opus
ERROR [abc123] [sess_abc123] [user_789] SSRF attempt blocked: http://169.254.169.254/latest/meta-data/
INFO [sess_abc123] Artifact table ready: response_items_openrouter_6d9a1b2c (key hash: 6d9a1b2c)
```

**Log levels:**
- `DEBUG`: Verbose tracing (disable in production)
- `INFO`: Normal operations, encryption setup
- `WARNING`: Degraded security (missing WEBUI_SECRET_KEY), SSRF blocks
- `ERROR`: Failures, exceptions, security incidents

### Monitoring Checklist

**Daily checks:**
- [ ] No SSRF warnings in logs
- [ ] Encryption key hash matches expected value
- [ ] No plaintext storage warnings
- [ ] API key rotation dates tracked

**Weekly checks:**
- [ ] Artifact table size growth (check for storage exhaustion)
- [ ] Redis cache hit rate (if multi-worker)
- [ ] Failed request rate by user

**Monthly checks:**
- [ ] Review API key usage costs (OpenRouter billing)
- [ ] Rotate `ARTIFACT_ENCRYPTION_KEY` if required by policy
- [ ] Clean up old artifact tables (after retention period)

### Security Incident Response

**If `ARTIFACT_ENCRYPTION_KEY` is compromised:**
1. Rotate key immediately (see rotation procedure above)
2. Notify affected users (old chat history inaccessible)
3. Review access logs for suspicious activity
4. Consider rotating `WEBUI_SECRET_KEY` if admin access compromised

**If API key is compromised:**
1. Revoke key at OpenRouter dashboard
2. Generate new API key
3. Update `OPENROUTER_API_KEY` valve
4. Review OpenRouter usage logs for unauthorized requests

**If SSRF exploit is discovered:**
1. Check logs for exploit attempts: `grep "SSRF" /var/log/openwebui/*.log`
2. Identify affected users: extract session IDs from logs
3. Block malicious user accounts if needed
4. Update SSRF rules if bypass discovered

---

## Compliance & Certifications

### GDPR Compliance

**Data minimization:**
- Only stores artifacts needed for conversation continuity
- Cleanup workers prune old artifacts via `TOOL_OUTPUT_RETENTION_TURNS`

**Right to erasure:**
```sql
-- Delete all artifacts for a specific user
DELETE FROM response_items_openrouter_<hash> WHERE user_id = 'user_123';
```

**Data portability:**
- Artifacts stored as JSON (Fernet-encrypted or plaintext)
- Decryption utility can export user data on request

### HIPAA Compliance

**Required configuration:**
- `WEBUI_SECRET_KEY`: ✅ Set
- `ARTIFACT_ENCRYPTION_KEY`: ✅ Set
- `ENCRYPT_ALL`: ✅ True
- Redis encryption: ✅ Enable TLS (external to pipe)

**Audit trail:**
- All requests logged with user ID, session ID, timestamps
- Error IDs correlate user reports with backend logs

### SOC 2 Type II

**Access control:**
- Open WebUI's authentication layer enforced before pipe invocation
- Session isolation via contextvars prevents cross-contamination

**Encryption at rest:**
- Fernet encryption for sensitive artifacts
- Key rotation procedures documented

**Monitoring:**
- SessionLogger provides per-request audit trail
- Error IDs enable incident correlation

---

## Quick Reference

### Encryption Setup

```bash
# 1. Generate and set WEBUI_SECRET_KEY
openssl rand -base64 32 | xargs -I {} echo "WEBUI_SECRET_KEY={}" >> .env

# 2. Restart Open WebUI
# Docker Compose:
docker compose restart open-webui
# Standalone Docker:
docker restart open-webui

# 3. Set ARTIFACT_ENCRYPTION_KEY via UI
# Admin → Functions → OpenRouter Responses Pipe → Valves → ARTIFACT_ENCRYPTION_KEY
# Value: "production-encryption-key-2024" (minimum 16 chars)

# 4. Confirm full encryption (default)
# Admin → Functions → OpenRouter Responses Pipe → Valves → ENCRYPT_ALL (leave True unless you explicitly need reasoning-only mode)

# 5. Verify encryption is active
docker logs open-webui 2>&1 | grep "Artifact table ready"
# Expected: "key hash: 6d9a1b2c" (not "no encryption")
```

### Key Rotation Cheat Sheet

```bash
# Rotate ARTIFACT_ENCRYPTION_KEY (low impact)
1. Announce maintenance window
2. Update valve in Open WebUI UI
3. Restart workers
4. Verify new table: grep "Artifact table ready" logs
5. Clean up old table after 30-90 days

# Rotate WEBUI_SECRET_KEY (high impact)
1. Document all encrypted valve values
2. Generate new key: openssl rand -hex 32
3. Update environment and restart
4. Re-enter all valve secrets in UI
5. Test functionality
```

### Security Checklist

**Production deployment:**
- [ ] `WEBUI_SECRET_KEY` set (≥32 chars)
- [ ] `ARTIFACT_ENCRYPTION_KEY` set (≥16 chars)
- [ ] `ENCRYPT_ALL=True` (default; only set to False if you intentionally need reasoning-only encryption)
- [ ] SSRF protection enabled (default, do not disable)
- [ ] TLS enabled for Open WebUI (external to pipe)
- [ ] Redis TLS enabled if multi-worker (external to pipe)
- [ ] Log retention policy defined (30-90 days)
- [ ] Key rotation schedule defined (90-180 days)
- [ ] Incident response runbook prepared

**References:**
- [Valves & Configuration Atlas](valves_and_configuration_atlas.md) - All valve defaults
- [Persistence, Encryption & Storage](persistence_encryption_and_storage.md) - Database schema
- [Production Readiness Report](production_readiness_report.md) - Security audit
- [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md) - SSRF protection details


> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## Related Topics

**Implementation Details:**
- **SSRF Protection**: [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md#ssrf-protection) - Technical implementation of URL validation and safe downloading
- **Encryption Implementation**: [Persistence, Encryption & Storage](persistence_encryption_and_storage.md) - Database schema, key derivation, encryption lifecycle
- **Configuration**: [Valves and Configuration Atlas](valves_and_configuration_atlas.md) - Security-related valve reference

**Operations & Compliance:**
- **Production Audit**: [Production Readiness Report](production_readiness_report.md) - End-to-end security checklist and compliance verification
- **Testing & Deployment**: [Testing, Bootstrap, and Operations](testing_bootstrap_and_operations.md) - Security testing procedures
- **Error Handling**: [Error Handling and User Experience](error_handling_and_user_experience.md) - Secure error messaging without information leakage

**Architecture:**
- **System Overview**: [Developer Guide and Architecture](developer_guide_and_architecture.md) - Security boundaries and session isolation
- **Concurrency**: [Concurrency Controls and Resilience](concurrency_controls_and_resilience.md) - Thread safety and isolation guarantees

