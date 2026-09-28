# Security & Encryption

This document provides security guidance for production deployments of the OpenRouter Responses pipe, covering secrets handling, artifact encryption at rest, SSRF protections for remote downloads, and operational controls for multi-user environments.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Persistence](persistence_encryption_and_storage.md) · [Multimodal](multimodal_ingestion_pipeline.md) · [Identifiers](request_identifiers_and_abuse_attribution.md)

---

## Overview (what the pipe does for security)

Security-relevant mechanisms implemented by the pipe include:

1. **Secret valve handling** via `EncryptedStr` (optionally encrypts secret valve values at rest when `WEBUI_SECRET_KEY` is set).
2. **Artifact encryption at rest** for persisted response artifacts when `ARTIFACT_ENCRYPTION_KEY` is configured.
3. **SSRF protection** for remote URL downloads when `ENABLE_SSRF_PROTECTION=True`, plus HTTPS-only defaults for remote URLs (HTTP allowlist available).
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

At 44 characters the two sides still derive different keys, and the divergence is harmless: the pipe's key is always the SHA-256-derived one, while Open WebUI uses a 44-character secret verbatim for its own valve column. Open WebUI encrypts the whole valves dict as one blob and decrypts it itself before the pipe sees a value, so the column's own ciphertext can never arrive as a field value. (The pipe only inspects that column to tell an unset configuration from an undecodable one; it never decrypts it.) A secret field's *value* inside that blob is the pipe's own `EncryptedStr` ciphertext, which is why the two derivations can differ harmlessly.

Recommended operator action:
- Set a strong `WEBUI_SECRET_KEY` in production deployments where valves may contain secrets.

Example:

```bash
export WEBUI_SECRET_KEY="$(openssl rand -base64 32)"
```

**Important:** `WEBUI_SECRET_KEY` protects *secret valve storage*. It does not, by itself, enable or disable artifact encryption. Artifact encryption is controlled by `ARTIFACT_ENCRYPTION_KEY` and related valves (next section).

**Warning:** If a secret valve value is stored with the `encrypted:` prefix but `WEBUI_SECRET_KEY` is missing, `EncryptedStr.decrypt()` strips the `encrypted:` prefix and returns the raw (still-encrypted) ciphertext; if `WEBUI_SECRET_KEY` is set but does not match the key used when the value was stored, `decrypt()` returns the original `encrypted:...` string unchanged. In either case the plaintext is not recovered, which can cause:

- Provider authentication failures (if `API_KEY` cannot be recovered).
- Every user's own settings falling back to their per-user default until the old key is restored. A valve row that will not decode is reported as wholly unreadable, and each of its fields then takes that user valve's own default — the **more restrictive** side for the fields whose polarity matters, such as `PERSIST_TOOL_RESULTS` and `PERSIST_REASONING_TOKENS`. The administrator's more permissive site-wide value does **not** fill in behind it, so rotating the key cannot widen what any user's chats hand to a model; it does mean every affected user is on defaults until the old key is restored, which is the cost of not guessing on their behalf. `REQUEST_ZDR` is the one field with the opposite polarity, and it keeps its own gate (see [OpenRouter ZDR](openrouter_zdr.md)).
- Artifact storage stopping (if `ARTIFACT_ENCRYPTION_KEY` cannot be recovered). The pipe warns once and drops what it would have written, so no artifact content is stored in the clear, and no new artifact content is stored until the correct `WEBUI_SECRET_KEY` is restored. The coordination lock rows the pipe needs to keep assembling (and the dashboard purge locks) are still written, and they carry no content. Artifacts written under the previous secret stay unreadable until it is.
- Session log archives being skipped (if `SESSION_LOG_ZIP_PASSWORD` cannot be recovered), which keeps them from being written under an unintended zip password.

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

Remote picture and video URLs are security-sensitive because they can be used for SSRF (Server-Side Request Forgery).

### Supported URL schemes

The pipe’s remote download subsystem accepts:
- `https://` (default)
- `http://` only when explicitly allowlisted via `ALLOW_INSECURE_HTTP` + `ALLOW_INSECURE_HTTP_HOSTS`

Other schemes are rejected.

### SSRF guard behavior

When `ENABLE_SSRF_PROTECTION=True` (default):
- The pipe fetches an address only when it is provably globally routable. Everything else is refused, rather than only the ranges someone remembered to list: loopback, RFC1918, link-local, multicast, reserved and unspecified are refused, and so are carrier-grade NAT (`100.64.0.0/10`, which is also Tailscale's default range), IPv6 site-local (`fec0::/10`) and any address that is simply not marked as globally routable. A future range added to the registries is refused the day it is registered rather than the day someone updates a list here.
- IPv6 forms that carry an IPv4 address inside them are judged on the addresses they carry: `::ffff:`-mapped, 6to4 (`2002::/16`), Teredo (`2001::/32`) and the NAT64 well-known prefix (`64:ff9b::/96`). `2002:7f00:1::` is 6to4 for `127.0.0.1` and is refused for that reason; `2002:808:808::` carries a public address and is allowed. A form that carries more than one address is allowed only when EVERY address it carries would be allowed on its own: a Teredo address encodes the tunnel server as well as the client, both chosen by whoever wrote the address, so `2001:0:7f00:1::f7f7:f7f7` — server `127.0.0.1`, client `8.8.8.8` — is refused for the server.
- Downloads that fail SSRF checks are rejected and logged; the pipe proceeds without crashing the request.
- Address resolution runs on its own bounded thread pool, so a hostile or dead host — one whose nameserver never answers — cannot consume the threads serving media, thumbnails, file reads or Open WebUI's own endpoints. The pool is shared only with other address checks, so a stall is bounded there; a check that is stalled, or queued behind other checks, is refused rather than delayed.

When `ENABLE_SSRF_PROTECTION=False`:
- The pipe may attempt to fetch internal URLs reachable from your Open WebUI environment. Only disable SSRF protection with a clear threat model and compensating controls.
HTTPS-only defaults still apply even if SSRF protection is disabled.

### Address validation and redirects

Validation and connection are the same decision, so the pipe does not check a URL and
then let the HTTP client resolve the name again:

- The model-icon, maker-profile and self-update fetches share one transport, which owns
  its HTTP connection pool. Name resolution for that pool goes through the SSRF gate, so
  the addresses the gate approved are the addresses the pipe dials. A name that answers
  with a public address for the check and a private one a moment later cannot steer the
  connection, because there is no second lookup for it to answer.
- The URL keeps its hostname, so TLS certificate verification, SNI and the connection
  pool all key on the name the caller asked for.
- The host is read with the same parser the connection is built from, so the gate and
  the connector cannot disagree about what the URL names. This matters because the
  internationalised-name rules the HTTP client applies map several Unicode codepoints to
  `.`, and a host written with one of them is an ordinary name to one parser and an IP
  literal to the other. The pipe reads the dialled host, not the literal text.
- That transport does not use an HTTP proxy from the environment. A proxy resolves the
  name itself, so an address this pipe validated would not be the address dialled. If a
  proxy variable is set (`HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` or their lowercase
  spellings), the pipe logs one warning naming it, because on a deployment that can only
  reach GitHub through that proxy the self-update check will fail to connect and the
  Updates tab is the only other place that is reported.
- Picture and video downloads take a different path: they do not follow redirects at
  all, and they pin the validated address into the request.
- The three transport fetches do follow redirects, up to a small fixed number of hops.
  Every hop is validated the same way as the first, covering both the release metadata
  and the release asset in the self-update flow. A redirect to a private or internal
  address is refused before the connection is attempted. The transport is taken per
  HOP, not per chain, so a chain that is already in flight when the valve changes does
  not finish under the setting it started with: the origin decides how long to hold a
  hop open, which would otherwise be how long the old setting lasted.
- Turning `ENABLE_SSRF_PROTECTION` back on takes effect on the next hop, which is the
  next request or the rest of a redirect chain already under way. The transport is
  REBUILT when the valve changes: the old session is closed and a new
  connection pool, a new address cache and a new resolver are created together. Clearing
  the address cache alone was not enough, because a keep-alive connection opened while
  the gate was off is reused without consulting any resolver, and an answer from a lookup
  that was already in flight lands in the cache after it has been cleared.
- The gate also re-runs on a **re-use**: an image a worker already downloaded is
  answered from its in-process memo on later turns, and the memo does not stand in for
  the gate. Every turn re-checks the URL before those bytes are sent, and a URL the
  current valve values refuse is dropped from the memo and fetched through the gated
  path instead. So every byte that reaches the provider passed the gate under the valve
  values in force at the moment it was sent, and tightening the valve or editing a host
  allowlist takes effect on the next turn rather than the next fetch. The re-check
  costs one address resolution and no download; a URL the check then refuses is fetched
  once more through the already-gated path, which runs its own address check, so a
  refusal pays two resolutions. A resolution that does not finish inside the
  address-check budget counts as a refusal, so a slow nameserver costs a re-fetch
  rather than a permission. Nothing has to be refused for the re-check to be paid for:
  a request that reuses `MAX_INPUT_IMAGES_PER_REQUEST` pictures pays one blocking
  resolution per reused picture, one after another (5 by default, 20 at the ceiling).

### What a blocked address is recorded as

A refused address is logged with the host, the port that was targeted and the resolved
IP that failed the check, at WARNING the first time and DEBUG on repeats, with the
warning repeating after a cooldown rather than latching for the life of the worker. The
query string is deliberately not recorded: this gate is reached for any picture or video
URL pasted into a chat, and those can carry credentials.

### Host allowlists an operator can set for forwarded reference URLs

The SSRF gate above answers whether an address is *globally routable*. It does not answer
whether it is *this deployment's* host, so an operator who needs a tighter rule has two
valves, and they do not work the same way:

- `ALLOW_INSECURE_HTTP_HOSTS` — exact-match, `host:port` entries, gates plaintext `http://`
  downloads. `example.com` does **not** cover `www.example.com`.
- `VIDEO_REFERENCE_ALLOWED_DOMAINS` — parent-domain match, bare hosts, gates the per-user
  reference links a video filter forwards to OpenRouter. `example.com` **does** cover
  `cdn.example.com`, and does not cover `notexample.com`. It covers the filter's own
  reference fields and the free-text `provider.options` box alike, because the same check
  runs over every address in the built request rather than over a named field. Empty — the
  default — means unrestricted.

`VIDEO_REFERENCE_ALLOWED_DOMAINS` takes **no `!` block entries and no CIDR ranges**, unlike
the Open WebUI host filter whose parent-domain dialect it follows. It is an *additional*
restriction, not a replacement: the `https://`/SSRF check still runs and can refuse a link
this list allows, and an entry here never widens the scheme policy.

One link is outside its reach, and deliberately so: a reference the media relay published
for the current request. The relay records every link it uploads in a per-request record
and that record is passed to the check, so the pipe's own uploaded attachments are
forwarded whatever this list holds — excluding them by host would refuse every relayed
attachment on any deployment that sets the valve. A *host address a user typed* is not
recorded, so it is not exempt: `https://evil.catbox.moe/x.png` typed into `provider.options`
is refused like any other unlisted host, even though the pipe's own uploads live on that
same public host. A caller that populates no record gets no exemption at all.

### Additional mitigations for downloads

Even when a URL passes SSRF checks, downloads are constrained by:
- `REMOTE_FILE_MAX_SIZE_MB`, and the cap the Open WebUI admin last saved under Admin → Settings → Documents → Max Upload Size — normally the lower of the two applies, except that a `REMOTE_FILE_MAX_SIZE_MB` left at its 50 MB default gives way to a larger admin cap, clipped to the pipe’s own 500 MB ceiling; clearing the admin’s box lifts Open WebUI’s cap, not this valve’s
- `REMOTE_DOWNLOAD_*` retry/time budget valves
- `BASE64_MAX_SIZE_MB` and `VIDEO_MAX_SIZE_MB` for certain inline/base64 payloads

The three transport fetches carry their own fixed caps, because none of them is
operator-configurable and each follows redirects, so the body is chosen by whatever the
last hop was: the model icon, the maker profile page and the self-update metadata and
asset are each read in chunks against a running total and abandoned once it is passed. A
declared `Content-Length` over the cap is refused before any body is read, but it is only
an early-out — a chunked response declares nothing, and the running total is what
enforces the limit.

Recommended operator action:
- Keep SSRF protection enabled.
- Apply outbound egress controls at the network layer (proxy allowlists, egress firewall rules).
- HTTP is disabled by default; only enable plaintext `http://` with a narrow allowlist (`ALLOW_INSECURE_HTTP_HOSTS`) and compensating egress controls.
- To constrain which hosts a per-user video reference link may name, set `VIDEO_REFERENCE_ALLOWED_DOMAINS` (parent-domain match, and it governs the `provider.options` route as well as the filter fields). Remember it cannot govern a link the media relay published for the request, by design.

### Log safety

`_redact_payload_blobs()` runs over every request payload the pipe records, at any log level. It reduces two kinds of value:

- **Large base64 blobs** — a `data:` URL's payload, or a bare blob under a key such as `b64_json`, is truncated to a marker with its length. This prevents multi-megabyte log entries.
- **Media links** — any value under a media-URL key (`image_url`, `file_url`, `video_url`, `url`, `file_data`) is reduced to scheme, host and port by `loggable_link()`. The path and query are dropped, so a presigned or signed CDN link cannot be recovered from a log. This holds for a link whose netloc does not parse and for a `data:` URL with no comma as well as for a well-formed `https://` one: a value that cannot be described at all is replaced by `[REDACTED]` rather than echoed.

The ERROR path is covered too. A refusal's log subject is the link the pipe declined, so subjects are built with `loggable_link()` rather than the raw URL. This means turning DEBUG on shows you every media field as `scheme://host:port` rather than the full URL — the pipe is telling you which host it was about to contact, not the signed path it was given.

A `data:` URL contributes only its media type (for example `data:image/png`); the bytes after it never reach a log, including when the URL carries no comma and its payload is the remainder of the string.

---

## Session log storage security (optional)

When `SESSION_LOG_STORE_ENABLED=True`, the pipe can persist per-request session logs to encrypted zip archives on disk.

Security considerations:
- Archives are encrypted using `SESSION_LOG_ZIP_PASSWORD` (treat as a secret).
- Archives are written under `SESSION_LOG_DIR` with a predictable hierarchy (use filesystem permissions accordingly).
- A request that carries no usable `chat_id`/`message_id` — the plain API route — is archived under `api/api-<request_id>.zip` while `SESSION_LOG_ARCHIVE_API_CALLS` is on, so machine traffic is captured on the same terms as chat traffic. The key is the request id, so one file per request and never shared between two calls.
- Retention and cleanup are controlled by `SESSION_LOG_RETENTION_DAYS` and the cleanup interval valve, and run while storage is on. Turning `SESSION_LOG_STORE_ENABLED` off stops the sweep, so archives already on disk survive until it is re-enabled and their window passes.

See: [Session Log Storage](session_log_storage.md).

---

## Multi-tenant considerations

### Isolation and identifiers

The pipe uses request-scoped `contextvars` to keep per-request state isolated across concurrent requests.

If you run a shared deployment, you may enable the OpenRouter `user` attribution identifier and/or attach Open WebUI identifiers into OpenRouter `metadata` via valves. This helps incident response and abuse attribution but can increase privacy risk if you forward unnecessary identifiers.

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
4. Keep `ENABLE_SSRF_PROTECTION=True`, keep HTTPS-only defaults, and enforce outbound egress policy (only allow HTTP if explicitly allowlisted).
5. Review retention (`ARTIFACT_CLEANUP_DAYS`, session log retention) and validate it matches your operational requirements.
