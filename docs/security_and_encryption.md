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
- If `WEBUI_SECRET_KEY` is **not set**, `EncryptedStr` behaves like a normal string for every value it never encrypted: those stay plaintext and `decrypt()` returns the original value. A value that carries the `encrypted:` prefix was stored under a key that is no longer configured, so it reads as unset from every reader — `decrypt()` returns `""`, `read()` returns `None`, and the dashboard reports the field as unreadable. Nothing is rewritten, so restoring `WEBUI_SECRET_KEY` makes every such row readable again.

At 44 characters the two sides still derive different keys, and the divergence is harmless: the pipe's key is always the SHA-256-derived one, while Open WebUI uses a 44-character secret verbatim for its own valve column. Open WebUI encrypts the whole valves dict as one blob and decrypts it itself before the pipe sees a value, so the column's own ciphertext can never arrive as a field value. (The pipe only inspects that column to tell an unset configuration from an undecodable one; it never decrypts it.) A secret field's *value* inside that blob is the pipe's own `EncryptedStr` ciphertext, which is why the two derivations can differ harmlessly.

Recommended operator action:
- Set a strong `WEBUI_SECRET_KEY` in production deployments where valves may contain secrets.

Example:

```bash
export WEBUI_SECRET_KEY="$(openssl rand -base64 32)"
```

**Important:** `WEBUI_SECRET_KEY` protects *secret valve storage*. It does not, by itself, enable or disable artifact encryption. Artifact encryption is controlled by `ARTIFACT_ENCRYPTION_KEY` and related valves (next section).

**Warning:** A secret valve value that carries the `encrypted:` prefix and cannot be decoded reads as unset from **every** reader — `decrypt()` returns `""`, `read()` returns `None`, and the dashboard reports the field as unreadable. There are three ways to arrive there: `WEBUI_SECRET_KEY` is missing, `WEBUI_SECRET_KEY` is set but no longer matches the one the value was stored under, or the stored row is damaged after it was written. The plaintext is not recovered, which can cause:

- Provider authentication failures (if `API_KEY` cannot be recovered).
- Every user's own settings falling back to their per-user default until the old key is restored. A valve row that will not decode is reported as wholly unreadable, and each of its fields then takes that user valve's own default — the **more restrictive** side for the fields whose polarity matters, such as `PERSIST_TOOL_RESULTS` and `PERSIST_REASONING_TOKENS`. The administrator's more permissive site-wide value does **not** fill in behind it, so rotating the key cannot widen what any user's chats hand to a model; it does mean every affected user is on defaults until the old key is restored, which is the cost of not guessing on their behalf. `REQUEST_ZDR` is the one field with the opposite polarity, and it keeps its own gate (see [OpenRouter ZDR](openrouter_zdr.md)).
- Artifact storage stopping (if `ARTIFACT_ENCRYPTION_KEY` cannot be recovered). The pipe warns once and drops what it would have written, so no artifact content is stored in the clear, and no new artifact content is stored until the correct `WEBUI_SECRET_KEY` is restored. The coordination lock rows the pipe needs to keep assembling (and the dashboard purge locks) are still written, and they carry no content. Artifacts written under the previous secret stay unreadable until it is. A damaged `ARTIFACT_ENCRYPTION_KEY` row has the same effect and the same warning.
- Session log archives being skipped (if `SESSION_LOG_ZIP_PASSWORD` cannot be recovered), which keeps them from being written under an unintended zip password. A damaged `SESSION_LOG_ZIP_PASSWORD` row is skipped the same way.

One residual is worth naming. A passphrase the operator types as `encrypted:` followed by an all-base64 character body beginning with `g` is judged a stored ciphertext that does not decode, and is refused as though it were damaged — the refusal is loud, so the operator re-enters the secret without the prefix. The same refusal covers a genuinely damaged row, which is why both warnings name the two causes rather than only a rotation.

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
- `ENCRYPT_ALL` governs what is written. A row written while it was on stays encrypted, in the table and in the replay cache, whatever the valve says afterwards.

### Table naming and key rotation implications

The artifact table name includes a short hash derived from `(ARTIFACT_ENCRYPTION_KEY + pipe_identifier)`, which means:

- Rotating `ARTIFACT_ENCRYPTION_KEY` results in a different table name.
- Old artifacts remain in the database, but the pipe will read/write using the table corresponding to the currently configured key.

Operational impact:
- Key rotation can intentionally reduce historical artifact replay (older marker references will not resolve unless you restore the prior key).
- A rotation takes effect for writes already in flight. The cipher is rebuilt against the current key on every call, so the store is never left holding a retired one, but a write that was already inside that cipher build when the change landed is still written under the previous key and cannot be read afterwards; that one row is dropped and a warning in the pipe’s log names its artifact kind (never its content).
- A rotation also reaches rows still waiting in the Redis pending queue. A row that was buffered under the previous key and has not been written yet is discarded at the next flush rather than written into the new key's table, where nothing could ever read it; a warning names its artifact kind and count. The row is never written at all, so it will not be in the new table to look for.
- Plan rotations as an operational change and communicate the impact to users if you rely on long-lived artifact replay.

### Recommended configurations (security vs operational cost)

| Mode | `ARTIFACT_ENCRYPTION_KEY` | `ENCRYPT_ALL` | Intended use |
| --- | --- | --- | --- |
| No pipe-level artifact encryption | empty | n/a | Development and low-risk deployments where DB/storage is already strongly protected and artifact sensitivity is low. |
| Reasoning-only encryption | set | `False` | Reduce sensitivity exposure while limiting encryption overhead to reasoning payloads. |
| Full artifact encryption | set | `True` | Multi-tenant deployments and environments where persisted tool outputs/reasoning may contain sensitive data. |

---

## SSRF protection for remote downloads

Remote picture and video URLs are security-sensitive because they can be used for SSRF (Server-Side Request Forgery), and so is an attached file link: the pipe never fetches one, but the provider does, so the address it points at is checked before anything is forwarded. Every video spelling a caller can send — `video_url`, `input_video` and `video` alike — is security-sensitive in that same way, and is checked by the same gates before anything is forwarded.

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
- A video link the pipe never fetches is checked before it is passed on. Every non-`data:` `video_url` block — YouTube-shaped, schemeless, or a plain `http(s)` link — is put to the same address gate, and the provider is given the link only when the host resolves to a public address; a link whose scheme is neither `http` nor `https` is refused with `Video URL blocked by security policy (only http and https links are allowed)`. A refused link is not forwarded, and the person is told it was refused. Those checks draw on the same request-wide `ADDRESS_CHECK_BUDGET_SECONDS` as the picture ones, so a message carrying many video links is bounded by that budget rather than by one lookup's budget per link, and a link that gets no time left reaches no verdict and is not sent — and a link whose check reached no verdict at all is refused with its own wording, which says the check did not finish rather than calling the address private. A picture a turn attaches draws on that same budget for its download's own check and the further one a failed download costs, so a turn carrying many unfetchable pictures is bounded by it too.
- A file link the pipe never fetches is checked before it is passed on, the same way. Every `file_data` or `file_url` value the provider would have to fetch — in both fields, and in all three block spellings `input_file`, a flat `file` and a nested `{"type": "file", "file": {…}}` — is put to the same address gate, and the provider is given the value only when the host resolves to a public address; a link whose scheme is neither `http` nor `https` is refused in its own words, because nothing was resolved in that case. A refused link is not forwarded, and the person is told it was refused: `Files: skipped N (…)`, with the reason named. A block that also carries a `file_id` is not dropped — the id is still sent and only the refused field is removed, silently, because a report about a request that still carries the attachment is a duplicate. A value the provider would *not* fetch is not a link and is never checked: a `data:` URL in either field, raw base64 in `file_data`, and an Open WebUI file path, which is resolved from local storage instead. Unlike the video and picture checks, these are not drawn from `ADDRESS_CHECK_BUDGET_SECONDS`, so a turn with N attached links pays up to N x `ADDRESS_CHECK_SECONDS` serially, before the first byte goes upstream.
- An address in a generation request's provider options is put to the same gate, whichever key it sits under and however deeply it nests — including when the option's value is a JSON document rather than a bare link, whether that document was filled in from a filter control or posted with the request. The gate only reads: a value that passes the check is forwarded byte for byte as it was written. A control too large, or nested too deeply, for the check to certify is refused rather than sent, and the refusal names the path the address was found at.
- Address resolution runs on its own bounded thread pool, so a hostile or dead host — one whose nameserver never answers — cannot consume the threads serving media, thumbnails, file reads or Open WebUI's own endpoints. With `ENABLE_SSRF_PROTECTION=True` the pool is shared only with other address checks, so a stalled check reaches no verdict inside its budget rather than delaying anyone, and a queue behind a full pool reaches no verdict the same way; a download or a generation with no verdict still sends nothing, so that half is a refusal, and a picture the pipe already holds is served, because there the entry is being asked whether those bytes are still permitted and an answer that never arrived is not a denial. With `ENABLE_SSRF_PROTECTION=False` the vetted transport's own name resolution runs on a pool of the pipe's too, so the thread it occupies is bounded there; that path has no resolution budget, so a name that queues behind a full pool waits its turn and is charged to its reach's own hop budget rather than refused, and neither arm can interrupt a `getaddrinfo` already in flight — a refusal releases the caller, not the pool thread. It follows the pool its caller is using rather than choosing one: the model-icon fetch runs inside the background sweep and therefore draws on the narrower sweep pool, while the maker-profile fetch and the dashboard self-update draw on the wide one. Shutting the pipe down closes those pools, and a name resolution that races that teardown raises rather than answering — the same behaviour the gate-on path has always had. In practice that is an icon that does not appear, and a self-update that reports the update server as offline, rather than a leaked thread.

When `ENABLE_SSRF_PROTECTION=False`:
- The pipe may attempt to fetch internal URLs reachable from your Open WebUI environment. Only disable SSRF protection with a clear threat model and compensating controls.
- The file arm goes back to forwarding a link with nothing resolved: an `https://` attachment by URL to an intranet share or an internal docs server is passed on byte for byte again, which is the escape hatch for a deployment whose users attach documents that way. It is also the exposure the valve exists to close, so it restores the whole thing rather than only the file arm. Two refusals on this arm are the plaintext policy's and are *not* restored by it: a cleartext `http://` link still needs `ALLOW_INSECURE_HTTP` and an allowlisted host, and a link whose scheme is neither `http` nor `https` is still refused.
- A remote picture or file download resolves its host through the HTTP client rather than through the pipe's bounded address pool, so that one lookup still uses the event loop's default executor — the same pool media work and Open WebUI's own blocking endpoints share. This is recorded debt, not a design: the alternative is a `socket_factory` on the download client, which is a larger change to a path an operator has deliberately left unguarded.
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
  all, and with `ENABLE_SSRF_PROTECTION=True` they pin the validated address into the
  request. With it `False` there is no validated address to pin, so the URL goes out
  unpinned and the HTTP client resolves the name itself, outside the pipe's pools.
  The buffered and the streaming downloader run the same gate, so a verdict against one
  is a verdict against both.
- The three transport fetches do follow redirects, up to a small fixed number of hops.
  Every hop is validated the same way as the first, covering both the release metadata
  and the release asset in the self-update flow. A redirect to a private or internal
  address is refused before the connection is attempted. The transport is taken per
  HOP, not per chain, so a chain that is already in flight when the valve changes does
  not finish under the setting it started with: the origin decides how long to hold a
  hop open, which would otherwise be how long the old setting lasted.
- Turning `ENABLE_SSRF_PROTECTION` back on takes effect on the next hop, which is the
  next request or the rest of a redirect chain already under way. The transport is
  REBUILT when the valve changes: the retired session is never handed to a new hop and
  is closed as soon as its last in-flight hop releases it, so a settings change never
  truncates a download already under way, and a new connection pool, a new address cache
  and a new resolver are created together. Clearing the address cache alone was not
  enough, because a keep-alive connection opened while the gate was off is reused
  without consulting any resolver, and an answer from a lookup that was already in
  flight lands in the cache after it has been cleared.
- The gate also re-runs on a **re-use**: an image a worker already downloaded is
  answered from its in-process memo on later turns, and the memo does not stand in for
  the gate. The memo is per-person: an entry is served only to the person whose own
  turn downloaded it, so a second person on the same `(chat_id, url)` re-downloads
  rather than being inlined the first person's bytes, and a request the pipe could
  not resolve to a user is re-downloaded rather than being served a named person's
  bytes. Two such unresolved requests share one partition, so a caller the pipe
  could not resolve is re-downloaded only when a named owner wrote the entry; on
  the unlogged empty-`user_id` path two unresolved callers can still share one entry
  (TODO T569 in `tests/test_one_persons_reused_picture_is_never_sent_to_another.py`).
  Every turn re-checks the URL before those bytes are sent, and a URL the
  current valve values refuse is dropped from the memo and fetched through the gated
  path instead; the same is true of the remote-download size cap, so a picture memoised
  while that cap was higher is released rather than reused once it is lowered. So every byte that reaches the provider passed the gate under the valve
  values in force at the moment it was sent, and tightening the valve or editing a host
  allowlist takes effect on the next turn rather than the next fetch. The re-check
  costs one address resolution and no download; a URL the check then refuses is fetched
  once more through the already-gated path, which runs its own address check, so a
  refusal pays two resolutions. A resolution that does not finish inside the
  address-check budget reaches no verdict, so a slow nameserver never buys a
  permission, and on a re-use it costs neither the entry nor a re-fetch: the
  picture the pipe already holds is served, and the re-check runs again on the
  next turn. Nothing has to be refused for the re-check to be paid for:
  a request pays one blocking resolution per **distinct** URL it reuses this turn, each
  one inside the same request-wide address-check budget, so a picture cited twice in one
  reply is checked once and a turn carrying many pictures is bounded by that
  `ADDRESS_CHECK_BUDGET_SECONDS` (20.0) rather than by a lookup per picture.
  A turn's **video** links draw on that same request-wide budget, and the picture and
  video checks share it, so a message carrying many of either is bounded by the budget
  rather than by a lookup per link.

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

`_redact_payload_blobs()` runs over every request payload the pipe records, and over the body of every error response it records, at any log level. It reduces two kinds of value:

- **Large base64 blobs** — a `data:` URL's payload is reduced to `data:<media type> [redacted]` at any length, and a bare blob is truncated to a marker with its length under two different thresholds. A key the pipe itself sends a person's bytes under (`b64_json`, `b64`, `base64`, `image_base64`, `imageB64`, `data`, `input_audio`, `audio`) is truncated above `max_chars` (256 by default), so a short attachment cannot be logged whole. A bare blob under any other key is truncated only above 1024 characters, because `[A-Za-z0-9+/_-]{1024,}` also matches a digest or a very long ordinary string and truncating those would take the operator's own text with them. Below that floor, and below `max_chars` under a named key, the value reaches the log as it was — up to 256 characters (192 bytes) under a named key, and up to 1023 under any other. That residual is deliberate and is the price of the floor: it is a DEBUG-only record, `LOG_LEVEL` is the switch, and the first 64 characters of a truncated blob are kept on purpose so a value can still be recognised.
- **Media links** — any value under a media-URL key is reduced to scheme, host and port by `loggable_link()`, with any `user:password@` userinfo in the authority dropped as well. A key is matched **without regard to case or underscores**, so `image_url`, `imageUrl`, `imageURL` and `imageurl` are one key, not four, and so are `file_url`/`fileUrl`, `video_url`/`videoUrl`, `file_data`/`fileData` and `content_url`/`contentUrl`. A key's spelling is not a security boundary: the stems are derived from the one key set, so a future integration's sixth spelling is covered without a further edit. The path and query are dropped, so a presigned or signed CDN link cannot be recovered from a log, and the reduced form still names `scheme://host[:port]` (or `[REDACTED]` for a value that cannot be described at all) — the host is the operator's allowlist key. This holds for a link whose netloc does not parse and for a `data:` URL with no comma as well as for a well-formed `https://` one. A key outside that set is untouched, however URL-shaped its value: the rule is keyed on the key, never on the value, so the model's own prose survives DEBUG.

The ERROR path is covered too. A refusal's log subject is the link the pipe declined, so subjects are built with `loggable_link()` rather than the raw URL. This means turning DEBUG on shows you every media field as `scheme://host:port` rather than the full URL — the pipe is telling you which host it was about to contact, not the signed path it was given. It holds for both remote downloaders, the buffered and the streaming one, at every log site that names the link; a third downloader inherits the same rule. A refusal that is about the *file* rather than about the link names the kind of file — a field name, a media type, a modality — and never the person's file name; a refusal with no link to name logs `no source`. The name still travels to the model, where the Responses API's `filename` is documented as being "for model context", and to nowhere else.

So is the error-response body, and the WARNING a rejection raises. A provider's moderation echo quotes the person's own attachment back at them, so a body the pipe read back from a 4xx or a 5xx is parsed, redacted, and then rewritten string by string: every `data:` URL in it becomes `data:<media type> [redacted]` with no prefix of the payload left behind. The same rewrite is applied where a rejection is reported — the provider's message in the WARNING, the streaming producer's report of the same rejection, and the traceback that record carries. `str(exc)`, `raw_body`, and the error card's `{raw_body}` and `{flagged_excerpt}` are left byte-identical on purpose: those are what the person is shown, and the card is where the diagnosis lives. What is cut is the card's *derived* copies of that body -- the two JSON values and the inline copies of the provider's message -- at a fixed bound with a visible marker, while the payload itself is not, so the cut bounds what a provider can make the pipe repeat without hiding anything the person was sent. That is true for a saved chat, which has one reader. On a channel every member of the room reads the card, so the pipe renders it as though those values were empty and keeps the diagnosis that names no one: `error_id`, the model, the provider, `openrouter_code` and `status_code`. The card a channel reader sees is a strict subset of the card the requester sees. The scrub is applied where the record is written; the caller's JSON body is **composed**, not transported, and its `error.message` is filled only from values OpenRouter wrote at the top level of its own `error` object (or from the HTTP status line), so nothing sourced from `error.metadata.raw` reaches it at any nesting depth. The second case the composition holds against is a body that never parsed into an `error` object at all: on that fault the envelope names the endpoint and the upstream `Content-Type` — both derived by the pipe — and quotes no part of the body, on the ground that nothing in it was written by OpenRouter and a proxy in front of it wrote whatever it liked.

**What is deliberately left unscrubbed: the chat card and the operator's WARNING.** Both keep the provider's verbatim text, by decision, because the card is the diagnosis surface and a transcript outlives the incident. Bounding the derived copies is a length decision made alongside that one, not a scrubbing one: nothing is removed from the payload a reader is shown, and the copies around it are cut at 16,384 characters with a marker naming how many characters were removed. A provider that echoes a request's `Authorization` header into `error.metadata.raw` would therefore put that text in the card, the session archive, and any later turn's replay to the model. The pipe does not add a credential scrubber for it: a denylist against an open-ended set of provider phrasings is defeated by any provider that echoes in a different shape, and the field it would protect is the one the operator reads when nothing else has diagnosed the failure. The JSON envelope a client logs is closed against this; the card is not, and that is the trade.

This is deliberately stronger than Open WebUI, which caps an error body at `error_body[:1000]` — a length cap, not a scrub, and it has no redaction helper at all; the host also logs the whole `form_data` at DEBUG with no redaction of its own, so an operator running the host itself at DEBUG sees the payload whatever this pipe does. That last one is an action on the host's `LOG_LEVEL`, not something a pipe change can reach. The pipe's log and Open WebUI's will therefore disagree about the same rejected request: the pipe shows `data:image/png [redacted]` where Open WebUI keeps a thousand characters of the same body. That divergence is intended.

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
