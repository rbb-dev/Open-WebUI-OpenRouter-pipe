# Valve Reference

This document enumerates every valve exported by the OpenRouter Responses pipe.  
Valves are grouped by the `Pipe.Valves` (system-wide) and `Pipe.UserValves` (per-user overrides) structures defined in `openrouter_responses_pipe/openrouter_responses_pipe.py`.  
Each entry lists the default value, allowed range, and a short explanation of *how* the pipe uses the valve and *why* you might tune it.

---

## System Valves (`Pipe.Valves`)

### Connection & Authentication
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `BASE_URL` (str) | `OPENROUTER_API_BASE_URL` env or `https://openrouter.ai/api/v1` | n/a | Core endpoint for every OpenRouter request. Override only when routing through a proxy or gateway. |
| `API_KEY` (EncryptedStr) | `OPENROUTER_API_KEY` env or empty | n/a | API credential injected into all outgoing requests. Stored encrypted; leave blank to run in catalog-only mode. |
| `HTTP_CONNECT_TIMEOUT_SECONDS` (int) | 10 | 1 / n/a | TCP/TLS connect timeout passed to `httpx`. Raise if you routinely access OpenRouter through high-latency proxies. |
| `HTTP_TOTAL_TIMEOUT_SECONDS` (Optional[int]) | `None` | 1 / n/a | When set, caps total request time (including streaming). Keep `None` for long-running streams to avoid cutting off model output. |
| `HTTP_SOCK_READ_SECONDS` (int) | 300 | 1 / n/a | Idle read timeout used when `HTTP_TOTAL_TIMEOUT_SECONDS` is disabled to keep slow providers from stalling indefinitely. |

### Remote Intake & Payload Guards
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `REMOTE_DOWNLOAD_MAX_RETRIES` (int) | 3 | 0 / 10 | Number of retry attempts when downloading remote files/images. Set to 0 to disable retries. |
| `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS` (int) | 5 | 1 / 60 | Initial backoff delay; subsequent retries double the delay to stay polite with remote servers. |
| `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS` (int) | 45 | 5 / 300 | Hard ceiling on total retry time to keep queues moving during repeated failures. |
| `REMOTE_FILE_MAX_SIZE_MB` (int) | 50 | 1 / 500 | Cap for remote downloads *and* remote image/file blocks. When Open WebUI RAG uploads enforce `FILE_MAX_SIZE`, the effective cap auto-aligns to avoid ingesting larger files than RAG can store. |
| `SAVE_REMOTE_FILE_URLS` (bool) | `False` | n/a | When enabled, remote `file_url` inputs (including data URLs) are downloaded and re-hosted inside Open WebUI storage; when disabled the URLs are passed through untouched. Use this to control storage growth and trust boundaries. |
| `BASE64_MAX_SIZE_MB` (int) | 50 | 1 / 500 | Validates inline base64 payloads before decoding to prevent memory spikes from oversized uploads. |
| `VIDEO_MAX_SIZE_MB` (int) | 100 | 1 / 1000 | Maximum size for video payloads (remote URLs or data URLs). Protects against huge video inputs that exceed typical worker resources. |
| `FALLBACK_STORAGE_EMAIL` (str) | `OPENROUTER_STORAGE_USER_EMAIL` env or `openrouter-pipe@system.local` | n/a | Email assigned to the service user that owns uploads when no chat user exists (e.g., automation callers). |
| `FALLBACK_STORAGE_NAME` (str) | `OPENROUTER_STORAGE_USER_NAME` env or `"OpenRouter Pipe Storage"` | n/a | Display name for the fallback storage owner shown in OWUI’s file listings. |
| `FALLBACK_STORAGE_ROLE` (str) | `OPENROUTER_STORAGE_USER_ROLE` env or `"pending"` | n/a | Role applied to the fallback storage user if the pipe needs to auto-create it. Defaults to the low-privilege `"pending"` role; bump only when your Open WebUI permissions require a different service account scope. |
| `ENABLE_SSRF_PROTECTION` (bool) | `True` | n/a | Blocks remote downloads that resolve to private networks (localhost, RFC1918 ranges) to guard against SSRF attacks. Disable only if you deliberately fetch from an internal host. |

> **Security tip:** If you override `FALLBACK_STORAGE_ROLE` to a privileged value (`admin`, `system`, etc.), the pipe now emits a warning at runtime. Keep the fallback account least-privileged unless you explicitly need elevated storage permissions.

### Model Catalog & Reasoning Preferences
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `MODEL_ID` (str) | `"auto"` | n/a | Comma-separated list of OpenRouter model IDs to expose. `"auto"` imports the full Responses-capable catalog. |
| `MODEL_CATALOG_REFRESH_SECONDS` (int) | 3600 | 60 / n/a | Cache TTL for the model catalog fetcher. Lower to refresh capabilities more frequently at the cost of extra API calls. |
| `ENABLE_REASONING` (bool) | `True` | n/a | Requests reasoning traces whenever the selected model supports them. Turn off to reduce token use. |
| `REASONING_EFFORT` (Literal) | `"medium"` | `"minimal"`…`"high"` | Default reasoning effort OpenRouter should spend (higher effort ⇒ more reasoning tokens). |
| `REASONING_SUMMARY_MODE` (Literal) | `"auto"` | `"auto" / "concise" / "detailed" / "disabled"` | Controls how the pipe asks for reasoning summaries when available. |
| `PERSIST_REASONING_TOKENS` (Literal) | `"next_reply"` | `"disabled" / "next_reply" / "conversation"` | Governs how long reasoning artifacts stay in the conversation store. |

### Tool Execution & Artifact Storage
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `PERSIST_TOOL_RESULTS` (bool) | `True` | n/a | Keeps tool outputs in the scoped database so later turns can reference them. Disable for ephemeral tool responses. |
| `ARTIFACT_ENCRYPTION_KEY` (EncryptedStr) | `""` | ≥16 chars recommended | Derives the Fernet key protecting persisted reasoning/tool artifacts. Changing the key creates a new table namespace. |
| `ENCRYPT_ALL` (bool) | `False` | n/a | When `True`, encrypts *all* artifacts; otherwise only reasoning tokens are encrypted. |
| `ENABLE_LZ4_COMPRESSION` (bool) | `True` | n/a | Compresses large encrypted blobs (when `lz4` is installed) to reduce I/O. |
| `MIN_COMPRESS_BYTES` (int) | 0 | 0 / n/a | Size threshold (bytes) before LZ4 compression is attempted. Default 0 always tries compression; raise the value if you want to skip tiny payloads. |
| `ENABLE_STRICT_TOOL_CALLING` (bool) | `True` | n/a | Converts Open WebUI tool schemas to strict JSON Schema before sending to OpenRouter, improving tool-call determinism. |
| `MAX_FUNCTION_CALL_LOOPS` (int) | 10 | 1 / n/a | Safety net preventing runaway function-call loops when a model keeps requesting tools. |
| `ENABLE_WEB_SEARCH_TOOL` (bool) | `True` | n/a | Auto-attaches OpenRouter’s `web` plugin for models that advertise the capability. |
| `WEB_SEARCH_MAX_RESULTS` (Optional[int]) | 3 | 1 / 10 (or `None`) | Overrides how many search results the plugin should request. Set `None` to defer to provider defaults. |
| `REMOTE_MCP_SERVERS_JSON` (Optional[str]) | `None` | n/a | JSON definition of remote MCP servers to attach to every request. Useful for centrally enabling shared tools. |

### Logging, Streaming, and Concurrency
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `LOG_LEVEL` (Literal) | `GLOBAL_LOG_LEVEL` env or `"INFO"` | n/a | Minimum log level used by the pipe-level logger. |
| `MAX_CONCURRENT_REQUESTS` (int) | 200 | 1 / 2000 | Size of the global semaphore controlling in-flight OpenRouter calls. Lower to constrain resource usage. |
| `SSE_WORKERS_PER_REQUEST` (int) | 4 | 1 / 8 | Number of background tasks parsing streamed responses per request; adjust for CPU vs. latency trade-offs. |
| `STREAMING_CHUNK_QUEUE_MAXSIZE` (int) | 100 | 10 / 5000 | Backpressure limit for raw SSE chunks before slowing down the upstream stream. |
| `STREAMING_EVENT_QUEUE_MAXSIZE` (int) | 100 | 10 / 5000 | Buffer size for parsed SSE events awaiting downstream consumption. |
| `STREAMING_UPDATE_PROFILE` (Optional[Literal]) | `None` (`quick`/`normal`/`slow`) | n/a | Shortcut for simultaneously adjusting streaming char limits + flush timers. |
| `STREAMING_UPDATE_CHAR_LIMIT` (int) | 20 | 10 / 500 | Maximum characters batched per incremental update. Lower means chattier streams with lower latency. |
| `STREAMING_IDLE_FLUSH_MS` (int) | 250 | 0 / 2000 | Watchdog that flushes partial outputs when the model pauses. Set to 0 to disable idle flush behavior. |

### Tool Queue & Execution Limits
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `MAX_PARALLEL_TOOLS_GLOBAL` (int) | 200 | 1 / 2000 | Caps total concurrent tool executions across all requests. Prevents a burst of tool calls from exhausting resources. |
| `MAX_PARALLEL_TOOLS_PER_REQUEST` (int) | 5 | 1 / 50 | Per-request ceiling for concurrent tool workers to keep single chats from monopolizing the tool pool. |
| `TOOL_BATCH_CAP` (int) | 4 | 1 / 32 | Maximum compatible tool calls executed together when batching is allowed. |
| `TOOL_OUTPUT_RETENTION_TURNS` (int) | 10 | 0 / n/a | Number of recent user/assistant turns whose tool outputs are replayed verbatim; older ones are pruned to save tokens. |
| `TOOL_TIMEOUT_SECONDS` (int) | 60 | 1 / 600 | Individual tool call timeout. Increase only if tools regularly need more than a minute. |
| `TOOL_BATCH_TIMEOUT_SECONDS` (int) | 120 | 1 / n/a | Timeout for an entire batch of tool calls; generous by default to accommodate multi-tool workflows. |
| `TOOL_IDLE_TIMEOUT_SECONDS` (Optional[int]) | `None` | 1 / n/a | Optional idle timeout between queued tool executions. Leave `None` for unlimited idle time (best for sporadic tools). |

### Redis, Persistence, and Maintenance
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `ENABLE_REDIS_CACHE` (bool) | `True` | n/a | Enables the Redis write-behind cache once Open WebUI is running multiple workers with `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, and `WEBSOCKET_REDIS_URL`. |
| `REDIS_CACHE_TTL_SECONDS` (int) | 600 | 60 / 3600 | TTL for cached artifacts stored in Redis to serve multi-worker lookups quickly. |
| `REDIS_PENDING_WARN_THRESHOLD` (int) | 100 | 1 / 10000 | Emits warnings if pending Redis flush queue exceeds this size. |
| `REDIS_FLUSH_FAILURE_LIMIT` (int) | 5 | 1 / 50 | Disables Redis caching after N consecutive flush failures, falling back to direct DB writes. |
| `ARTIFACT_CLEANUP_DAYS` (int) | 90 | 1 / 365 | Retention window for the scheduled artifact cleanup job. |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` (float) | 1.0 | 0.5 / 24 | How frequently the cleanup worker wakes up to purge stale artifacts. |
| `DB_BATCH_SIZE` (int) | 10 | 5 / 20 | Number of artifacts grouped per DB commit, balancing throughput vs. transaction size. |

### Reporting & UI Behavior
| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `USE_MODEL_MAX_OUTPUT_TOKENS` (bool) | `False` | n/a | When `True`, forwards the provider’s `max_output_tokens` limit with every request. Helpful when providers reject unknown max-tokens values. |
| `SHOW_FINAL_USAGE_STATUS` (bool) | `True` | n/a | Appends elapsed time, cost, and token counts to the final status message emitted to Open WebUI. |
| `ENABLE_STATUS_CSS_PATCH` (bool) | `True` | n/a | Injects a CSS patch via `__event_call__` so Open WebUI can display multi-line status descriptions cleanly. |

---

## User Valves (`Pipe.UserValves`)

Per-user valves allow overrides on top of the global defaults. Any field set to the literal string `"inherit"` (case-insensitive) is treated as unset and falls back to the system valve value.

| Valve | Default | Min / Max | Usage & Rationale |
| --- | --- | --- | --- |
| `LOG_LEVEL` (Literal + `"INHERIT"`) | `"INHERIT"` | n/a | Lets a user opt into more (or fewer) logs without changing the workspace default. |
| `SHOW_FINAL_USAGE_STATUS` (bool) | `True` | n/a | Toggle per-user display of the final cost/usage banner. |
| `ENABLE_REASONING` (bool) | `True` | n/a | Users can disable reasoning traces for their own chats even when the workspace keeps it enabled. |
| `REASONING_EFFORT` (Literal) | `"medium"` | `"minimal"`…`"high"` | Personal preference for how “hard” the model should think when reasoning-capable. |
| `REASONING_SUMMARY_MODE` (Literal) | `"auto"` | `"auto" / "concise" / "detailed" / "disabled"` | Lets an individual pick how verbose the reasoning summaries should be. |
| `PERSIST_REASONING_TOKENS` (Literal, alias `"next_reply"`) | `"next_reply"` | `"disabled" / "next_reply" / "conversation"` | Per-user reasoning retention horizon. Alias retained for compatibility with the UI schema. |
| `PERSIST_TOOL_RESULTS` (bool) | `True` | n/a | Opt out of persisting tool outputs for a specific user’s chats. |
| `STREAMING_UPDATE_PROFILE` (Literal) | `"normal"` | `"quick" / "normal" / "slow"` | Personal streaming preset to trade latency vs. stability. |
| `STREAMING_UPDATE_CHAR_LIMIT` (int) | 20 | 10 / 500 | Per-user throttle on streaming batch size. |
| `STREAMING_IDLE_FLUSH_MS` (int) | 250 | 0 / 2000 | Per-user idle flush watchdog. Set to 0 to disable idle flush for that user only. |

---

Need to change a valve at runtime? Update the pipe settings inside Open WebUI (for system valves) or via per-user preferences (for user valves), then restart the pipe if noted in the UI. Use this reference to ensure the values you pick stay within supported ranges and align with how the manifold enforces guardrails.***
