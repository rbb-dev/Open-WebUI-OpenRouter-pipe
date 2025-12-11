# valves & configuration atlas

**file:** `docs/valves_and_configuration_atlas.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:2421-2870` (Valve definitions) plus accompanying docs referenced below.

Valves are the sole configuration surface for the OpenRouter Responses pipe. This reference mirrors the order of `Pipe.Valves` and `Pipe.UserValves`, groups related toggles, and explains how each value is used at runtime. Defaults reflect the current code; update this file whenever you add or change a valve.

---

## 1. system valves (`Pipe.Valves`)

### 1.1 connection & authentication

| Valve | Default | Range | Notes |
| --- | --- | --- | --- |
| `BASE_URL` | env `OPENROUTER_API_BASE_URL` or `https://openrouter.ai/api/v1` | n/a | Target endpoint for every OpenRouter call. Override when routing through a proxy/gateway. |
| `API_KEY` | env `OPENROUTER_API_KEY` | n/a | Stored as `EncryptedStr`. Required for catalog fetches and responses. Leave blank for catalog-only mode. |
| `HTTP_CONNECT_TIMEOUT_SECONDS` | 10 | 1--∞ | TCP/TLS connect timeout for `httpx`. Increase on high-latency networks. |
| `HTTP_TOTAL_TIMEOUT_SECONDS` | `None` | ≥1 or null | Caps the entire request (including streaming). Keep `None` for long outputs. |
| `HTTP_SOCK_READ_SECONDS` | 300 | ≥1 | Idle read timeout when `HTTP_TOTAL_TIMEOUT_SECONDS` is unset. Prevents hung streams. |

### 1.2 remote intake & payload guards

| Valve | Default | Range | Notes |
| --- | --- | --- | --- |
| `REMOTE_DOWNLOAD_MAX_RETRIES` | 3 | 0--10 | Retries for `_download_remote_url`. Set 0 to disable. |
| `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS` | 5 | 1--60 | Starting backoff delay (doubling each attempt). |
| `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS` | 45 | 5--300 | Hard ceiling on time spent retrying a download. |
| `REMOTE_FILE_MAX_SIZE_MB` | 50 | 1--500 | Applies to remote downloads + inline image/file payloads. Auto-clamped to Open WebUI"s `FILE_MAX_SIZE` when defined. |
| `SAVE_REMOTE_FILE_URLS` | False | bool | Download + re-host `file_url` links. Leave false unless you trust upstream hosts. |
| `SAVE_FILE_DATA_CONTENT` | True | bool | Forces `file_data` blobs into storage so transcripts stay lean. |
| `BASE64_MAX_SIZE_MB` | 50 | 1--500 | Validates base64 data before decoding. |
| `IMAGE_UPLOAD_CHUNK_BYTES` | 1 MB | 64 KB--8 MB | Max buffer when converting `/api/v1/files/...` images into `data:` URLs. |
| `VIDEO_MAX_SIZE_MB` | 100 | 1--1000 | Guardrail for video data URLs. |
| `FALLBACK_STORAGE_EMAIL` | `openrouter-pipe@system.local` | n/a | Owner of uploads when no chat user exists. |
| `FALLBACK_STORAGE_NAME` | `OpenRouter Pipe Storage` | n/a | Display name for the fallback account. |
| `FALLBACK_STORAGE_ROLE` | `pending` | n/a | Role assigned to the fallback account. Warns if set to a privileged value. |
| `ENABLE_SSRF_PROTECTION` | True | bool | Blocks downloads to private/link-local ranges. Disable only when intentionally fetching internal hosts. |

### 1.3 model catalog & reasoning knobs

| Valve | Default | Range | Notes |
| --- | --- | --- | --- |
| `MODEL_ID` | `auto` | CSV | Restrict exposed models. `auto` imports the entire catalog. **Task payloads (`__task__`) bypass this allowlist** so Open WebUI housekeeping (titles, tags, etc.) keeps working even when end-user models are locked down. Interactive chats and API calls still enforce the list. |
| `MODEL_CATALOG_REFRESH_SECONDS` | 3600 | ≥60 | Cache TTL for the catalog. |
| `ENABLE_REASONING` | True | bool | Requests reasoning traces whenever supported. |
| `REASONING_EFFORT` | `medium` | minimal/low/medium/high | Default `reasoning.effort` value. |
| `TASK_MODEL_REASONING_EFFORT` | `low` | minimal/low/medium/high | Effort applied when Open WebUI schedules background tasks (titles, tags, etc.) against this pipe. The default `low` adds lightweight reasoning while leaving streaming control to Open WebUI; drop to `minimal` to keep the fastest, no-web-search behavior, or raise to medium/high for more deliberate task-specific outputs. |
| `REASONING_SUMMARY_MODE` | `auto` | auto/concise/detailed/disabled | Governs reasoning summary verbosity. |
| `AUTO_CONTEXT_TRIMMING` | True | bool | When enabled, automatically attaches OpenRouter’s `middle-out` transform so long prompts degrade gracefully instead of hard-failing with 400-over-limit errors. Disable only if you manage `transforms` manually per request. |
| `PERSIST_REASONING_TOKENS` | `next_reply` | disabled/next_reply/conversation | Retention horizon; see `docs/history_reconstruction_and_context.md`. |
| `MAX_INPUT_IMAGES_PER_REQUEST` | 5 | 1--20 | Cap on `input_image` blocks. |
| `IMAGE_INPUT_SELECTION` | `user_then_assistant` | `user_turn_only`/`user_then_assistant` | Selection policy when user omits new images. |

### 1.4 tool execution & artifacts

| Valve | Default | Range | Notes |
| --- | --- | --- | --- |
| `PERSIST_TOOL_RESULTS` | True | bool | Store tool outputs for replay. Disable for ephemeral runs. |
| `ARTIFACT_ENCRYPTION_KEY` | `""` | string | Enables Fernet encryption. ≥16 chars recommended. |
| `ENCRYPT_ALL` | False | bool | Encrypt every artifact, not just reasoning. |
| `ENABLE_LZ4_COMPRESSION` | True | bool | Requires `lz4` package. Compresses large payloads. |
| `MIN_COMPRESS_BYTES` | 0 | ≥0 | Skip compression for tiny blobs. |
| `ENABLE_STRICT_TOOL_CALLING` | True | bool | Enforce strict JSON Schema (see `docs/tooling_and_integrations.md`). |
| `MAX_FUNCTION_CALL_LOOPS` | 10 | ≥1 | Safety valve for multi-step tool loops. |
| `ENABLE_WEB_SEARCH_TOOL` | True | bool | Auto-attach OpenRouter"s `web` plugin when supported. Requests whose effective `reasoning.effort` resolves to `minimal` skip the plugin entirely (OpenRouter rejects `web_search` when effort is `minimal`), so raise the effort or disable this valve if you need tool-free runs. |
| `WEB_SEARCH_MAX_RESULTS` | 3 | 1--10 or null | Overrides the plugin"s `max_results`. |
| `REMOTE_MCP_SERVERS_JSON` | None | JSON | See `docs/tooling_and_integrations.md` for schema. |

### 1.5 logging, streaming, concurrency

| Valve | Default | Range | Notes |
| --- | --- | --- | --- |
| `LOG_LEVEL` | env `GLOBAL_LOG_LEVEL` or `INFO` | enum | Pipe-wide log threshold. |
| `MAX_CONCURRENT_REQUESTS` | 200 | 1--2000 | Size of the global request semaphore. |
| `SSE_WORKERS_PER_REQUEST` | 4 | 1--8 | SSE parser tasks per request. |
| `STREAMING_CHUNK_QUEUE_MAXSIZE` | 100 | 10--5000 | Raw chunk buffer limit. |
| `STREAMING_EVENT_QUEUE_MAXSIZE` | 100 | 10--5000 | Parsed event buffer limit. |
| `MAX_PARALLEL_TOOLS_GLOBAL` | 200 | 1--2000 | Global tool semaphore. |
| `MAX_PARALLEL_TOOLS_PER_REQUEST` | 5 | 1--50 | Per-request worker cap. |
| `TOOL_BATCH_CAP` | 4 | 1--32 | Max compatible tool calls per batch. |
| `TOOL_OUTPUT_RETENTION_TURNS` | 10 | ≥0 | See `docs/history_reconstruction_and_context.md`. |
| `TOOL_TIMEOUT_SECONDS` | 60 | 1--600 | Per-tool timeout. |
| `TOOL_BATCH_TIMEOUT_SECONDS` | 120 | ≥1 | Timeout for an entire batch. |
| `TOOL_IDLE_TIMEOUT_SECONDS` | None | ≥1 or null | Cancels idle tool queues if set. |

### 1.6 redis, persistence, maintenance

| Valve | Default | Range | Notes |
| --- | --- | --- | --- |
| `ENABLE_REDIS_CACHE` | True | bool | Activates Redis write-behind when the environment supports it. |
| `REDIS_CACHE_TTL_SECONDS` | 600 | 60--3600 | TTL for cached artifacts. |
| `REDIS_PENDING_WARN_THRESHOLD` | 100 | 1--10000 | Log warning when pending queue exceeds this size. |
| `REDIS_FLUSH_FAILURE_LIMIT` | 5 | 1--50 | Disable Redis after this many consecutive flush failures. |
| `ARTIFACT_CLEANUP_DAYS` | 90 | 1--365 | Age threshold for DB cleanup job. |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` | 1.0 | 0.5--24 | Cleanup cadence. |
| `DB_BATCH_SIZE` | 10 | 5--20 | Rows per DB transaction when draining Redis. |

### 1.7 reporting & UI tweaks

| Valve | Default | Notes |
| --- | --- | --- |
| `USE_MODEL_MAX_OUTPUT_TOKENS` | False | Forwards provider-advertised `max_output_tokens` automatically. |
| `SHOW_FINAL_USAGE_STATUS` | True | Includes timing/cost/tokens in the final status bubble. |
| `ENABLE_STATUS_CSS_PATCH` | True | Injects a CSS helper so multi-line statuses render nicely. |
| `OPENROUTER_ERROR_TEMPLATE` | (see source) | Markdown template for OpenRouter 400 responses. Supports all placeholders listed in `docs/openrouter_integrations_and_telemetry.md#template-variables` plus Handlebars-style `{{#if variable}} ... {{/if}}` blocks. Lines drop automatically when a placeholder is empty, so admins can restructure the template without touching Python. |

---

## 2. user valves (`Pipe.UserValves`)

Per-user overrides mirror global valves where it makes sense. The literal string `"inherit"` (any case) means "fall back to system default".

| Valve | Default | Range | Notes |
| --- | --- | --- | --- |
| `LOG_LEVEL` | `INHERIT` | enum | Lets a single user opt into DEBUG logging without affecting others. |
| `SHOW_FINAL_USAGE_STATUS` | True | bool | Per-user toggle for final usage banner. |
| `ENABLE_REASONING` | True | bool | Disable reasoning for a specific user. |
| `REASONING_EFFORT` | `medium` | enum | Personal preference for reasoning depth. |
| `REASONING_SUMMARY_MODE` | `auto` | enum | Personal verbosity setting for reasoning summaries. |
| `PERSIST_REASONING_TOKENS` | `next_reply` | enum | Same semantics as system valve but scoped to the user. |
| `PERSIST_TOOL_RESULTS` | True | bool | Opt out of tool persistence for a single user. |

---

## 3. valve hygiene

1. **Document every change** -- update this file and `docs/documentation_index.md` when you add or rename a valve. Reviewers block PRs that ship undocumented knobs.
2. **Validate ranges** -- let Pydantic enforce types and ranges; avoid manual `min/max` clamping elsewhere.
3. **Secret handling** -- use `EncryptedStr` for anything sensitive. Without `WEBUI_SECRET_KEY`, encrypted valves silently degrade to plaintext; log messages warn operators when that happens.
4. **Testing** -- extend `tests/test_pipe_guards.py` when you add guards or new interactions (e.g., a valve that toggles Redis behavior).

With this atlas you can tune the manifold confidently, knowing exactly how each knob affects the system.
