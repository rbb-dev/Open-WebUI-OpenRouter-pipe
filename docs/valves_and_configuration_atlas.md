# Valves & Configuration Atlas

This document is the authoritative reference for the pipe’s configuration surface: **Open WebUI valves**.

Defaults and valve names are verified against `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py` and are intended to match current runtime behavior.

> **Quick navigation:** [Docs Home](README.md) · [Security](security_and_encryption.md) · [Multimodal](multimodal_ingestion_pipeline.md) · [Telemetry](openrouter_integrations_and_telemetry.md) · [Errors](error_handling_and_user_experience.md)

---

## How valves work

- **System valves** (`Pipe.Valves`) apply globally to the function (all users).
- **User valves** (`Pipe.UserValves`) allow per-user overrides for a limited subset of settings.
- When both are present, the pipe merges user valves into system valves; unset values are ignored.
  - When user valves are provided as a dict, the literal string `inherit` (case-insensitive) is treated as “unset”.
  - The pipe does **not** allow per-user overrides of the global `LOG_LEVEL`.

**Secret handling**
- Some valves use `EncryptedStr` to mark secret values (for example API keys and zip passwords). Open WebUI’s handling of encrypted values depends on Open WebUI’s own secret configuration (for example `WEBUI_SECRET_KEY`). Treat `EncryptedStr` as *sensitive* and protect backups accordingly.

---

## System valves (`Pipe.Valves`)

### Connection and authentication

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `BASE_URL` | `str` | `env OPENROUTER_API_BASE_URL, else https://openrouter.ai/api/v1` | OpenRouter API base URL. Override this if you are using a gateway or proxy. |
| `API_KEY` | `EncryptedStr` | `env OPENROUTER_API_KEY (empty if unset)` | Your OpenRouter API key. Defaults to the `OPENROUTER_API_KEY` environment variable. |
| `HTTP_REFERER_OVERRIDE` | `str` | `""` | Override `HTTP-Referer` for OpenRouter app attribution. Must be a full URL including scheme (e.g. `https://example.com`), not just a hostname. When empty, the pipe uses its default project URL. |
| `HTTP_CONNECT_TIMEOUT_SECONDS` | `int` | `10` | Seconds to wait for the TCP/TLS connection to OpenRouter before failing. |
| `HTTP_TOTAL_TIMEOUT_SECONDS` | `Optional[int]` | `null` | Overall HTTP timeout (seconds) for OpenRouter requests. Set to null to disable the total timeout so long-running streaming responses are not interrupted. |
| `HTTP_SOCK_READ_SECONDS` | `int` | `300` | Idle read timeout (seconds) applied to active streams when `HTTP_TOTAL_TIMEOUT_SECONDS` is disabled. |

### Remote downloads, multimodal intake, and SSRF

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `REMOTE_DOWNLOAD_MAX_RETRIES` | `int` | `3` | Maximum number of retry attempts for downloading remote images and files. Set to 0 to disable retries. |
| `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS` | `int` | `5` | Initial delay in seconds before the first retry attempt. Subsequent retries use exponential backoff (delay * 2^attempt). |
| `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS` | `int` | `45` | Maximum total time in seconds to spend on retry attempts. Retries stop if this time limit is exceeded. |
| `REMOTE_FILE_MAX_SIZE_MB` | `int` | `50` | Maximum size in MB for downloading remote files/images. When Open WebUI RAG is enabled, the pipe automatically caps downloads to Open WebUI’s `FILE_MAX_SIZE` (if set). |
| `SAVE_REMOTE_FILE_URLS` | `bool` | `True` | When True, remote URLs and data URLs in the `file_url` field are downloaded/parsed and re-hosted in Open WebUI storage. When False, `file_url` values pass through untouched. Recommended in code: disable to avoid unexpected storage growth. |
| `SAVE_FILE_DATA_CONTENT` | `bool` | `True` | When True, base64 content and URLs in the `file_data` field are parsed/downloaded and re-hosted in Open WebUI storage to prevent chat history bloat. When False, `file_data` values pass through untouched. |
| `BASE64_MAX_SIZE_MB` | `int` | `50` | Maximum size in MB for base64-encoded files/images before decoding. Larger payloads are rejected. |
| `IMAGE_UPLOAD_CHUNK_BYTES` | `int` | `1048576 (1 MiB)` | Max bytes buffered when loading Open WebUI-hosted images before forwarding them to a provider. Lower values reduce peak memory usage. |
| `VIDEO_MAX_SIZE_MB` | `int` | `100` | Maximum size in MB for video files (remote URLs or data URLs). Videos exceeding this limit are rejected. |
| `FALLBACK_STORAGE_EMAIL` | `str` | `env OPENROUTER_STORAGE_USER_EMAIL, else openrouter-pipe@system.local` | Owner email used when multimodal uploads occur without a chat user (for example, API automations). |
| `FALLBACK_STORAGE_NAME` | `str` | `env OPENROUTER_STORAGE_USER_NAME, else OpenRouter Pipe Storage` | Display name for the fallback storage owner. |
| `FALLBACK_STORAGE_ROLE` | `str` | `env OPENROUTER_STORAGE_USER_ROLE, else pending` | Role assigned to the fallback storage account when auto-created. Defaults to a low-privilege role; override only if your deployment needs a dedicated service role. |
| `ENABLE_SSRF_PROTECTION` | `bool` | `True` | Enable SSRF protection for remote URL downloads. When enabled, blocks requests to private IP ranges (localhost, RFC1918, link-local, etc.). |
| `MAX_INPUT_IMAGES_PER_REQUEST` | `int` | `5` | Maximum number of image inputs (user attachments plus assistant fallbacks) to include in a single provider request. |
| `IMAGE_INPUT_SELECTION` | `Literal[\"user_turn_only\", \"user_then_assistant\"]` | `user_then_assistant` | Controls which images are forwarded to the provider. `user_turn_only` restricts inputs to the current user message; `user_then_assistant` falls back to the most recent assistant-generated images when the user did not attach any. |

### Models, catalog refresh, and reasoning

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `MODEL_ID` | `str` | `auto` | Comma-separated OpenRouter model IDs to expose in Open WebUI. `auto` imports every available Responses-capable model. |
| `MODEL_CATALOG_REFRESH_SECONDS` | `int` | `3600` | How long to cache the OpenRouter model catalog (seconds) before refreshing. |
| `FREE_MODEL_FILTER` | `Literal["all","only","exclude"]` | `all` | Filter models based on summed OpenRouter pricing fields. `all` disables filtering; `only` restricts to free models (sum==0 and at least one numeric pricing value); `exclude` hides free models. |
| `TOOL_CALLING_FILTER` | `Literal["all","only","exclude"]` | `all` | Filter models based on tool calling support (supported_parameters includes `tools` or `tool_choice`). `all` disables filtering; `only` restricts to tool-capable models; `exclude` hides tool-capable models. |
| `UPDATE_MODEL_IMAGES` | `bool` | `True` | When enabled, sync OpenRouter model icons into Open WebUI model metadata (`meta.profile_image_url`) as PNG data URLs. Disabling avoids extra outbound fetches and model-metadata writes. |
| `UPDATE_MODEL_CAPABILITIES` | `bool` | `True` | When enabled, sync Open WebUI model capability checkboxes (`meta.capabilities`) from the OpenRouter catalog (and frontend capability signals like native web search). Disabling avoids model-metadata writes. |
| `UPDATE_MODEL_DESCRIPTIONS` | `bool` | `True` | When enabled, sync Open WebUI model descriptions (`meta.description`) from the OpenRouter `/models` catalog. Disabling avoids model-metadata writes and preserves operator-managed descriptions. |
| `ENABLE_REASONING` | `bool` | `True` | Enable reasoning requests whenever supported by the selected model/provider. |
| `THINKING_OUTPUT_MODE` | `Literal[\"open_webui\", \"status\", \"both\"]` | `open_webui` | Controls where in-progress thinking is surfaced while a response is being generated. |
| `ENABLE_ANTHROPIC_INTERLEAVED_THINKING` | `bool` | `True` | When enabled and the selected model is `anthropic/...`, sends `x-anthropic-beta: interleaved-thinking-2025-05-14` to opt into Claude interleaved thinking streams. |
| `ENABLE_ANTHROPIC_PROMPT_CACHING` | `bool` | `True` | When enabled and the selected model is `anthropic/...`, inserts `cache_control` breakpoints into system + user text blocks (up to 4) to enable Claude prompt caching for large stable prefixes (system prompts, tools, RAG context). |
| `ANTHROPIC_PROMPT_CACHE_TTL` | `Literal[\"5m\", \"1h\"]` | `5m` | TTL for Claude prompt caching breakpoints (ephemeral cache). System valve only; default `5m`. |
| `AUTO_CONTEXT_TRIMMING` | `bool` | `True` | Automatically attaches OpenRouter’s `middle-out` transform so long prompts are trimmed from the middle instead of failing with context errors. |
| `REASONING_EFFORT` | `Literal[\"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\"]` | `medium` | Default reasoning effort requested from supported models. |
| `REASONING_SUMMARY_MODE` | `Literal[\"auto\", \"concise\", \"detailed\", \"disabled\"]` | `auto` | Controls the reasoning summary emitted by supported models. |
| `GEMINI_THINKING_LEVEL` | `Literal[\"auto\", \"low\", \"high\"]` | `auto` | Controls `thinking_level` for Gemini 3.x models. `auto` maps minimal/low effort to LOW and everything else to HIGH. |
| `GEMINI_THINKING_BUDGET` | `int` | `1024` | Base thinking budget (tokens) for Gemini 2.5 models (0 disables thinking). |
| `PERSIST_REASONING_TOKENS` | `Literal[\"disabled\", \"next_reply\", \"conversation\"]` | `conversation` | Reasoning retention: `disabled` keeps nothing; `next_reply` keeps thoughts until the following assistant reply finishes; `conversation` keeps them for the full chat history. |
| `TASK_MODEL_REASONING_EFFORT` | `Literal[\"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\"]` | `low` | Reasoning effort requested for Open WebUI task payloads (titles/tags/etc.) when they target this pipe’s models. |

### Tool execution and function calling

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `TOOL_EXECUTION_MODE` | `Literal["Pipeline","Open-WebUI"]` | `Pipeline` | Select the tool execution backend. `Pipeline` executes tool calls inside the pipe (batching/breakers/special backends). `Open-WebUI` bypasses the internal executor and returns tool calls to Open WebUI to execute; tool result persistence in the pipe is disabled in this mode. |
| `ENABLE_STRICT_TOOL_CALLING` | `bool` | `True` | When True, converts Open WebUI registry tools to strict JSON Schema for more predictable function calling. Applies only when `TOOL_EXECUTION_MODE="Pipeline"` (pass-through forwards schemas as-is). |
| `MAX_FUNCTION_CALL_LOOPS` | `int` | `25` | Maximum number of full “model → tools → model” execution cycles allowed per request. |
| `MAX_PARALLEL_TOOLS_GLOBAL` | `int` | `200` | Maximum number of tool executions allowed concurrently per process. |
| `MAX_PARALLEL_TOOLS_PER_REQUEST` | `int` | `5` | Maximum number of tool executions allowed concurrently per request. |
| `BREAKER_MAX_FAILURES` | `int` | `5` | Number of failures allowed per breaker window before requests, tools, or DB ops are temporarily blocked. Set higher to reduce trip frequency in noisy environments. |
| `BREAKER_WINDOW_SECONDS` | `int` | `60` | Sliding window length (seconds) used when counting breaker failures. |
| `BREAKER_HISTORY_SIZE` | `int` | `5` | Maximum failures remembered per user/tool breaker. Increase when using very high `BREAKER_MAX_FAILURES` so history is not truncated. |
| `TOOL_BATCH_CAP` | `int` | `4` | Maximum number of tool calls executed in one batch (per loop) when batching is possible. |
| `TOOL_TIMEOUT_SECONDS` | `int` | `60` | Per-tool timeout (seconds). |
| `TOOL_BATCH_TIMEOUT_SECONDS` | `int` | `120` | Timeout (seconds) for completing an entire tool batch. |
| `TOOL_IDLE_TIMEOUT_SECONDS` | `Optional[int]` | `null` | Idle timeout (seconds) for tool execution when no progress is observed. |
| `TOOL_SHUTDOWN_TIMEOUT_SECONDS` | `float` | `10.0` | Maximum seconds to wait for per-request tool workers to drain/stop during request cleanup. `0` cancels immediately. |
| `PERSIST_TOOL_RESULTS` | `bool` | `True` | Persist tool call results across conversation turns. When disabled, tool results stay ephemeral. |
| `TOOL_OUTPUT_RETENTION_TURNS` | `int` | `10` | How many turns tool outputs remain replayable/available before being eligible for pruning. |

### Persistence, encryption, and compression

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ARTIFACT_ENCRYPTION_KEY` | `EncryptedStr` | `(empty)` | Encrypt reasoning tokens (and optionally all persisted artifacts). Changing the key creates a new table; prior artifacts become inaccessible. |
| `ENCRYPT_ALL` | `bool` | `True` | Encrypt every persisted artifact when `ARTIFACT_ENCRYPTION_KEY` is set. When False, only reasoning tokens are encrypted. |
| `ENABLE_LZ4_COMPRESSION` | `bool` | `True` | When True (and LZ4 is available), compress large encrypted artifacts to reduce DB read/write overhead. |
| `MIN_COMPRESS_BYTES` | `int` | `0` | Payloads at or above this size (bytes) are candidates for compression before encryption. `0` always attempts compression. |

### Streaming and concurrency

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `MAX_CONCURRENT_REQUESTS` | `int` | `200` | Maximum number of in-flight OpenRouter requests allowed per process. |
| `SSE_WORKERS_PER_REQUEST` | `int` | `4` | Number of stream processing workers spawned per request (fan-out for parsing/emitting). |
| `STREAMING_CHUNK_QUEUE_MAXSIZE` | `int` | `0` | Maximum number of raw SSE chunks buffered before applying backpressure. `0` means unbounded. |
| `STREAMING_EVENT_QUEUE_MAXSIZE` | `int` | `0` | Maximum number of parsed stream events buffered before applying backpressure. `0` means unbounded. |
| `STREAMING_EVENT_QUEUE_WARN_SIZE` | `int` | `1000` | Warning threshold for buffered stream events. |
| `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` | `int` | `0` | Maximum number of per-request items buffered for the middleware streaming bridge (`pipe(stream=True)` generator). `0` means unbounded. |
| `MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS` | `float` | `1.0` | When `MIDDLEWARE_STREAM_QUEUE_MAXSIZE>0`, maximum seconds to wait while enqueueing an item before aborting the request. Set to `0` to disable the timeout (not recommended). |

### Redis cache and cost snapshots

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ENABLE_REDIS_CACHE` | `bool` | `True` | Enable Redis write-behind cache when `REDIS_URL` and multi-worker mode are detected. |
| `REDIS_CACHE_TTL_SECONDS` | `int` | `600` | TTL (seconds) for cached artifacts/state stored in Redis. |
| `REDIS_PENDING_WARN_THRESHOLD` | `int` | `100` | Warn when Redis write-behind backlog exceeds this many pending items. |
| `REDIS_FLUSH_FAILURE_LIMIT` | `int` | `5` | Fail-open threshold: when flush failures reach this count, the pipe degrades and stops attempting flushes until conditions improve. |
| `COSTS_REDIS_DUMP` | `bool` | `False` | When True, push per-request usage snapshots into Redis for downstream cost analytics. |
| `COSTS_REDIS_TTL_SECONDS` | `int` | `900` | TTL (seconds) for cost snapshots stored in Redis. |

### Cleanup and database batching

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ARTIFACT_CLEANUP_DAYS` | `int` | `90` | Retention window (days) for persisted artifacts before cleanup (measured from `created_at`, which is refreshed on DB reads). |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` | `float` | `1.0` | Cleanup cadence (hours). |
| `DB_BATCH_SIZE` | `int` | `10` | Rows per DB transaction when draining Redis / batching persistence work. |

### Web search

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `AUTO_ATTACH_ORS_FILTER` | `bool` | `True` | Auto-enable the OpenRouter Search toggleable filter in each compatible model’s Advanced Settings (by updating the model’s `filterIds`), so the OpenRouter Search slider appears only where it works. |
| `AUTO_INSTALL_ORS_FILTER` | `bool` | `True` | Auto-install / auto-update the companion OpenRouter Search filter function into Open WebUI’s Functions DB (recommended with `AUTO_ATTACH_ORS_FILTER`). |
| `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER` | `bool` | `True` | Auto-enable OpenRouter Search by default on compatible models (by updating the model’s `defaultFilterIds`). This replicates “enabled by default” behavior while still allowing per-chat/per-model control. |
| `WEB_SEARCH_MAX_RESULTS` | `int` | `3` | Maximum number of results to request from the web search tool. |

See: [Web Search (Open WebUI) vs OpenRouter Search](web_search_owui_vs_openrouter_search.md).

**Note:** Open WebUI Direct Tool Servers are configured in Open WebUI (External Tools) and are not controlled by valves in this pipe.

### Direct uploads (bypass OWUI RAG)

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `AUTO_ATTACH_DIRECT_UPLOADS_FILTER` | `bool` | `True` | Auto-enable the Direct Uploads toggleable filter in each compatible model’s Advanced Settings (by updating the model’s `filterIds`), so the switch appears only where it can work. |
| `AUTO_INSTALL_DIRECT_UPLOADS_FILTER` | `bool` | `True` | Auto-install / auto-update the companion Direct Uploads filter function into Open WebUI’s Functions DB (recommended with `AUTO_ATTACH_DIRECT_UPLOADS_FILTER`). |

Notes:
- The **per-modality toggles** (files/audio/video) are implemented as **filter user valves** (under the Valves/knobs UI for the filter), not as separate switches in the Tools menu.
- Size limits and MIME/format allowlists are implemented as **filter valves** (configured on the filter function itself in Open WebUI).
- When direct uploads force `/chat/completions` (e.g. video, or audio formats not eligible for `/responses`) but an admin enforces `/responses` for the model, the pipe emits an **endpoint override conflict** error.
- Direct “files” are sent via `/responses` `input_file` when the request uses `/responses`, or via `/chat/completions` `type:"file"` blocks when the request must route to chat.

See: [OpenRouter Direct Uploads (bypass OWUI RAG)](openrouter_direct_uploads.md).

#### Companion filter valves (admin)

These are configured on the **Direct Uploads** filter function (Open WebUI → Admin → Functions → filter → Valves).

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `DIRECT_TOTAL_PAYLOAD_MAX_MB` | `int` | `50` | Maximum total size (MB) across all diverted direct uploads in a single request. |
| `DIRECT_FILE_MAX_UPLOAD_SIZE_MB` | `int` | `50` | Maximum size (MB) for a single diverted direct file upload. |
| `DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB` | `int` | `25` | Maximum size (MB) for a single diverted direct audio upload. |
| `DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB` | `int` | `20` | Maximum size (MB) for a single diverted direct video upload. |
| `DIRECT_FILE_MIME_ALLOWLIST` | `str` | `application/pdf,text/plain,text/markdown,application/json,text/csv` | Comma-separated MIME allowlist for diverted direct generic files. Non-allowlisted types are fail-open (left on normal OWUI RAG/Knowledge path). |
| `DIRECT_AUDIO_MIME_ALLOWLIST` | `str` | `audio/*` | Comma-separated MIME allowlist for diverted direct audio files. |
| `DIRECT_VIDEO_MIME_ALLOWLIST` | `str` | `video/mp4,video/mpeg,video/quicktime,video/webm` | Comma-separated MIME allowlist for diverted direct video files. |
| `DIRECT_AUDIO_FORMAT_ALLOWLIST` | `str` | `wav,mp3,aiff,aac,ogg,flac,m4a,pcm16,pcm24` | Comma-separated audio format allowlist (derived from filename/MIME and/or sniffed container). |
| `DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST` | `str` | `wav,mp3` | Comma-separated audio formats eligible for `/responses` `input_audio.format`. |

#### Companion filter user valves (per-user)

These appear in the filter’s user-facing “knobs” UI and control what gets diverted as direct uploads.

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `DIRECT_FILES` | `bool` | `False` | When enabled, divert eligible chat file uploads and forward them as direct document inputs. |
| `DIRECT_AUDIO` | `bool` | `False` | When enabled, divert eligible chat audio uploads and forward them as direct audio inputs. |
| `DIRECT_VIDEO` | `bool` | `False` | When enabled, divert eligible chat video uploads and forward them as direct video inputs (via `/chat/completions`). |

### Reporting, UI behavior, and request identifiers

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `USE_MODEL_MAX_OUTPUT_TOKENS` | `bool` | `False` | When enabled, forwards provider-advertised `max_output_tokens` automatically. |
| `SHOW_FINAL_USAGE_STATUS` | `bool` | `True` | Includes timing/cost/tokens in the final status message. |
| `ENABLE_STATUS_CSS_PATCH` | `bool` | `True` | Injects a CSS helper so multi-line statuses render cleanly in the Open WebUI UI. |
| `SEND_END_USER_ID` | `bool` | `False` | When enabled, sends the OpenRouter top-level `user` field using the Open WebUI user ID, and also adds `metadata.user_id`. See [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md). |
| `SEND_SESSION_ID` | `bool` | `False` | When enabled, sends OpenRouter `session_id` using Open WebUI `__metadata__[\"session_id\"]` (if present) and adds `metadata.session_id`. |
| `SEND_CHAT_ID` | `bool` | `False` | When enabled, adds `metadata.chat_id` using Open WebUI `__metadata__[\"chat_id\"]`. |
| `SEND_MESSAGE_ID` | `bool` | `False` | When enabled, adds `metadata.message_id` using Open WebUI `__metadata__[\"message_id\"]`. |

### Session log storage

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `SESSION_LOG_STORE_ENABLED` | `bool` | `False` | When True, persist per-request SessionLogger output to encrypted zip files on disk. Persistence is skipped when required IDs are missing (`user_id`, `session_id`, `chat_id`, `message_id`). |
| `SESSION_LOG_DIR` | `str` | `session_logs` | Base directory for encrypted session log archives. |
| `SESSION_LOG_ZIP_PASSWORD` | `EncryptedStr` | `(empty)` | Password used to encrypt session log zip files (pyzipper AES). |
| `SESSION_LOG_RETENTION_DAYS` | `int` | `90` | Retention window (days) for stored session log archives. |
| `SESSION_LOG_CLEANUP_INTERVAL_SECONDS` | `int` | `3600` | How often (seconds) to run the session log cleanup loop when storage is enabled. |
| `SESSION_LOG_ZIP_COMPRESSION` | `Literal[\"stored\", \"deflated\", \"bzip2\", \"lzma\"]` | `lzma` | Zip compression algorithm for session log archives. |
| `SESSION_LOG_ZIP_COMPRESSLEVEL` | `Optional[int]` | `null` | Compression level (0–9) for deflated/bzip2 compression. Ignored for stored/lzma. |
| `SESSION_LOG_MAX_LINES` | `int` | `20000` | Maximum number of in-memory SessionLogger records retained per request (older entries are dropped). |
| `SESSION_LOG_FORMAT` | `Literal[\"jsonl\", \"text\", \"both\"]` | `jsonl` | Archive log file format: `jsonl` writes `logs.jsonl`, `text` writes `logs.txt`, `both` writes both files. |

### Support contact and error templates

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `SUPPORT_EMAIL` | `str` | `(empty)` | Optional support email address inserted into user-facing error templates. |
| `SUPPORT_URL` | `str` | `(empty)` | Optional support URL inserted into user-facing error templates. |
| `OPENROUTER_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for OpenRouter 400 responses. Supports Handlebars-style `{{#if var}}...{{/if}}` blocks. |
| `ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE` | `str` | `built-in default` | Markdown template emitted when a request requires a different OpenRouter endpoint than the one enforced by endpoint override valves. |
| `DIRECT_UPLOAD_FAILURE_TEMPLATE` | `str` | `built-in default` | Markdown template emitted when OpenRouter Direct Uploads cannot be applied (e.g. incompatible attachment combinations or pre-flight validation failures). |
| `AUTHENTICATION_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for OpenRouter auth failures. |
| `INSUFFICIENT_CREDITS_TEMPLATE` | `str` | `built-in default` | Markdown template for OpenRouter “insufficient credits” failures. |
| `RATE_LIMIT_TEMPLATE` | `str` | `built-in default` | Markdown template for OpenRouter rate limits. |
| `SERVER_TIMEOUT_TEMPLATE` | `str` | `built-in default` | Markdown template for upstream/provider timeouts. |
| `NETWORK_TIMEOUT_TEMPLATE` | `str` | `built-in default` | Markdown template for network timeouts. |
| `CONNECTION_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for connection failures. |
| `SERVICE_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for OpenRouter 5xx errors. |
| `INTERNAL_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for unexpected internal errors. |
| `MAX_FUNCTION_CALL_LOOPS_REACHED_TEMPLATE` | `str` | `built-in default` | Markdown template emitted when the request reaches `MAX_FUNCTION_CALL_LOOPS` while the model is still requesting additional tool/function calls. |
| `MODEL_RESTRICTED_TEMPLATE` | `str` | `built-in default` | Markdown template emitted when the requested model is blocked by `MODEL_ID` and/or model filter valves. |

**Note:** To customize templates safely, prefer small edits and validate with real error cases. Template variable sets and formatting expectations are described in [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md) and [Error Handling & User Experience](error_handling_and_user_experience.md).

### Logging

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `LOG_LEVEL` | `Literal[\"DEBUG\", \"INFO\", \"WARNING\", \"ERROR\", \"CRITICAL\"]` | `env GLOBAL_LOG_LEVEL, else INFO` | Select logging level. Recommend INFO or WARNING for production; use DEBUG for diagnosis. |

---

## User valves (`Pipe.UserValves`)

User valves provide per-user behavior overrides for a subset of settings.

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `SHOW_FINAL_USAGE_STATUS` | `bool` | `True` | Display tokens, time, and cost at the end of each reply. |
| `ENABLE_REASONING` | `bool` | `True` | While the AI works, show its step-by-step reasoning when supported. |
| `THINKING_OUTPUT_MODE` | `Literal[\"open_webui\", \"status\", \"both\"]` | `open_webui` | Choose where to show the model’s thinking while it works. |
| `ENABLE_ANTHROPIC_INTERLEAVED_THINKING` | `bool` | `True` | When enabled and the selected model is `anthropic/...`, send `x-anthropic-beta: interleaved-thinking-2025-05-14` to opt into Claude interleaved thinking streams. |
| `REASONING_EFFORT` | `Literal[\"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\"]` | `medium` | Choose how much thinking the AI should do before answering (higher depth is slower but more thorough). |
| `REASONING_SUMMARY_MODE` | `Literal[\"auto\", \"concise\", \"detailed\", \"disabled\"]` | `auto` | Choose how detailed the reasoning summary should be. |
| `PERSIST_REASONING_TOKENS` | `Literal[\"disabled\", \"next_reply\", \"conversation\"]` | `next_reply` | User-level reasoning retention preference. |
| `PERSIST_TOOL_RESULTS` | `bool` | `True` | Let the AI reuse outputs from tools later in the conversation. |
