# Valves & Configuration Atlas

This document is the authoritative reference for the pipe’s configuration surface: **Open WebUI valves**.

Defaults and valve names are verified against the source code and are intended to match current runtime behavior.

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
| `DEFAULT_LLM_ENDPOINT` | `Literal["responses", "chat_completions"]` | `responses` | Which OpenRouter endpoint to use by default. `responses` applies prompt caching via a top-level `cache_control`; `chat_completions` uses per-block breakpoints, needed for Bedrock/Vertex-routed Claude caching and some provider features. Per-model force lists override it. |
| `FORCE_CHAT_COMPLETIONS_MODELS` | `str` | `""` | Comma-separated glob patterns of model ids forced to `/chat/completions`. Matches both slash and dotted ids; case-sensitive. Globs are literal about the `~` prefix, so `~anthropic/...` router aliases need their own pattern (e.g. `~anthropic/*`). |
| `FORCE_RESPONSES_MODELS` | `str` | `""` | Comma-separated glob patterns of model ids forced to `/responses`. Overrides `FORCE_CHAT_COMPLETIONS_MODELS` when both match. |
| `AUTO_FALLBACK_CHAT_COMPLETIONS` | `bool` | `True` | Retry once against `/chat/completions` if `/responses` fails with an endpoint/model-support error before any visible output has streamed. |
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
| `SAVE_REMOTE_FILE_URLS` | `bool` | `True` | When True, remote URLs and data URLs in the `file_url` field are downloaded/parsed and re-hosted in Open WebUI storage (default; keeps chats replayable if the source link dies, at the cost of storage growth). When False, `file_url` values pass through untouched. |
| `SAVE_FILE_DATA_CONTENT` | `bool` | `True` | When True, base64 content and URLs in the `file_data` field are parsed/downloaded and re-hosted in Open WebUI storage to prevent chat history bloat. When False, `file_data` values pass through untouched. |
| `BASE64_MAX_SIZE_MB` | `int` | `50` | Maximum size in MB for base64-encoded files/images before decoding. Larger payloads are rejected. |
| `IMAGE_UPLOAD_CHUNK_BYTES` | `int` | `1048576 (1 MiB)` | Max bytes buffered when loading Open WebUI-hosted images before forwarding them to a provider. Lower values reduce peak memory usage. |
| `VIDEO_MAX_SIZE_MB` | `int` | `100` | Max MB for inline base64 (`data:`) videos and for stored videos re-read to extract frames; oversized are rejected/skipped. Remote `http(s)`/YouTube video links are forwarded to the provider unmeasured. |
| `FALLBACK_STORAGE_EMAIL` | `str` | `env OPENROUTER_STORAGE_USER_EMAIL, else openrouter-pipe@system.local` | Owner email used when multimodal uploads occur without a chat user (for example, API automations). |
| `FALLBACK_STORAGE_NAME` | `str` | `env OPENROUTER_STORAGE_USER_NAME, else OpenRouter Pipe Storage` | Display name for the fallback storage owner. |
| `FALLBACK_STORAGE_ROLE` | `str` | `env OPENROUTER_STORAGE_USER_ROLE, else pending` | Role assigned to the fallback storage account when auto-created. Defaults to a low-privilege role; override only if your deployment needs a dedicated service role. |
| `ENABLE_SSRF_PROTECTION` | `bool` | `True` | Enable SSRF protection for remote URL downloads. When enabled, blocks requests to private IP ranges (localhost, RFC1918, link-local, etc.). HTTPS-only defaults still apply even if SSRF protection is disabled. |
| `ALLOW_INSECURE_HTTP` | `bool` | `False` | Allow plaintext HTTP remote URLs when explicitly enabled. HTTP is disabled by default; only enable alongside a narrow allowlist. |
| `ALLOW_INSECURE_HTTP_HOSTS` | `str` | `""` | Comma-separated list of hosts or host:port entries allowed for plaintext HTTP. Exact match only (no wildcards). Empty means no HTTP allowed. Example: `example.com, example.org:8080, 203.0.113.10`. |
| `ALLOW_UNKNOWN_SIZE_CLOUD_READS` | `bool` | `False` | Global Valve only (not user-overridable). When `STORAGE_PROVIDER` is non-local (s3/gcs/azure) and a referenced Open WebUI file has no declared size in its metadata, reads fail closed by default to avoid an unbounded cloud download. Set `True` to permit such reads; the file is still copied to a private temp and capped by `BASE64_MAX_SIZE_MB` after download. This is a download-safety gate, not an authorization bypass. |
| `MAX_INPUT_IMAGES_PER_REQUEST` | `int` | `5` | Maximum number of image inputs (user attachments plus assistant fallbacks) to include in a single provider request. |
| `IMAGE_INPUT_SELECTION` | `Literal[\"user_turn_only\", \"user_then_assistant\"]` | `user_then_assistant` | Controls which images are forwarded to the provider. `user_turn_only` restricts inputs to the current user message; `user_then_assistant` falls back to the most recent assistant-generated images when the user did not attach any. |

### Models, catalog refresh, and reasoning

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `MODEL_ID` | `str` | `auto` | Comma-separated OpenRouter model IDs to expose in Open WebUI. `auto` imports every available Responses-capable model. |
| `MODEL_CATALOG_REFRESH_SECONDS` | `int` | `3600` | How long to cache the OpenRouter model catalog (seconds) before refreshing. |
| `VARIANT_MODELS` | `str` | `""` (empty) | Comma-separated virtual routing variants, each `base_id:variant_tag` — e.g. `openai/gpt-4o:exacto,anthropic/claude-sonnet-4.5:extended`. Each adds a picker model that reuses the base model's description, icon and capabilities with the tag appended to the display name, and sends the full `base_id:tag` upstream so OpenRouter applies that routing. Documented tags: `free`, `thinking`, `online`, `nitro`, `exacto`, `extended` (the tag is lower-cased and passed through, not validated against that list). A case-sensitive `base_id@preset/slug` form adds an OpenRouter preset instead. An entry whose base model is absent from the catalog is skipped, with a warning naming it. |
| `NEW_MODEL_ACCESS_CONTROL` | `Literal["public","admins"]` | `admins` | Default access grants for **new** OpenRouter model overlays inserted into Open WebUI (existing access grants are preserved on update). `public` grants read access to all users (wildcard access grant). `admins` creates no access grants (private) and relies on Open WebUI's `BYPASS_ADMIN_ACCESS_CONTROL` for admin access; otherwise admins must be granted access explicitly. |
| `FREE_MODEL_FILTER` | `Literal["all","only","exclude"]` | `all` | Filter models based on summed OpenRouter pricing fields. `all` disables filtering; `only` restricts to free models (sum==0 and at least one numeric pricing value); `exclude` hides free models. |
| `TOOL_CALLING_FILTER` | `Literal["all","only","exclude"]` | `all` | Filter models based on tool calling support (supported_parameters includes `tools` or `tool_choice`). `all` disables filtering; `only` restricts to tool-capable models; `exclude` hides tool-capable models. |
| `ZDR_MODELS_ONLY` | `bool` | `False` | Hide models that are not ZDR-capable (based on `/endpoints/zdr`). Catalog filter and request admission: a hidden model is also refused if requested directly. Never sends `provider.zdr=true`. |
| `ZDR_ENFORCE` | `bool` | `False` | Enforce ZDR on every request by sending `provider.zdr=true` and rejecting non‑ZDR models. Routing suffixes the pipe synthesises (`:nitro`, `:floor`, `:online`) are checked against their base model, while a suffix OpenRouter lists as its own model (`:free`, `:thinking`) is judged on its own endpoints; `provider.zdr=true` then restricts routing to ZDR endpoints server-side. |
| `ALLOW_USER_ZDR_OVERRIDE` | `bool` | `True` | Allow users to request ZDR per chat via `REQUEST_ZDR` (ignored when `ZDR_ENFORCE` is enabled). |
| `UPDATE_MODEL_IMAGES` | `bool` | `True` | When enabled, sync OpenRouter model icons into Open WebUI model metadata (`meta.profile_image_url`) as PNG data URLs. Disabling avoids extra outbound fetches and model-metadata writes. |
| `UPDATE_MODEL_CAPABILITIES` | `bool` | `True` | When enabled, sync Open WebUI model capability checkboxes (`meta.capabilities`) from the OpenRouter catalog (and frontend capability signals like native web search). Also the switch that lets two boxes be unticked on models that answer with a picture or a clip: `File context` on every one of them, and `Built-in tools` when `DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS` is on. Both are written only where the model has no setting yet. Disabling avoids model-metadata writes. |
| `DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS` | `bool` | `True` | Unticks Open WebUI's `Built-in tools` box on image and video models when the pipe first adds them, because those models answer with a picture or a clip rather than a tool call. Tick it back on for a single model and that choice is kept; the pipe fills this in only where no setting exists. Requires `UPDATE_MODEL_CAPABILITIES`. Covers `Built-in tools` alone — `File context` is unticked on the same models whenever `UPDATE_MODEL_CAPABILITIES` is on, whatever this is set to, because leaving it on makes an attachment trigger an extra billed round-trip whose retrieved text is pasted into a picture or clip prompt. |
| `UPDATE_MODEL_DESCRIPTIONS` | `bool` | `False` | When enabled, sync Open WebUI model descriptions (`meta.description`) from the OpenRouter frontend catalog. Disabling avoids model-metadata writes and preserves operator-managed descriptions. |
| `ENABLE_REASONING` | `bool` | `True` | Enable reasoning requests whenever supported by the selected model/provider. |
| `THINKING_OUTPUT_MODE` | `Literal[\"open_webui\", \"status\", \"both\"]` | `open_webui` | Controls where in-progress thinking is surfaced while a response is being generated. |
| `ENABLE_ANTHROPIC_INTERLEAVED_THINKING` | `bool` | `True` | When enabled and the selected model is Anthropic (`anthropic/...` or a `~anthropic/...` router alias), sends `x-anthropic-beta: interleaved-thinking-2025-05-14` to opt into Claude interleaved thinking streams. |
| `ENABLE_ANTHROPIC_PROMPT_CACHING` | `bool` | `True` | When enabled and the selected model is Anthropic (`anthropic/...` or a `~anthropic/...` router alias), enables Claude prompt caching. On `/responses` (the default endpoint) a single top-level `cache_control` is sent (OpenRouter routes Anthropic-direct for top-level caching, excluding Bedrock/Vertex). On `/chat/completions` per-block `cache_control` breakpoints are inserted instead (up to Anthropic's 4-breakpoint limit; adaptive allocation: with tools → 1 tool + 1 system + 2 user = 4; without tools → 1 system + 3 user = 4), which also works across Bedrock/Vertex. Tool conversion functions preserve any existing `cache_control` on tool definitions. |
| `ANTHROPIC_PROMPT_CACHE_TTL` | `Literal[\"5m\", \"1h\"]` | `5m` | TTL for Claude prompt caching breakpoints (ephemeral cache). System valve only; default `5m`. |
| `AUTO_CONTEXT_TRIMMING` | `bool` | `True` | Automatically enables OpenRouter’s `context-compression` plugin so long prompts are trimmed from the middle instead of failing with context errors. |
| `REASONING_EFFORT` | `Literal[\"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\"]` | `medium` | Default reasoning effort requested from supported models. For Claude Opus/Sonnet models, `xhigh` additionally sets `verbosity: "max"` (maps to Anthropic's `output_config.effort: "max"`); older Claude models gracefully fall back to `high`. |
| `REASONING_SUMMARY_MODE` | `Literal[\"auto\", \"concise\", \"detailed\", \"disabled\"]` | `auto` | Controls the reasoning summary emitted by supported models. |
| `GEMINI_THINKING_BUDGET` | `int` | `1024` | Base thinking budget (tokens) for Gemini 2.5 models (0 disables thinking). |
| `PERSIST_REASONING_TOKENS` | `Literal[\"disabled\", \"next_reply\", \"conversation\"]` | `conversation` | Reasoning retention: `disabled` keeps nothing; `next_reply` keeps thoughts until the following assistant reply finishes; `conversation` keeps them for the full chat history. |
| `TASK_MODEL_REASONING_EFFORT` | `Literal[\"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\"]` | `low` | Reasoning effort requested for Open WebUI task payloads (titles/tags/etc.) when they target this pipe’s models. |

See: [OpenRouter Zero Data Retention (ZDR)](openrouter_zdr.md).

### Tool execution and function calling

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `TOOL_EXECUTION_MODE` | `Literal["Pipeline","Open-WebUI"]` | `Pipeline` | Select the tool execution backend. `Pipeline` executes tool calls inside the pipe (batching/breakers/special backends). `Open-WebUI` bypasses the internal executor and returns tool calls to Open WebUI to execute; tool result persistence in the pipe is disabled in this mode. |
| `SHOW_TOOL_CARDS` | `bool` | `False` | Show collapsible cards in chat with tool name, arguments, and results. When disabled (default), tools run silently without visual status indicators. Pipe-executed tool cards apply only when `TOOL_EXECUTION_MODE="Pipeline"`; OpenRouter server-tool cards (web search, fetch, datetime, advisor, subagent) render in either mode. |
| `ENABLE_STRICT_TOOL_CALLING` | `bool` | `True` | When True, converts Open WebUI registry tools to strict JSON Schema for more predictable function calling. Applies only when `TOOL_EXECUTION_MODE="Pipeline"` (pass-through forwards schemas as-is). |
| `MAX_FUNCTION_CALL_LOOPS` | `int` | `25` | Maximum number of full “model → tools → model” execution cycles allowed per request. Applies only when `TOOL_EXECUTION_MODE=”Pipeline”`. When the limit is reached, pending tool calls receive stub responses so the model can synthesize a final answer. |
| `MAX_PARALLEL_TOOLS_GLOBAL` | `int` | `200` | Maximum number of tool executions allowed concurrently per process. |
| `MAX_PARALLEL_TOOLS_PER_REQUEST` | `int` | `5` | Maximum number of tool executions allowed concurrently per request. |
| `BREAKER_MAX_FAILURES` | `int` | `5` | Number of failures allowed per breaker window before requests, tools, or DB ops are temporarily blocked. Set higher to reduce trip frequency in noisy environments. |
| `BREAKER_WINDOW_SECONDS` | `int` | `60` | Sliding window length (seconds) used when counting breaker failures. |
| `BREAKER_HISTORY_SIZE` | `int` | `5` | Maximum failures the per-user database circuit breaker remembers. Keep at or above `BREAKER_MAX_FAILURES` so history is not truncated below the trip count (request and per-tool breakers size their own history automatically). |
| `TOOL_BATCH_CAP` | `int` | `4` | Maximum number of tool calls executed in one batch (per loop) when batching is possible. |
| `TOOL_TIMEOUT_SECONDS` | `int` | `60` | Per-tool timeout (seconds). |
| `TOOL_BATCH_TIMEOUT_SECONDS` | `int` | `120` | Timeout (seconds) for completing an entire tool batch. |
| `TOOL_IDLE_TIMEOUT_SECONDS` | `Optional[int]` | `null` | Idle timeout (seconds) for tool execution when no progress is observed. |
| `TOOL_SHUTDOWN_TIMEOUT_SECONDS` | `float` | `10.0` | Maximum seconds to wait for per-request tool workers to drain/stop during request cleanup. `0` cancels immediately. |
| `PERSIST_TOOL_RESULTS` | `bool` | `False` | Persist tool call results across conversation turns. Disabled by default: later turns rely on the assistant's summaries and can re-run tools; enable to replay raw tool outputs. |
| `TOOL_OUTPUT_RETENTION_TURNS` | `int` | `10` | How many turns tool outputs remain replayable/available before being eligible for pruning. |

Behavior note (no valve):
- Pipeline mode applies dynamic, model-aware tool output budgeting. Oversized live/replayed `function_call_output` payloads can be replaced with omission stubs so the model stays in-context.
- When `MAX_FUNCTION_CALL_LOOPS` is reached, pending tool calls receive stub responses advising the model to synthesize from existing context, and the model gets one additional generation turn.
- Failed/omitted tool outputs remain model-visible for continuation, but are not persisted and not rendered as tool cards.

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
| `STREAMING_CHUNK_QUEUE_WARN_SIZE` | `int` | `1000` | Warning threshold for the buffered raw-chunk queue; logs a rate-limited backend warning when the backlog is high (monitoring only). |
| `STREAMING_EVENT_QUEUE_MAXSIZE` | `int` | `0` | Maximum number of parsed stream events buffered before applying backpressure. `0` means unbounded. |
| `STREAMING_EVENT_QUEUE_WARN_SIZE` | `int` | `1000` | Warning threshold for buffered stream events. |
| `STREAMING_DELTA_CHAR_LIMIT` | `int` | `256` | Nagle coalescing toggle. `> 0` enables adaptive backpressure-driven batching for both text and reasoning deltas. `0` (with `IDLE_FLUSH_MS=0`) = passthrough mode (1:1 emission). See [Streaming Pipeline § 6](streaming_pipeline_and_emitters.md#6-nagle-inspired-adaptive-delta-coalescing). |
| `STREAMING_IDLE_FLUSH_MS` | `int` | `30` | Idle flush timeout (ms) for the Nagle coalescer. Ensures buffered deltas are delivered when the upstream producer pauses. `0` disables time-based flushing. |
| `STREAMING_NAGLE_MIN_FLUSH_CHARS` | `int` | `3` | Minimum buffered chars before the Nagle coalescer yields a batch at end-of-cycle. `3` smooths single-char jitter (default). `1` = pure Nagle. `5–10` = aggressive reduction. Idle timeout still guarantees delivery. |
| `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` | `int` | `0` | Maximum number of per-request items buffered for the middleware streaming bridge (`pipe(stream=True)` generator). `0` means unbounded. |
| `MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS` | `float` | `1.0` | When `MIDDLEWARE_STREAM_QUEUE_MAXSIZE>0`, maximum seconds to wait while enqueueing an item before dropping that item and continuing the stream. Set to `0` to disable the timeout (not recommended). |

### Redis cache and cost snapshots

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ENABLE_REDIS_CACHE` | `bool` | `True` | Buffer artifact writes through Redis when `REDIS_URL` and multi-worker mode are detected. |
| `REDIS_CACHE_TTL_SECONDS` | `int` | `600` | TTL (seconds) for cached artifacts/state stored in Redis. |
| `REDIS_PENDING_WARN_THRESHOLD` | `int` | `100` | Warn when Redis write-behind backlog exceeds this many pending items. |
| `REDIS_FLUSH_FAILURE_LIMIT` | `int` | `5` | Alert threshold: after this many consecutive flush failures the pipe logs a critical alert; write-behind is not disabled — the flusher backs off and keeps retrying, resuming when flushes succeed. |
| `COSTS_REDIS_DUMP` | `bool` | `False` | When True, push per-request usage snapshots into Redis for downstream cost analytics. |
| `COSTS_REDIS_TTL_SECONDS` | `int` | `900` | TTL (seconds) for cost snapshots stored in Redis. |

### Cleanup and database batching

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ARTIFACT_CLEANUP_DAYS` | `int` | `90` | Retention window (days) for persisted artifacts before cleanup (measured from `created_at`, which is refreshed on DB reads). |
| `ARTIFACT_CLEANUP_INTERVAL_HOURS` | `float` | `1.0` | Cleanup cadence (hours). |
| `DB_BATCH_SIZE` | `int` | `10` | Rows per DB transaction when draining Redis / batching persistence work. |

### OpenRouter server tools

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ENABLE_WEB_SEARCH` | `bool` | `True` | Enable the OpenRouter Web Search server tool. When disabled, web search toggles are hidden from users. |
| `ENABLE_WEB_FETCH` | `bool` | `True` | Enable the OpenRouter Web Fetch server tool. When disabled, web fetch toggles are hidden from users. |
| `ENABLE_DATETIME` | `bool` | `True` | Enable the OpenRouter Datetime server tool (free, no additional cost). When disabled, datetime toggles are hidden from users. |
| `ENABLE_ADVISOR` | `bool` | `True` | Enable the OpenRouter Advisor server tool (consult a higher-intelligence model mid-generation; extra paid call, default-off per chat). When disabled, advisor toggles are hidden from users. |
| `ENABLE_SUBAGENT` | `bool` | `True` | Enable the OpenRouter Subagent server tool (delegate tasks to a cheaper worker model; extra paid call, default-off per chat). When disabled, subagent toggles are hidden from users. |
| `ENABLE_SEARCH_MODELS` | `bool` | `True` | Enable the OpenRouter model-search server tool (let the model search the OpenRouter catalog). When disabled, model-search toggles are hidden from users. |
| `ENABLE_IMAGE_GENERATION` | `bool` | `True` | Enable the OpenRouter Image Generation server tool. When disabled, image generation toggles are hidden from users. |
| `AUTO_INSTALL_WEB_TOOLS_FILTER` | `bool` | `True` | Automatically install/update the OpenRouter Web Tools filter function in Open WebUI. |
| `AUTO_ATTACH_WEB_TOOLS_FILTER` | `bool` | `True` | Automatically attach the OpenRouter Web Tools filter to all pipe models (so the toggle appears in the Integrations menu). |
| `AUTO_DEFAULT_WEB_TOOLS_FILTER` | `bool` | `False` | When enabled, marks the OR Web Tools filter as a Default Filter on all pipe models (pre-enabled per chat; users can still turn it off). |
| `AUTO_INSTALL_IMAGE_GEN_FILTER` | `bool` | `True` | Automatically install/update the OpenRouter Image Generation filter function in Open WebUI. |
| `AUTO_ATTACH_IMAGE_GEN_FILTER` | `bool` | `True` | Automatically attach the OpenRouter Image Generation filter to all pipe models. |

See: [OpenRouter Server Tools](openrouter_server_tools.md) for filter valves, user valves, and the data flow.
See: [Web Search: OWUI vs OpenRouter](web_search_owui_vs_openrouter_search.md) for the distinction between Open WebUI native web search and OpenRouter web search.

**Note:** Open WebUI Direct Tool Servers are configured in Open WebUI (External Tools) and are not controlled by valves in this pipe.

### OpenRouter native image generation

Image-output models (Sourceful Riverflow, Black Forest Labs FLUX, ByteDance Seedream, Google Gemini Image, OpenAI GPT-5 Image, etc.) — **distinct** from the legacy `openrouter_image_gen` filter (OpenAI Responses-API server tool) controlled by `ENABLE_IMAGE_GENERATION` / `AUTO_INSTALL_IMAGE_GEN_FILTER` / `AUTO_ATTACH_IMAGE_GEN_FILTER` above.

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ENABLE_OPENROUTER_IMAGE_GENERATION` | `bool` | `True` | Expose OpenRouter native image-output models as chat models. Pure-image-only models (FLUX, Riverflow, Seedream) are discovered via `/api/v1/models?output_modalities=image`. Multimodal text+image models (gpt-5-image, gemini-image variants) stay in the chat catalog and get their own settings panel, like every other image model. Setting this to `False` empties the image model list and clears its refresh timestamp, so pure-image-only models vanish from OWUI's dropdown immediately. |
| `AUTO_INSTALL_IMAGE_FILTERS` | `bool` | `True` | Install and keep up to date one settings panel per image model, offering the settings that model publishes to OpenRouter plus four controls every panel carries (`IMAGE_PROVIDER_OPTIONS_JSON`, `IMAGE_REFERENCE_MODE`, `IMAGE_REFERENCE_URLS`, `IMAGE_SIZE`). If a model's settings list cannot be read on a refresh it keeps the settings from its last successful read; a model never read gets no panel at all rather than a guessed set. |
| `AUTO_ATTACH_IMAGE_FILTERS` | `bool` | `True` | Attach each image model's own settings panel to it, so its settings appear in the chat controls when that model is selected. A single model can opt out with the `disable_image_filter_auto_attach` advanced parameter. |
| `AUTO_DEFAULT_IMAGE_FILTERS` | `bool` | `True` | Always keep the attached image filters enabled by default on image-output models. Re-asserted on every catalog metadata sync. Setting this to `False` stops new models being defaulted but does not detach panels already marked default — clear those on the model itself. |

Notes:
- The image catalog uses the **shared** `MODEL_CATALOG_REFRESH_SECONDS` TTL (no separate cache valve).
- Generated images are stored, linked to the chat and rendered by the same path that has handled `gpt-5-image` end-to-end since long before this feature, so they behave like any other attachment in the conversation.
- `OR Web Tools` and `OR Web Search` overlays are **capability-gated to skip image-output models** — these models do not support tool use and would fail with HTTP 404 ("No endpoints found that support tool use") if web search were attached. Video-generation models are skipped for the same reason.
- Validation: a setting the model does not publish is dropped before the request goes out and the user is told which one, rather than being sent and refused by the provider. A provider option typed as a JSON list or object that does not parse is rejected as soon as it is entered, with the setting named.
- Multimodal image models on the chat route are validated the same way. Their settings are put through the same published contract on the way out, so a value outside it is withheld and named rather than sent.

See: [OpenRouter Image Generation](openrouter_image_generation.md).

#### Companion filter user valves (per-user, per-filter)

One filter is installed per image model, named after that model, and attached
only to it.

Its user valves are not fixed and are not listed here: they are built from the
settings the model publishes to OpenRouter, so they differ per model and change
when the model does. A typed parameter becomes a field named `IMAGE_<PARAMETER>`
— `IMAGE_ASPECT_RATIO` for `aspect_ratio`, and so on. A provider-specific option
becomes a free-text field under the same naming rule.

For a worked example, `recraft/recraft-v3` publishes `aspect_ratio` (six
choices), `n` (1 to 6) and the provider options `style`, `controls` and
`text_layout`, so its filter carries `IMAGE_ASPECT_RATIO`, `IMAGE_N`,
`IMAGE_STYLE`, `IMAGE_CONTROLS` and `IMAGE_TEXT_LAYOUT`.

Four controls appear on **every** image model that has a panel at all, whatever
its contract publishes, because a request carries them for any model and no
model's contract describes them:

| Valve | Title in chat | Type | Default | Maps to |
| --- | --- | --- | --- | --- |
| `IMAGE_PROVIDER_OPTIONS_JSON` | Provider options | `str` (JSON object) | `""` | `provider.options`, keyed by provider slug |
| `IMAGE_REFERENCE_MODE` | Reference images | `Literal["auto", "latest-only", "none"]` | `"auto"` | which attached images become `input_references` |
| `IMAGE_REFERENCE_URLS` | Reference image links | `str` (JSON array) | `""` | extra `input_references` entries, placed first |
| `IMAGE_SIZE` | Output size | `str` | `""` | top-level `size`, sent unvalidated |

`IMAGE_PROVIDER_OPTIONS_JSON` is the image sibling of
`VIDEO_PROVIDER_OPTIONS_JSON` and writes the same place the provider routing
filter writes, merging into it rather than replacing it — so a user typing one
option does not discard an operator's routing choice. Six of the forty recorded
models publish an empty passthrough list, so it is the only way to address a
provider on those. `IMAGE_SIZE` is rendered on every model because **no**
endpoint record publishes a `size` descriptor; its value is sent as typed and
the provider decides, which the control's own text says.

`IMAGE_REFERENCE_MODE` chooses which of the pictures attached to the turn are
sent as references: `auto` sends every one, oldest first; `latest-only` sends
just the most recent; `none` sends none of them. `IMAGE_REFERENCE_URLS` takes a
JSON list of `https` links or `data:` URLs, placed ahead of the attached ones so
an explicit choice survives on the many models that take a single reference.
Those links go through the same safety gate as every other URL the pipe fetches,
and a refused link fails the request rather than generating without it. A
request carries at most 16 references, the request format's own ceiling; where a
model publishes a lower limit the lower one applies, and anything over it is
dropped with a note in the chat saying how many and why.

A provider option whose accepted values OpenRouter publishes renders as a choice
rather than free text — today that is `moderation` (`auto`, `low`), delivered on
the six OpenAI image models that name it.

Where a model is served by several providers whose published choices differ, the
values only some of them accept are still offered, marked on the control; the
pipe fits the chosen value to whichever company serves the request and tells you
if that company does not accept it. Measured across all forty recorded
models this affects one value: `4K` for `resolution` on
`google/gemini-3-pro-image`.

The prompt a model receives is the message typed in the chat with the model's
own system text in front of it, so a Workspace model's house style reaches an
image model instead of being dropped. A turn with no typed text is still refused
rather than generating from the system text alone.

Where every provider that could serve a request publishes native streaming, the
request asks for the streamed form and each preview is reported as a status
line; vector models stream text rather than pictures and report `Drawing the
image…` once. Either way the answer is the same finished-image markdown.

**Skip-when-default sentinel**: an empty string for text and choice fields, and
an empty numeric field for numbers, means "not set" and is left out of the
request, so the model's own default applies. Numbers use an empty field rather
than `0` because `0` is a legal value for some published settings.

**Routing**: the filter writes the chosen values flat into `body.image_config`.
The image adapter then splits them, using the same published record, into
top-level request parameters and provider-scoped options. A filter writes only
for the model it belongs to, in any of the id forms Open WebUI produces.

### OpenRouter video generation

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ENABLE_VIDEO_GENERATION` | `bool` | `True` | Expose OpenRouter async video-generation models as chat models. Video models are always treated as not ZDR-capable. |
| `AUTO_INSTALL_VIDEO_FILTERS` | `bool` | `True` | Automatically install/update model-specific OpenRouter Video Generation companion filters in Open WebUI. |
| `AUTO_ATTACH_VIDEO_FILTERS` | `bool` | `True` | Automatically attach each model-specific video filter to its matching OpenRouter video model. |
| `AUTO_DEFAULT_VIDEO_FILTERS` | `bool` | `True` | Always keep the per-model video filter enabled by default on its video model. Re-asserted on every catalog metadata sync. Models that mandate a per-model parameter (`personGeneration` for Veo, `quality` for Sora) cannot be driven without the filter; parameter-free models still generate. Setting this to `False` stops new models being defaulted but does not detach filters already marked default — clear those on the model itself. |
| `ENABLE_OPENROUTER_FUSION` | `bool` | `True` | Master switch for OpenRouter Fusion (multi-model judge panel) support. When enabled the pipe installs the **OpenRouter Fusion** filter, auto-wires it to the fusion models only, and guarantees deliberation on every fusion-model chat request by appending the activating `{"id": "fusion"}` plugins entry AND `tool_choice: "required"` (task/title requests are never injected or forced; a caller-supplied `tool_choice` or `enabled:false` entry wins). Setting to `False` deactivates the installed filter on the next `pipes()` call and stops the injection and forcing — Fusion is then fully off. See [openrouter_fusion.md](openrouter_fusion.md). |
| `AUTO_INSTALL_FUSION_FILTER` | `bool` | `True` | Automatically install/update the OpenRouter Fusion filter function in Open WebUI. |
| `AUTO_ATTACH_FUSION_FILTER` | `bool` | `True` | Automatically attach the OpenRouter Fusion filter to the `openrouter/fusion` model **only** (never to other models). Other models can use Fusion only if an admin manually attaches the filter and turns on its `ALLOW_ON_NON_FUSION_MODELS` **filter** valve (default `False`; while off the filter no-ops on any non-fusion model — injecting no plugin and no forcing). The filter's own admin valves (`ALLOW_ON_NON_FUSION_MODELS`, `priority`) are filter `Valves`, not pipe valves — documented in [openrouter_fusion.md](openrouter_fusion.md#filter-admin-valves-on-the-filter-itself). |
| `AUTO_DEFAULT_FUSION_FILTER` | `bool` | `True` | Mark the OpenRouter Fusion filter as a Default Filter on the `openrouter/fusion` model (pre-enabled per chat). Does **not** force Fusion to run — the per-user *Always run Fusion* toggle defaults off. Re-asserted on every catalog metadata sync. |
| `FUSION_BACKEND` | `Literal["openrouter", "internal"]` | `"internal"` | Which engine runs deliberation for the dedicated fusion models. `openrouter`: the hosted Fusion service runs the panel and judge server-side. `internal`: the pipe runs the same panel → judge → synthesis flow itself as ordinary pipe model calls — panel members inherit the chatting user's full Open WebUI tool surface (knowledge bases, tool servers, pipe server tools), every per-model dial applies (ZDR, reasoning effort, cost attribution), and a failed member degrades gracefully instead of killing the run. The live panel UI and per-chat controls are identical on both. See [openrouter_fusion.md](openrouter_fusion.md#engine-backends). |
| `FUSION_PANEL_SYSTEM_PROMPT` | `str` | tuned multi-model default | System prompt every panel member receives on the internal fusion engine. Enforces independent, committed, citation-backed answers and forbids revealing the deliberation machinery. Edit to reshape panel behaviour; applies from the next fusion chat. |
| `FUSION_JUDGE_SYSTEM_PROMPT` | `str` | tuned multi-model default | System prompt for the internal engine's judge (temperature 0). **Caution:** the judge must emit one strict JSON object with exactly the keys `consensus`, `contradictions`, `partial_coverage`, `unique_insights`, `blind_spots` — the live Analysis panel and the synthesis stage depend on that contract. A judge that stops producing valid JSON gets one repair attempt, then the run falls back to no-analysis mode. |
| `FUSION_SYNTHESIS_SYSTEM_PROMPT` | `str` | tuned multi-model default | System prompt for the internal engine's final stage — the model that receives the panel drafts plus the judge's analysis and writes the user-facing answer. Enforces composing from the strongest material (never averaging) and secrecy about the machinery. |
| `VIDEO_INITIAL_POLL_DELAY_SECONDS` | `float` | `5.0` | Initial delay before polling a newly submitted OpenRouter video generation job. |
| `VIDEO_POLL_INTERVAL_SECONDS` | `float` | `5.0` | Base polling interval for video jobs. |
| `VIDEO_POLL_BACKOFF_FACTOR` | `float` | `1.2` | Backoff multiplier after each non-terminal poll. |
| `VIDEO_POLL_INTERVAL_MAX_SECONDS` | `float` | `20.0` | Maximum interval between video status polls. |
| `VIDEO_MAX_POLL_TIME_SECONDS` | `int` | `600` | Maximum wall-clock polling time before a visible timeout failure is persisted. |
| `VIDEO_STATUS_POLL_MAX_ERRORS` | `int` | `5` | Consecutive status-poll failures before the lifecycle fails visibly. |
| `REMOTE_VIDEO_MAX_SIZE_MB` | `int` | `500` | Maximum generated video download size. The download is streamed to a bounded temp file and aborted during streaming if this cap is exceeded. |
| `VIDEO_DOWNLOAD_CHUNK_SIZE` | `int` | `1048576` | Chunk size used while streaming generated video content to a temp file. |
| `MAX_CONCURRENT_VIDEO_GENS` | `int` | `2` | Maximum active video lifecycles per pipe process. |
| `MAX_CONCURRENT_VIDEO_GENS_PER_USER` | `int` | `2` | Maximum active video lifecycles per user per pipe process. |
| `VIDEO_FRAME_IMAGE_MAX_BYTES` | `int` | `12582912` | Maximum decoded size for one image frame passed to video generation. |
| `VIDEO_FRAME_TOTAL_MAX_BYTES` | `int` | `52428800` | Maximum combined decoded size for all image frames in one video request. |
| `VIDEO_FRAME_IMAGE_MIME_ALLOWLIST` | `str` | `image/jpeg,image/png,image/webp` | Comma-separated MIME allowlist for video frame images. |
| `VIDEO_OUTPUT_MIME_ALLOWLIST` | `str` | `video/mp4,video/webm` | Comma-separated MIME allowlist for generated video downloads after content sniffing. |
Notes:
- Generated videos are not buffered as full Python `bytes`; they go through the canonical helpers (`MultimodalHandler._download_remote_url_streaming` → `OwuiFileGateway.upload_to_owui_storage_from_path` → `OwuiFileGateway.try_link_file_to_chat`), which apply the SSRF gate, exponential-backoff retry, size cap, MIME sniff, OWUI `upload_file_handler` insert, and chat-file link in one shot. The same helpers are reused by image generation.
- The adapter persists a hidden `videojob` marker into the assistant message immediately after `submit()` returns a job_id, by emitting an OWUI socket `'message'` event (which routes through `Chats.upsert_message_to_chat_by_id_and_message_id`). A later request for the same message resumes polling that job instead of submitting a second job.
- Chats with no stored row (`chat_id` beginning with `temporary:`, `local:` or `channel:`) cannot persist markers or final assistant content to Open WebUI chat storage. The on-submit `'message'` emit is skipped for them. They remain in-process only.
- `Pipe.close()` cancels in-process video lifecycles. OpenRouter has no cancel endpoint here; the on-submit `videojob` marker is what allows the next user request for that message to resume polling rather than submit a duplicate job.
- Video filters are generated per model from OpenRouter video metadata. Unsupported controls are not exposed: for example Sora text-only models do not show frame controls, and a model with no negative-prompt passthrough does not show that control.
- Seed and audio are offered unless the catalog says outright the model has neither. A published `false` hides the control; a published `null` — which the catalog uses for some models — shows it, and leaving it alone sends nothing so the model's own default applies. On the recorded catalog this is `minimax/hailuo-2.3` for seed and the two `alibaba/happyhorse` entries for audio.
- Attachments the frame controls do not claim — extra images, a clip, a sound file — are sent as `input_references` rather than discarded. Each is checked against the same limits as a frame, but a reference that fails one is left out with a warning notice in the chat naming it and the reason, and the render proceeds; a frame that fails one fails the whole request. References draw on their own combined-size budget, separate from the frames'.
- User-supplied passthrough URLs (`VIDEO_AUDIO_URL`, `VIDEO_LAST_IMAGE_URL`, `VIDEO_REFERENCE_VIDEO_URL`, and JSON-array references) go through the same safety gate as every other URL the pipe fetches before being forwarded to OpenRouter — blocks `file://`, private IPs, loopback, and unallowlisted `http://`.
- A request needs a prompt **or** something to generate from: a turn with no words but an attached image, reference or clip is submitted rather than refused.
- A job that returns several clips has all of them delivered, each stored as its own file.
- An explicit `VIDEO_SIZE` wins over a `VIDEO_RESOLUTION` tier or a `VIDEO_ASPECT_RATIO` that contradicts it: the losing one is dropped with a warning notice in the chat naming it, rather than sent and rejected upstream.

See: [OpenRouter Video Generation](openrouter_video_generation.md).

#### Companion filter user valves (per-user, per-model)

Each video model gets its OWN filter function in Open WebUI. The `UserValves` rendered into each filter source vary per model, gated on the model's catalog metadata (`supported_durations`, `allowed_passthrough_parameters`, top-level `seed` / `generate_audio` flags, etc.). The full union of valves across variants is below; the [OpenRouter Video Generation](openrouter_video_generation.md#filter-uservalve-identifiers-master-reference) doc has the per-model exposure matrix.

**Core UserValves** — gated on `supported_*` catalog fields:

| Valve | Type | Default | Maps to | Gate |
| --- | --- | --- | --- | --- |
| `VIDEO_PROVIDER_OPTIONS_JSON` | `str` | `""` | `provider.options` raw JSON keyed by provider slug | always |
| `VIDEO_DURATION` | `Literal[0, …]` | `0` | top-level `duration` (seconds) | `supported_durations` non-empty |
| `VIDEO_ASPECT_RATIO` | `Literal["", …]` | `""` | top-level `aspect_ratio` | `supported_aspect_ratios` non-empty |
| `VIDEO_RESOLUTION` | `Literal["", …]` | `""` | top-level `resolution` | `supported_resolutions` non-empty |
| `VIDEO_SIZE` | `Literal["", …]` | `""` | top-level `size` (`WIDTHxHEIGHT`) | `supported_sizes` non-empty |
| `VIDEO_FRAME_MODE` | `Literal["auto", "none", "first_only"(, "first_last")]` | `"auto"` | shapes `frame_images[]` from chat-attached images | `supported_frame_images` non-empty |
| `VIDEO_NEGATIVE_PROMPT` | `str` | `""` | top-level `negative_prompt` (or `negativePrompt` on Veo) | `"negative_prompt"` or `"negativePrompt"` in `allowed_passthrough_parameters` |
| `VIDEO_GENERATE_AUDIO` | `Literal["model_default", "on", "off"]` | `"model_default"` | top-level `generate_audio` (boolean) | catalog top-level `generate_audio` present and not `false` |
| `VIDEO_SEED` | `int` (`ge=0`) | `0` | top-level `seed` | catalog top-level `seed` present and not `false` |
| `VIDEO_AUDIO_URL` | `str` | `""` | passthrough `audio` (URL) | `"audio"` allowed |
| `VIDEO_REFERENCE_VIDEO_URL` | `str` | `""` | passthrough `video` | `"video"` allowed |
| `VIDEO_REFERENCE_VIDEOS_JSON` | `str` (JSON array) | `""` | passthrough `videos` | `"videos"` allowed |
| `VIDEO_REFERENCE_IMAGES_JSON` | `str` (JSON array) | `""` | passthrough `images` | `"images"` allowed |
| `VIDEO_LAST_IMAGE_URL` | `str` | `""` | passthrough `last_image` | `"last_image"` allowed |

**Typed passthrough UserValves** — added in the verification + upgrade pass so users no longer need raw JSON for known per-model knobs. Each renders only when its corresponding string is in the model's `allowed_passthrough_parameters`:

| Valve | Type | Default | Maps to | Exposed on |
| --- | --- | --- | --- | --- |
| `VIDEO_PERSON_GENERATION` | `Literal["", "allow_all", "allow_adult", "dont_allow"]` | `""` | passthrough `personGeneration` | Veo trio |
| `VIDEO_CONDITIONING_SCALE` | `float` (`ge=0.0`, `le=1.0`) | `0.0` | passthrough `conditioningScale` | Veo trio |
| `VIDEO_CFG_SCALE` | `float` (`ge=0.0`, `le=1.0`) | `0.0` | passthrough `cfg_scale` | Kling v3.0 (Pro, Standard) |
| `VIDEO_ENHANCE_PROMPT` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `enhancePrompt` (boolean) | Veo trio |
| `VIDEO_PROMPT_OPTIMIZER` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `prompt_optimizer` (boolean) | Hailuo |
| `VIDEO_FAST_PRETREATMENT` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `fast_pretreatment` (boolean) | Hailuo |
| `VIDEO_PROMPT_EXTEND` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `prompt_extend` (boolean) | Wan 2.7 |
| `VIDEO_RATIO` | `str` | `""` | passthrough `ratio` | Wan 2.7 |
| `VIDEO_ENABLE_PROMPT_EXPANSION` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `enable_prompt_expansion` (boolean) | Wan 2.6 |
| `VIDEO_SHOT_TYPE` | `str` | `""` | passthrough `shot_type` | Wan 2.6 |
| `VIDEO_WATERMARK` | `Literal["model_default", "on", "off"]` | `"model_default"` | passthrough `watermark` (boolean) | Seedance trio |
| `VIDEO_REQ_KEY` | `str` | `""` | passthrough `req_key` | Seedance trio |
| `VIDEO_QUALITY` | `Literal["", "standard", "hd"]` | `""` | passthrough `quality` | Sora 2 Pro |
| `VIDEO_STYLE` | `str` | `""` | passthrough `style` | Sora 2 Pro |

**Skip-when-default sentinel**: a valve set to its default value (`""`, `0`, `0.0`, or `"model_default"`) is **NOT** included in the request body. The upstream provider's own default applies. 3-state Literals translate `"on"` → `True`, `"off"` → `False`, `"model_default"` → omitted.

**Routing**: the fields the model's catalog entry publishes as supported (`duration`, `aspect_ratio`, `resolution`, `size`, `seed`, `generate_audio`) land at the request body root. Every other passthrough field, `negative_prompt` included, goes under `provider.options.<slug>` keyed by the provider slug the catalog publishes — written at the root it is accepted and ignored. `VIDEO_PROVIDER_OPTIONS_JSON` writes to the same place; it is the escape hatch for untyped fields, not a different destination. The placement is recorded in `tests/fixtures/openrouter_video_passthrough_probe.json`: the same knob sent three ways, and only the `provider.options.<slug>` form changed the produced video.

### OpenRouter video intent classifier

A task-model classifier that reads recent chat turns and attachments before an OpenRouter video request to decide what the new video should reference (reuse a prior frame, adopt an attached image, or start fresh), optionally asking one clarifying question. Every failure degrades open — the paid video still generates from the latest message. See [openrouter_video_intent_classifier.md](openrouter_video_intent_classifier.md).

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `VIDEO_INTENT_ENABLED` | `bool` | `True` | Master switch. When off, video requests bypass the classifier — only the latest user message is sent (no cross-turn context, clarifying questions, or frame reuse). |
| `VIDEO_INTENT_TASK_MODEL_MODE` | `internal` / `external` | `external` | Which Open WebUI global Task Model runs the classifier: `external`=`TASK_MODEL_EXTERNAL`, `internal`=`TASK_MODEL`. |
| `VIDEO_INTENT_TASK_MODEL_FALLBACK` | `none` / `other_task_model` | `other_task_model` | Second-attempt strategy when the chosen Task Model fails: `none`=stop; `other_task_model`=try the other global Task Model (deduped if identical/unset). |
| `VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT` | `bool` | `True` | Skip the classifier on a chat's first turn with no attachments (nothing to reference), saving a wasted Task Model call. |
| `VIDEO_INTENT_MAX_CLARIFICATIONS` | `int` | `1` | Per-chat cap on consecutive clarifying questions before the classifier proceeds on its best guess. `0` disables the loop. User-overridable per video model. |
| `VIDEO_INTENT_FRAME_EXTRACTION_INDEX` | `first` / `last` | `last` | Overshoot fallback only: which frame to reuse when a requested timestamp runs past a prior clip's end. User-overridable per video model. |
| `VIDEO_INTENT_TIMEOUT_S` | `int` | `8` | Per-attempt timeout (seconds) for the classifier's Task Model call; on timeout/failure the pipe degrades open (latest message only) and never blocks the paid render. |
| `VIDEO_INTENT_CONFIRM_MODE` | `always` / `on_reference` / `low_confidence` / `never` | `on_reference` | When to show the disclosure footer. `on_reference` = a prior video's frame is reused OR multiple frames combined (a lone attachment does not trigger it). User-overridable per video model. |
| `VIDEO_INTENT_MAX_CALLS_PER_CHAT` | `int` | `0` | Cost guard: max classifier calls per chat (`0`=unlimited). Per-worker in-memory counter; over the cap, classification is skipped and the render still proceeds. |
| `VIDEO_INTENT_MAX_CALLS_PER_USER_DAY` | `int` | `0` | Cost guard: max classifier calls per user per UTC day (`0`=unlimited). Per-worker in-memory counter, resets at UTC midnight/restart. |
| `VIDEO_INTENT_LOG_DECISIONS` | `bool` | `False` | When True, log the per-turn classification summary (intent, confidence, language, frame counts, latency, fallback/failure flags, hashed chat id) at INFO instead of DEBUG. Always written; excludes the verbatim prompt and the model's free-text reason. |

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
| `DIRECT_PDF_PARSER` | `Literal["Native","PDF Text","Mistral OCR"]` | `"Native"` | Selects the OpenRouter PDF parsing engine for PDF uploads (requires `DIRECT_FILES` enabled). |

### Provider routing filters

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `ADMIN_PROVIDER_ROUTING_MODELS` | `str` | `""` | Comma-separated list of model slugs (e.g., `openai/gpt-4o, anthropic/claude-3.5-sonnet`) for which to generate admin-only provider routing filters. These filters enforce provider preferences (order, fallbacks, ZDR, etc.) that users cannot override or disable. Which of those a listed model actually offers depends on its request format — see the note below. Leave empty to disable. |
| `USER_PROVIDER_ROUTING_MODELS` | `str` | `""` | Comma-separated list of model slugs for which to generate user-configurable provider routing filters. Users can toggle these filters per-chat and configure their own provider preferences via UserValves. Which preferences a listed model offers depends on its request format — see the note below. Models in both lists get filters with admin defaults and user overrides. |
| `AUTO_DEFAULT_PROVIDER_ROUTING_FILTERS` | `bool` | `True` | Pre-enables attached provider routing filters in new chats (via `defaultFilterIds`), so saved provider preferences apply without users switching the filter on per chat. The filter is a no-op until preferences are set. Disable to make users opt in per chat. A video model has no such filter to pre-enable. |

Notes:
- Provider routing filters are generated dynamically from OpenRouter's public per-model endpoints API (`/api/v1/models/{author}/{slug}/endpoints`), which lists every provider serving the model; the frontend catalog is only a degraded single-provider fallback when that fetch fails.
- Each filter exposes an **ORDER dropdown** (human-readable provider names): full priority permutations for up to 4 providers, and a linear "X first" preference per provider beyond that (full permutations would grow factorially).
- Admin-only filters use `toggle=False` (always run, cannot be disabled per-chat).
- User-configurable filters use `toggle=True` (can be toggled on/off per-chat) and start enabled in new chats while `AUTO_DEFAULT_PROVIDER_ROUTING_FILTERS` is on.
- Provider routing is **not applied** to task model requests (title, tags, follow-ups).
- The controls a generated filter draws depend on the model's request format: chat completions carries all of them, the dedicated image format carries only `ORDER`, `ONLY`, `IGNORE`, `SORT`, `SORT_PARTITION` and `ALLOW_FALLBACKS`, and the video format carries none — so no filter is installed for a video model and an existing one is deactivated. See [OpenRouter Provider Routing](openrouter_provider_routing.md).
- Variant-only providers (e.g., Venice serving only `:free` variants) are excluded from base model routing options.

See: [OpenRouter Provider Routing](openrouter_provider_routing.md).

#### Generated filter valves (per model)

Each generated provider routing filter has these valves (admin and/or user depending on visibility):

| Valve | Type | Default | Maps to OpenRouter API |
| --- | --- | --- | --- |
| `ORDER` | `Literal[...]` | `"(no preference)"` | `provider.order` — Provider priority ordering |
| `ALLOW_FALLBACKS` | `bool` | `True` | `provider.allow_fallbacks` — Use backup providers if preferred unavailable |
| `REQUIRE_PARAMETERS` | `bool` | `False` | `provider.require_parameters` — Only use providers supporting all request params |
| `ZDR` | `bool` | `False` | `provider.zdr` — Zero Data Retention enforcement |
| `QUANTIZATION` | `Literal[...]` | `"(no preference)"` | `provider.quantizations` — Filter by quantization level (when available) |
| `MAX_PRICE_IMAGE` | `float` | `0` | `provider.max_price.image` — Max price per image ($/image), 0=no limit |
| `MAX_PRICE_AUDIO` | `float` | `0` | `provider.max_price.audio` — Max price for audio ($/unit), 0=no limit |
| `MAX_PRICE_REQUEST` | `float` | `0` | `provider.max_price.request` — Max price per request ($/request), 0=no limit |

Which of these are drawn depends on the model's request format. A chat model
carries all 18, including `SORT_PARTITION` (whether ranking groups endpoints by
model first) and `ENFORCE_DISTILLABLE_TEXT`, and `SORT` accepts `exacto`
alongside `price`, `throughput` and `latency`. A model that returns pictures and
no text carries only `ORDER`, `ONLY`, `IGNORE`, `SORT`, `SORT_PARTITION` and
`ALLOW_FALLBACKS`. A video model carries none, so listing one creates nothing
and a filter left from an earlier release is deactivated. The full per-control
reference is in [OpenRouter Provider Routing](openrouter_provider_routing.md).

### Reporting, UI behavior, and request identifiers

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `USE_MODEL_MAX_OUTPUT_TOKENS` | `bool` | `False` | When enabled **and the request carries no limit of its own**, fills `max_output_tokens` from the provider-advertised `max_completion_tokens` in the catalog; a model that advertises none is left unchanged. Disabled, the pipe adds no limit of its own and provider defaults apply. The valve controls the pipe's automatic value, not the caller's: a `max_tokens` of 1 or above is forwarded unchanged in either state. OpenRouter documents the parameter as "1 or above" and Open WebUI's slider reaches -2, so a value below 1 is sent as no cap — and this valve's automatic ceiling then applies if it is on. |
| `SHOW_FINAL_USAGE_STATUS` | `bool` | `True` | Includes timing/cost/tokens in the final status message. |
| `FINAL_USAGE_STATUS_STYLE` | `Literal["text","icons"]` | `text` | Choose text labels or icons for the final usage status line. |
| `USAGE_STATUS_ICON_SET` | `str` | `⧗,$,⇅,▲,▼,↺,▽` | CSV icon set for final usage status fields (time,cost,total,input,output,cached,reasoning). Used only when `FINAL_USAGE_STATUS_STYLE="icons"`. |

Notes:
- `FINAL_USAGE_STATUS_STYLE="text"` uses labels like “Time”, “Cost”, and “Total tokens”.
- `FINAL_USAGE_STATUS_STYLE="icons"` swaps those labels for the icon set. You can also pass **words** as the CSV entries if you want custom labels (e.g., `Time,Cost,Total,Input,Output,Cached,Reasoning`).
| `SEND_END_USER_ID` | `bool` | `False` | When enabled, sends the OpenRouter top-level `user` field (value chosen by `END_USER_ID_SOURCE`), and always adds `metadata.user_id` with the Open WebUI user GUID. See [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md). |
| `END_USER_ID_SOURCE` | `Literal["id", "email", "name"]` | `"id"` | What the `user` field carries when `SEND_END_USER_ID` is on: the OWUI GUID, the user's email, or their display name (email/name fall back to the GUID when empty). `email`/`name` send PII to OpenRouter — enable deliberately. |
| `SEND_SESSION_ID` | `bool` | `False` | When enabled, adds `metadata.session_id` using Open WebUI `__metadata__[\"session_id\"]` (metadata only). |
| `SEND_CACHE_SESSION_ID` | `bool` | `True` | Sends `session_id` = `HMAC-SHA256(WEBUI_SECRET_KEY, chat_id)` to pin each conversation to one provider for prompt-cache warmth. Skipped if `WEBUI_SECRET_KEY` is unset. See [Prompt-cache session affinity](openrouter_integrations_and_telemetry.md#216-prompt-cache-session-affinity-session_id). |
| `SEND_CHAT_ID` | `bool` | `False` | When enabled, adds `metadata.chat_id` using Open WebUI `__metadata__[\"chat_id\"]`. |
| `SEND_MESSAGE_ID` | `bool` | `False` | When enabled, adds `metadata.message_id` using Open WebUI `__metadata__[\"message_id\"]`. |
| `ENABLE_PLUGIN_SYSTEM` | `bool` | `False` | Master switch for the plugin system. When `False`, all plugin hooks are skipped (zero overhead); when `True`, registered plugins load and their hooks dispatch. Takes effect immediately without restart. See [Plugin System](plugin_system.md). |

### Session log storage

| Valve | Type | Default (verified) | Purpose / notes |
| --- | --- | --- | --- |
| `SESSION_LOG_STORE_ENABLED` | `bool` | `False` | When True, persist SessionLogger output to encrypted zip files on disk, assembled per message turn (`chat_id`, `message_id`). Persistence is skipped when required IDs are missing (`user_id`, `session_id`, `chat_id`, `message_id`). |
| `SESSION_LOG_DIR` | `str` | `session_logs` | Base directory for encrypted session log archives. |
| `SESSION_LOG_ZIP_PASSWORD` | `EncryptedStr` | `(empty)` | Password used to encrypt session log zip files (pyzipper AES). |
| `SESSION_LOG_RETENTION_DAYS` | `int` | `90` | Retention window (days) for stored session log archives. |
| `SESSION_LOG_CLEANUP_INTERVAL_SECONDS` | `int` | `3600` | How often (seconds) to run the session log cleanup loop when storage is enabled. |
| `SESSION_LOG_ZIP_COMPRESSION` | `Literal[\"stored\", \"deflated\", \"bzip2\", \"lzma\"]` | `lzma` | Zip compression algorithm for session log archives. |
| `SESSION_LOG_ZIP_COMPRESSLEVEL` | `Optional[int]` | `null` | Compression level (0–9) for deflated/bzip2 compression. Ignored for stored/lzma. |
| `SESSION_LOG_MAX_LINES` | `int` | `20000` | Maximum number of in-memory SessionLogger records retained per request (older entries are dropped). |
| `SESSION_LOG_FORMAT` | `Literal[\"jsonl\", \"text\", \"both\"]` | `jsonl` | Archive log file format. `logs.jsonl` is always written; `jsonl` writes only it, while `text` and `both` additionally write `logs.txt` (so `text` and `both` yield the same file set). |
| `SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS` | `int` | `30` | How often each process scans the DB for completed/stale turns to assemble into zip archives. |
| `SESSION_LOG_ASSEMBLER_JITTER_SECONDS` | `int` | `10` | Per-process jitter added to the assembler loop to avoid multi-worker lockstep. |
| `SESSION_LOG_ASSEMBLER_BATCH_SIZE` | `int` | `25` | Max turns processed per assembler tick. |
| `SESSION_LOG_STALE_FINALIZE_SECONDS` | `int` | `43200` | If no terminal segment arrives for a turn, assemble an incomplete archive after this timeout. |
| `SESSION_LOG_LOCK_STALE_SECONDS` | `int` | `1800` | DB lock row stale timeout (multi-worker safety). |
| `ENABLE_TIMING_LOG` | `bool` | `False` | When True, capture function entrance/exit timing data. Writes to `TIMING_LOG_FILE` directly (not session archives). See [Session Log Storage](session_log_storage.md#timing-instrumentation). |
| `TIMING_LOG_FILE` | `str` | `logs/timing.jsonl` | File path for timing log output when `ENABLE_TIMING_LOG` is True. Parent directories are created automatically. |

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
| `PAYLOAD_TOO_LARGE_TEMPLATE` | `str` | `built-in default` | Markdown template for OpenRouter HTTP 413 responses, when the request payload exceeds the size OpenRouter accepts. Supports `{error_id}`, `{timestamp}`, `{openrouter_code}`, `{openrouter_message}`, `{model_identifier}` and `{support_email}`, plus Handlebars-style `{{#if name}}...{{/if}}` blocks that render only when the value is present. |
| `NETWORK_TIMEOUT_TEMPLATE` | `str` | `built-in default` | Markdown template for network timeouts. |
| `CONNECTION_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for connection failures. |
| `SERVICE_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for OpenRouter 5xx errors. |
| `INTERNAL_ERROR_TEMPLATE` | `str` | `built-in default` | Markdown template for unexpected internal errors. |
| `MODEL_RESTRICTED_TEMPLATE` | `str` | `built-in default` | Markdown template emitted when the requested model is blocked by `MODEL_ID` and/or model filter valves. |
| `STREAM_INTERRUPTED_TEMPLATE` | `str` | `built-in default` | Markdown appended when a streamed reply ends without a completion event; partial content is preserved. |

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
| `ENABLE_ANTHROPIC_INTERLEAVED_THINKING` | `bool` | `True` | When enabled and the selected model is Anthropic (`anthropic/...` or a `~anthropic/...` router alias), send `x-anthropic-beta: interleaved-thinking-2025-05-14` to opt into Claude interleaved thinking streams. |
| `REASONING_EFFORT` | `Literal[\"none\", \"minimal\", \"low\", \"medium\", \"high\", \"xhigh\"]` | `medium` | Choose how much thinking the AI should do before answering (higher depth is slower but more thorough). |
| `REASONING_SUMMARY_MODE` | `Literal[\"auto\", \"concise\", \"detailed\", \"disabled\"]` | `auto` | Choose how detailed the reasoning summary should be. |
| `PERSIST_REASONING_TOKENS` | `Literal[\"disabled\", \"next_reply\", \"conversation\"]` | `next_reply` | User-level reasoning retention preference. |
| `PERSIST_TOOL_RESULTS` | `bool` | `False` | Let the AI reuse outputs from tools later in the conversation. |
| `REQUEST_ZDR` | `bool` | `False` | Request ZDR routing for this chat. |

## Plugin-exported valves (pipe_dashboard)

These appear in the merged Valves UI when `ENABLE_PLUGIN_SYSTEM` is on; full context in `docs/plugins_pipe_dashboard.md`.

| Valve | Type | Default | Description |
|-------|------|---------|-------------|
| `PIPE_DASHBOARD_ENABLE` | `bool` | `False` | Show/hide the Pipe Dashboard virtual model in the model selector. |
| `PIPE_DASHBOARD_USAGE_COLLECT` | `bool` | `False` | Persist one usage record per completed request (user, model, tokens, tools, cost) to a dedicated `dashboard_` table powering the dashboard's Usage tab. Read live at record time. |
| `PIPE_DASHBOARD_USAGE_RETENTION_DAYS` | `int` (1–365) | `30` | Retention for collected usage records; a jittered, lock-guarded purge task deletes older rows. Read live inside the purge loop. |
| `PIPE_DASHBOARD_UPDATE_ENABLE` | `bool` | `True` | Enables the dashboard's Update tab and its actions (check, apply, restore, delete snapshot). Off: the tab reports disabled and every update action — including auto-update — fails closed server-side. |
| `PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP` | `int` (1–10) | `3` | Previous-version snapshots retained as dedicated OWUI Files records (metadata on the file itself — immune to function-editor saves); oldest pruned first (record, then blob). Identical-content retries reuse the matching snapshot instead of consuming a slot. |
| `PIPE_DASHBOARD_UPDATE_REPO` | `str` | `rbb-dev/Open-WebUI-OpenRouter-pipe` | GitHub `owner/repo` the updater tracks for tagged releases (fork support). All URLs are built server-side from this valve; the browser never supplies a repo. Malformed → `bad repo` state; missing/private → `repo not found`. |
| `PIPE_DASHBOARD_UPDATE_AUTO` | `bool` | `False` | Auto-apply eligible releases. Multi-worker: workers elect ONE update leader via OWUI's Redis lock — only the leader checks GitHub (~6h); followers probe the lease hourly and take over if it expires. Single-worker/no-redis: that worker checks directly. Backs off on rate limits, re-reads valves from the DB each cycle, runs headless with the dashboard model switched off as long as the Update tab is enabled. |
| `PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS` | `int` (0–720) | `168` | Quarantine: a release must be at least this old before auto-update applies it, so a bad release yanked within the window never reaches auto-updaters. `0` = immediate. Manual updates ignore it. |
