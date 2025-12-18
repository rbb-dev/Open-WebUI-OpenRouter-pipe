# tools, plugins, and extra integrations

**file:** `docs/tooling_and_integrations.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:308-420, 5790-7100, 8280-8425`

The Responses manifold treats tool calling as a first-class subsystem. This document covers the entire pipeline--how tool schemas are built, how execution queues behave, how breakers protect the system, and how optional integrations (web search, MCP, filter injections) are merged without surprises.

> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## 1. schema assembly (`build_tools`)

1. **Prerequisite** -- The target model must advertise `function_calling` support via `ModelFamily.supports("function_calling", model_id)`. If the capability is missing the pipe skips every other step and sends `tools=[]`.
2. **Open WebUI registry** -- `ResponsesBody.transform_owui_tools(__tools__)` converts each registry entry into an OpenAI-style `{type:"function", name, description, parameters}` dict. When `ENABLE_STRICT_TOOL_CALLING=True`, `_strictify_schema` rewrites every object schema to:
   * Set `additionalProperties=False` on every object node (root + nested).
   * Require all declared properties but make optional ones nullable (`{"type": ["string", "null"]}`).
   * **Auto-infer missing types** -- Properties without a `type` key are automatically fixed: empty schemas `{}` become `{"type": "object"}`, schemas with `properties` but no type get `{"type": "object"}`, schemas with `items` but no type get `{"type": "array"}`. This defensive inference ensures malformed schemas don't cause OpenAI validation errors.
   * Cache transformed schemas (`_STRICT_SCHEMA_CACHE_SIZE=128`) so repeat calls are cheap.
3. **MCP servers** -- `REMOTE_MCP_SERVERS_JSON` accepts either one JSON object or a list of objects describing remote servers. `_build_mcp_tools` validates URLs (`http/https/ws/wss` only), whitelists fields (`server_label`, `server_url`, `require_approval`, `allowed_tools`, `headers`), and emits `{"type": "mcp", ...}` entries.
4. **Filter-provided `extra_tools`** -- Upstream filters can attach a list of already-sanitized tool dicts on `completions_body.extra_tools`. The builder simply appends them.
5. **OpenRouter plugins** -- When `ENABLE_WEB_SEARCH_TOOL=True` and the model"s pricing indicates paid web search support, the standard `{ "type": "web_search" }` plugin is appended automatically.
6. **Deduplication** -- `_dedupe_tools` removes duplicates by `(type, name)` identity; later entries win so user overrides beat registry defaults.

---

## 2. execution context (`_ToolExecutionContext`)

Every request that might call a function initializes `_ToolExecutionContext`:

| Field | Meaning |
| --- | --- |
| `queue` | `asyncio.Queue` that buffers `_QueuedToolCall` objects. Bounded so producers back off when tools pile up. |
| `workers` | List of worker tasks (up to `MAX_PARALLEL_TOOLS_PER_REQUEST`). Each worker batches compatible calls (same function, JSON-equal args) up to `TOOL_BATCH_CAP`. |
| `semaphore` | Global semaphore (`MAX_PARALLEL_TOOLS_GLOBAL`) shared across requests, ensuring one chat cannot starve the process. |
| `idle_timeout` | Optional timer (`TOOL_IDLE_TIMEOUT_SECONDS`) that cancels workers if no new tool calls arrive, preventing zombie tasks. |
| `timeout_error` | Remembered string used when workers abort so the user sees why outputs are missing.

Workers call `_execute_tool_batch`, which:

1. Fetches the Open WebUI callable from `__tools__`. Missing entries raise a structured error so the model can recover.
2. Runs the callable either synchronously (in the event loop) or via `asyncio.to_thread` / `asyncio.ensure_future` depending on whether it is async.
3. Applies per-call timeouts: `TOOL_TIMEOUT_SECONDS` for individual tools, `TOOL_BATCH_TIMEOUT_SECONDS` for the whole batch.
4. Persists outputs via `_persist_artifacts` (unless `PERSIST_TOOL_RESULTS=False`). Outputs get ULID markers so `_transform_messages_to_input` can replay them during the next request.

---

## 3. breaker windows & safety nets

| Breaker | Trigger | Recovery |
| --- | --- | --- |
| Per-user breaker (`_breaker_records`) | `BREAKER_MAX_FAILURES` hits inside `BREAKER_WINDOW_SECONDS` (defaults: 5 / 60s). History depth follows `BREAKER_HISTORY_SIZE`. | Automatically clears when enough time passes or successes are recorded. Emits a status message so the user knows the request is throttled. |
| Tool-type breaker (`_tool_breakers[user][tool_name]`) | Same valves but scoped per tool type, so one flaky tool can’t take down the chat. | Status message explains which tool was disabled. The loop continues with other tools. |
| DB breaker (`_db_breakers[user]`) | Same limits applied to persistence failures (DB offline, migrations missing). | Temporarily skips persistence and emits "DB ops skipped due to repeated errors" so operators investigate. |

When a breaker fires, tool events are still surfaced to the UI. Users see which tool failed, why it was skipped, and whether the system will retry later.

---

## 4. multi-step loop with Responses API

1. **Model emits `tool_call` event** via SSE. `_consume_sse` converts it into `_QueuedToolCall` entries and enqueues them.
2. **Workers execute tools** as described above, persisting outputs and building `tool_result` blocks.
3. **Re-injection** -- `_append_tool_outputs_to_input` adds each result to the in-flight `input[]` list so the model receives structured feedback before continuing generation.
4. **Loop control** -- `MAX_FUNCTION_CALL_LOOPS` caps how many full "model → tools → model" cycles a single request may perform. Hitting the cap surfaces a warning and ends the run cleanly.

---

## 5. web search & MCP integrations

* **Web search**: `build_plugins()` (inside `Pipe.pipe()`) inspects the selected model"s pricing metadata. If the provider charges for `web_search` and the valve is enabled, the `{ "id": "web" }` plugin is attached. The pipe also exposes `WEB_SEARCH_MAX_RESULTS` to override the provider"s default result count.
* **MCP servers**: Beyond schema registration, MCP tools participate in the exact same execution loop. The schema describes remote servers; when the model requests one, the tool executor forwards the call using the MCP transport defined in OpenRouter"s Responses API. Because these tools can be expensive, keep `MAX_PARALLEL_TOOLS_PER_REQUEST` conservative.

---

## 6. telemetry & UX

* **Status updates** (`StatusMessages`) announce when tools start, finish, or fail. These messages are delivered through the Open WebUI event stream so the user sees real-time progress.
* **Notifications** (`_emit_notification`) summarize tool results or errors in the right-hand notification tray when enabled.
* **Usage tracking** merges per-tool token/cost estimates into the final completion status so operators can audit expensive workflows.

---

## 7. extending the tool system

1. Add new schema sources by editing `build_tools()`--for example, fetch a workspace-wide tool list from another service, merge it, and document the behavior here.
2. When introducing a new breaker or queue, define matching valves (limits, timeouts) and update `docs/valves_and_configuration_atlas.md` so operators know how to tune them.
3. If you implement new output types (e.g., streaming tool logs), ensure `_persist_artifacts` stores them and `_transform_messages_to_input` knows how to replay or prune them.

The manifold"s tool subsystem is intentionally strict and verbose. Treat it as shared infrastructure for every future tool innovation you add.
