# Tools, plugins, and integrations

**Scope:** How tool schemas are built, how `function_call` items are executed, and how optional integrations (web search plugin, MCP tool specs) are attached.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [🔒 Security](security_and_encryption.md)

This pipe supports OpenRouter Responses API tool calling with a controlled execution pipeline and operator-tunable concurrency limits. It also supports optional integrations:

- OpenRouter web-search (as a `plugins` entry, not a function tool).
- Remote MCP server definitions (as `tools` entries of type `mcp`).

---

## Tool schema assembly (`build_tools`)

Tool *schemas* are assembled by `build_tools(...)` and attached to the outgoing Responses request as `tools`.

### Preconditions

- Tools are only attached when the selected model is recognized as supporting `function_calling`.
- If the model does not support function calling, the pipe sends no tool schemas.

### Tool sources (in order)

1. **Open WebUI tool registry** (`__tools__` dict)
   - Converted to OpenAI tool specs (`{"type":"function","name",...}`) via `ResponsesBody.transform_owui_tools(...)`.
   - When `ENABLE_STRICT_TOOL_CALLING=true`, each tool schema is strictified:
     - Object nodes get `additionalProperties: false`.
     - All declared properties are marked required; properties that were not explicitly required become nullable (their type gains `"null"`).
     - Missing property `type` values are inferred defensively (object/array) so schemas remain valid.
     - A small LRU cache (size 128) avoids repeated strictification work for identical schemas.

2. **Remote MCP servers** (`REMOTE_MCP_SERVERS_JSON`)
   - Parsed from a JSON string (a single object or an array of objects).
   - Each valid entry produces a tool spec `{ "type": "mcp", ... }`.
   - Validation and safety notes:
     - Payloads over 1MB are ignored.
     - `server_label` and `server_url` are required.
     - `server_url` schemes are restricted to `http`, `https`, `ws`, `wss` (must include a host).
     - Only the official MCP keys are forwarded: `server_label`, `server_url`, `require_approval`, `allowed_tools`, `headers`.

3. **Extra tools** (`extra_tools`)
   - A caller-provided list of already OpenAI-format tool specs is appended as-is (non-dict entries are ignored).

### Deduplication

After assembly, tools are deduplicated by `(type, name)` identity. If duplicates exist, the **later** entry wins.

---

## Tool execution lifecycle (Responses API loop)

Tool execution happens in the request loop that follows each Responses API call:

1. The pipe calls the provider (streaming mode for normal chats).
2. When a `response.completed` event arrives, the pipe inspects the response `output` list.
3. Any `output` items with `type == "function_call"` are treated as tool calls to execute locally.
4. The pipe executes the tools and converts each result into `function_call_output` items.
5. The `function_call` items (normalized) and their outputs are appended to the next request’s `input[]`, and the loop continues until either:
   - no more `function_call` items are returned, or
   - `MAX_FUNCTION_CALL_LOOPS` is reached.

Notes:

- If a tool name is missing or not present in the tool registry, the pipe returns a structured `function_call_output` indicating the failure.
- The pipe does not “stream” tool outputs mid-request. Tools are executed between Responses calls.

---

## Concurrency, batching, and timeouts (per request)

Tools are executed via a per-request worker pool backed by a bounded queue:

- Queue size: 50 tool calls per request (bounded).
- Worker count: `MAX_PARALLEL_TOOLS_PER_REQUEST`.
- Per-request semaphore: limits concurrent tool executions per request.
- Global semaphore: `MAX_PARALLEL_TOOLS_GLOBAL` limits tool executions across all requests.

Batching behavior:

- Tool calls may be batched when they share the same tool name and do not declare dependency/ordering blockers in arguments.
- If tool arguments include any of: `depends_on`, `_depends_on`, `sequential`, `no_batch`, the call is treated as non-batchable.
- Batching does not require identical arguments; it is a concurrency optimization, not a deduplication mechanism.

Timeouts and retries:

- Each tool call is run with a per-call timeout (`TOOL_TIMEOUT_SECONDS`).
- Tool calls are retried up to 2 attempts (per call) when they raise exceptions.
- Tool batches are guarded by a batch timeout (derived from `TOOL_BATCH_TIMEOUT_SECONDS` and the per-call timeout).
- If the tool queue stays idle for `TOOL_IDLE_TIMEOUT_SECONDS`, the worker loop cancels pending work and surfaces an error.

---

## Breakers (stability controls)

The pipe applies a shared breaker window (`BREAKER_MAX_FAILURES` within `BREAKER_WINDOW_SECONDS`) across different subsystems:

- **Per-user request breaker:** prevents repeated failing requests from thrashing the system.
- **Per-user, per-tool-type breaker:** temporarily disables executing tool calls of a given *type* (for example, `function`) for a user after repeated tool failures.
- **Per-user DB breaker:** can temporarily suppress persistence-related work after repeated database failures.

When a tool breaker is open, tool calls are skipped and a status message is emitted to the UI (best effort).

---

## OpenRouter web-search plugin

The web-search integration is attached as a `plugins` entry (not as a `tools` function):

- If the selected model supports `web_search_tool` and either:
  - `ENABLE_WEB_SEARCH_TOOL=true`, or
  - the model’s catalog-derived features indicate web-search capability,
  then the pipe appends `{ "id": "web" }` to `plugins`.
- If `WEB_SEARCH_MAX_RESULTS` is set, it is included as `max_results`.
- If reasoning effort is `minimal`, the pipe skips adding the web-search plugin.

---

## Notes on MCP tool specs

`REMOTE_MCP_SERVERS_JSON` attaches MCP tool server definitions to the request. Calls involving MCP may appear in provider output as `mcp_call` items.

For persistence behavior and replay rules of tool artifacts, see:

- [Persistence, Encryption & Storage](persistence_encryption_and_storage.md)
- [History Reconstruction & Context Replay](history_reconstruction_and_context.md)
