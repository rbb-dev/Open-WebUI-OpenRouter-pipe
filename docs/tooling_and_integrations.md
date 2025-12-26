# Tools, plugins, and integrations

**Scope:** How tool schemas are built, how `function_call` items are executed, and how Open WebUI tool sources (registry + Direct Tool Servers) are attached.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [🔒 Security](security_and_encryption.md)

This pipe supports OpenRouter Responses API tool calling with a controlled execution pipeline and operator-tunable concurrency limits. Tool sources and integrations:

- Open WebUI tool registry tools (server-side Python tools).
- Open WebUI **Direct Tool Servers** (client-side OpenAPI tools executed in the browser via Socket.IO).
- OpenRouter web-search (as a `plugins` entry, not a function tool).

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

2. **Open WebUI Direct Tool Servers** (`__metadata__["tool_servers"]`)
   - These are user-configured OpenAPI tool servers that Open WebUI executes client-side.
   - Open WebUI includes the selected servers in the request body as `tool_servers`; for pipes this arrives under `__metadata__["tool_servers"]`.
   - This pipe:
     - advertises the tools to the model using OpenAPI `operationId` values as tool names (**no namespacing**, collisions overwrite; OWUI-compatible), and
     - executes tool calls via the Socket.IO bridge (`__event_call__`) by emitting `execute:tool` so the browser performs the request.
   - Direct tools are only advertised when `__event_call__` is available; without an active Socket.IO session there is no safe execution path, so the pipe skips them.

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
- When `MAX_FUNCTION_CALL_LOOPS` is reached while the model is still requesting more tool rounds, the pipe emits a user-facing warning (toast + markdown message) explaining that the response stopped early and suggesting increasing the limit.

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

## Open WebUI Direct Tool Servers

Direct Tool Servers are configured and executed by Open WebUI, but advertised/executed through this pipe:

- Configure servers in **User Settings → External Tools → Manage Tool Servers** (and ensure the server is enabled/toggled).
- Select tool servers for a chat in the tool picker (Open WebUI sends the selected servers in `tool_servers`).
- When the model calls a direct tool, the pipe emits `execute:tool` via `__event_call__` and the browser performs the OpenAPI request.

Failure handling:
- Direct tool execution is wrapped in `try/except`; tool crashes never crash the pipe/session.
- On failure the tool returns an error payload to the model (and the pipe may emit an OWUI notification best-effort).

---

## MCP note (removed)

This pipe no longer implements “remote MCP server connectivity” (previously surfaced as `REMOTE_MCP_SERVERS_JSON`) because it bypasses Open WebUI’s tool server configuration surface and RBAC/permissions model.

If you want MCP tools in Open WebUI, use an MCP→OpenAPI proxy/aggregator (for example **MCPO** or **MetaMCP**) and add the resulting OpenAPI server through Open WebUI’s tool server UI so access control and future tool server changes remain centralized in OWUI.

For persistence behavior and replay rules of tool artifacts, see:

- [Persistence, Encryption & Storage](persistence_encryption_and_storage.md)
- [History Reconstruction & Context Replay](history_reconstruction_and_context.md)
