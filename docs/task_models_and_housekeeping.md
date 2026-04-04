# Task models and housekeeping

**Scope:** How the pipe handles Open WebUI “task” requests (`__task__`) such as title/tag/summary generation, and how to operate them safely in production.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md)

Open WebUI can issue two different kinds of requests through `__task__`:

- **Housekeeping tasks** such as generating a chat title, tags, follow-ups, queries, autocomplete, emoji, or image prompts. These should be fast, inexpensive, and low-risk.
- **MOA merged-response synthesis** (`moa_response_generation`), which is user-visible and should behave like a normal chat response.

The pipe treats these categories differently.

---

## How the pipe detects a task request

The pipe treats a request as a task when the special `__task__` argument is present (a dict or task name). When detected:

- Housekeeping tasks log a DEBUG message (`Detected task model: ...`) and use the dedicated task adapter path.
- `moa_response_generation` keeps the normal streaming/tool-execution path even though `__task__` is present.

---

## Housekeeping task behavior (what is different vs normal chat)

### Non-streaming request

Housekeeping tasks are forced to **non-streaming** behavior (`stream=false`) and processed as a single request/response. The pipe then extracts plain text from the Responses payload.

### Output extraction rules

The pipe extracts housekeeping task output text from:

- `output[].type == "message"` items containing `content[].type == "output_text"`, concatenated with newlines.
- Fallback: a top-level `output_text` string (some providers return a collapsed field).

If the provider returns no usable text, the pipe returns a safe placeholder error string to Open WebUI rather than raising an exception.

### Model whitelist bypass (task-mode only)

If a `MODEL_ID` allowlist is configured, normal chat requests enforce it. Housekeeping tasks can **bypass** the allowlist so housekeeping continues even when the selected task model is not in the allowlist.

Important nuance:

- Reasoning overrides described below apply only when the task request targets a model that the pipe considers “owned” (i.e., inside the pipe’s allowed model set when an allowlist is configured). When a task bypasses the allowlist, the pipe does not force task reasoning overrides for that model.
- Model catalog filters (for example `FREE_MODEL_FILTER` and `TOOL_CALLING_FILTER`) control which models are shown to users and enforced for normal chat requests. Housekeeping tasks are not guaranteed to respect those filters; choose task models explicitly if you need “free only” or “tool calling required” behavior for housekeeping.

### Task reasoning effort override (valve-gated)

For housekeeping tasks targeting models the pipe “owns”, the pipe overrides the request’s reasoning configuration using `TASK_MODEL_REASONING_EFFORT` (default: `low`):

- If the model supports the modern `reasoning` parameter, the pipe sets `reasoning.effort` and keeps reasoning enabled.
- If the model supports only the legacy `include_reasoning` flag, the pipe toggles it based on the configured effort.
- If the model supports neither, reasoning is disabled.

### Request-field filtering still applies

Housekeeping tasks are still passed through the same OpenRouter request-field filter (only documented OpenRouter Responses fields are retained; explicit `null` values are dropped).

### Cost snapshots can still be recorded

If the provider returns a `usage` object for the task request, the pipe can emit the same Redis-based cost snapshot telemetry as normal requests (when enabled), scoped to the task’s user.

## MOA merged-response behavior

`moa_response_generation` is intentionally **not** treated as housekeeping:

- It keeps the selected chat model instead of switching to a dedicated task model.
- It honours the incoming `stream` flag.
- It follows the normal chat/request path, including tools, provider routing, direct uploads, model restrictions, and normal chat-visible error handling.
- It does not use the housekeeping task adapter or task-specific fallback stub behaviour.

---

## Configuration guidance (operators)

Housekeeping tasks run frequently. The safest approach is to configure housekeeping to use a **dedicated task model configuration** that is:

- Low-latency and cost-efficient for short outputs.
- Configured to produce concise strings (titles/tags/summaries) rather than long prose.
- Not dependent on external tools or plugins (task requests do not execute tool loops).

If you need tasks to be as fast as possible, reduce `TASK_MODEL_REASONING_EFFORT` (for example to `minimal` or `none`). If task quality is inadequate, increase it (for example `medium`).

---

## Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Tasks often return `[Task error] ...` | Provider errors or repeated request failures in the housekeeping task adapter | Check backend logs for `Task model attempt ... failed` (DEBUG gives full stack traces). |
| Task outputs are overly verbose | Housekeeping prompt/model configuration encourages long-form responses | Tune the task prompt/model configuration for short outputs; consider lowering `TASK_MODEL_REASONING_EFFORT`. |
| Tasks are unexpectedly expensive | Housekeeping is targeting a high-cost model or generating long outputs | Confirm the configured task model and review `usage`/cost snapshots (if enabled). |
| Housekeeping tasks bypass the model allowlist unexpectedly | The request is using the housekeeping task adapter path | Treat this as expected behavior; if you need strict enforcement, control task model selection at the Open WebUI admin/config level. |
| MOA ignores housekeeping task settings | `moa_response_generation` now uses normal chat semantics | This is expected; MOA keeps the selected chat model and normal chat features. |

---

## Relevant valves

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. Housekeeping task behavior is primarily controlled by:

- `TASK_MODEL_REASONING_EFFORT` (default: `low`)
- `USE_MODEL_MAX_OUTPUT_TOKENS` (affects whether the pipe injects a model max for requests, including tasks)
