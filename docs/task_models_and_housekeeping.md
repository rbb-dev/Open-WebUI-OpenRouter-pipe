# Task models and housekeeping

**Scope:** How the pipe handles Open WebUI “task” requests (`__task__`) such as title/tag/summary generation, and how to operate them safely in production.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md)

Open WebUI can issue background “task” requests for housekeeping (for example, generating a chat title). These requests should be fast, inexpensive, and low-risk. The pipe treats them differently from user-facing chat requests.

---

## How the pipe detects a task request

The pipe treats a request as a task when the special `__task__` argument is present (a dict). When detected:

- The pipe logs a DEBUG message (`Detected task model: ...`).
- The request is routed to a dedicated task path instead of the normal streaming/tool-execution path.

---

## Task request behavior (what is different vs normal chat)

### Non-streaming request

Task requests are forced to **non-streaming** behavior (`stream=false`) and processed as a single request/response. The pipe then extracts plain text from the Responses payload.

### Output extraction rules

The pipe extracts task output text from:

- `output[].type == "message"` items containing `content[].type == "output_text"`, concatenated with newlines.
- Fallback: a top-level `output_text` string (some providers return a collapsed field).

If the provider returns no usable text, the pipe returns a safe placeholder error string to Open WebUI rather than raising an exception.

### Model whitelist bypass (task-mode only)

If a `MODEL_ID` allowlist is configured, normal chat requests enforce it. Task requests can **bypass** the allowlist so housekeeping continues even when the selected task model is not in the allowlist.

Important nuance:

- Reasoning overrides described below apply only when the task request targets a model that the pipe considers “owned” (i.e., inside the pipe’s allowed model set when an allowlist is configured). When a task bypasses the allowlist, the pipe does not force task reasoning overrides for that model.
- Model catalog filters (for example `FREE_MODEL_FILTER` and `TOOL_CALLING_FILTER`) control which models are shown to users and enforced for normal chat requests. Task requests are not guaranteed to respect those filters; choose task models explicitly if you need “free only” or “tool calling required” behavior for housekeeping.

### Task reasoning effort override (valve-gated)

For task requests targeting models the pipe “owns”, the pipe overrides the request’s reasoning configuration using `TASK_MODEL_REASONING_EFFORT` (default: `low`):

- If the model supports the modern `reasoning` parameter, the pipe sets `reasoning.effort` and keeps reasoning enabled.
- If the model supports only the legacy `include_reasoning` flag, the pipe toggles it based on the configured effort.
- If the model supports neither, reasoning is disabled.

### Request-field filtering still applies

Task requests are still passed through the same OpenRouter request-field filter (only documented OpenRouter Responses fields are retained; explicit `null` values are dropped).

### Cost snapshots can still be recorded

If the provider returns a `usage` object for the task request, the pipe can emit the same Redis-based cost snapshot telemetry as normal requests (when enabled), scoped to the task’s user.

---

## Configuration guidance (operators)

Task requests run frequently. The safest approach is to configure housekeeping to use a **dedicated task model configuration** that is:

- Low-latency and cost-efficient for short outputs.
- Configured to produce concise strings (titles/tags/summaries) rather than long prose.
- Not dependent on external tools or plugins (task requests do not execute tool loops).

If you need tasks to be as fast as possible, reduce `TASK_MODEL_REASONING_EFFORT` (for example to `minimal` or `none`). If task quality is inadequate, increase it (for example `medium`).

---

## Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Tasks often return `[Task error] ...` | Provider errors or repeated request failures | Check backend logs for `Task model attempt ... failed` (DEBUG gives full stack traces). |
| Task outputs are overly verbose | Task prompt/model configuration encourages long-form responses | Tune the task prompt/model configuration for short outputs; consider lowering `TASK_MODEL_REASONING_EFFORT`. |
| Tasks are unexpectedly expensive | Housekeeping is targeting a high-cost model or generating long outputs | Confirm the configured task model and review `usage`/cost snapshots (if enabled). |
| Tasks bypass the model allowlist unexpectedly | `__task__` is present so allowlist enforcement is bypassed | Treat this as expected behavior; if you need strict enforcement, control task model selection at the Open WebUI admin/config level. |

---

## Relevant valves

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. Task-mode behavior is primarily controlled by:

- `TASK_MODEL_REASONING_EFFORT` (default: `low`)
- `USE_MODEL_MAX_OUTPUT_TOKENS` (affects whether the pipe injects a model max for requests, including tasks)
