# Task models and housekeeping

**Scope:** How the pipe handles Open WebUI “task” requests (`__task__`) such as title/tag/summary generation, and how to operate them safely in production.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md)

Open WebUI can issue two different kinds of requests through `__task__`:

- **Housekeeping tasks** such as generating a chat title, tags, follow-ups, queries, autocomplete, emoji, or image prompts. These should be fast, short, and low-risk.
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

Housekeeping tasks are forced to **non-streaming** behavior (`stream=false`) and processed as a single request/response. They use whichever OpenRouter endpoint a non-streamed chat turn would use under the same valves — `Default API endpoint`, both `Models forced to …` valves and the responses-to-chat fallback included — because Open WebUI sends a task through the same `generate_chat_completion` as a chat turn. Two request-side inputs fix the endpoint ahead of the valves, and a task honours both exactly as a chat turn does: a request-level `preset`, or a direct video/audio upload that requires chat. A chat reply is converted to Responses shape first, so the extraction below reads the same payload either way. The pipe then extracts plain text from that payload.

### Output extraction rules

The pipe extracts housekeeping task output text from:

- `output[].type == "message"` items containing `content[].type == "output_text"`, concatenated with newlines.
- Fallback: a top-level `output_text` string (some providers return a collapsed field).

If the provider returns no usable text, the pipe returns a safe placeholder error string to Open WebUI rather than raising an exception.

### How many requests a failing task makes

A task tries twice. While `AUTO_FALLBACK_CHAT_COMPLETIONS` is on, one attempt is itself two requests where the fallback fires, so a failing task makes at most **three**: either `responses, chat/completions` in the first attempt with one left, or `responses, chat/completions, chat/completions` when the retry goes straight back to the endpoint the first attempt settled on. A retry does not re-run a fallback it has already taken. With the fallback off, the bound is two.

### Model whitelist bypass (task-mode only)

If a `MODEL_ID` allowlist is configured, normal chat requests enforce it. Housekeeping tasks can **bypass** the allowlist so housekeeping continues even when the selected task model is not in the allowlist.

Important nuance:

- Reasoning overrides described below apply only when the task request targets a model that the pipe considers “owned” (i.e., inside the pipe’s allowed model set when an allowlist is configured). When a task bypasses the allowlist, the pipe does not force task reasoning overrides for that model.
- Model catalog filters (for example `FREE_MODEL_FILTER` and `TOOL_CALLING_FILTER`) control which models are shown to users and enforced for normal chat requests. Housekeeping tasks are not guaranteed to respect those filters; choose task models explicitly if you need “free only” or “tool calling required” behavior for housekeeping.

### Task reasoning effort override (valve-gated)

For housekeeping tasks targeting models the pipe “owns”, the pipe overrides the request’s reasoning configuration using `TASK_MODEL_REASONING_EFFORT` (default: `low`):

- If the model supports the modern `reasoning` parameter, the pipe sets `reasoning.effort` and keeps reasoning enabled. A `none` on a model whose reasoning is mandatory becomes the lowest level its catalog entry lists other than `none`, and no level at all when it lists no other level.
- If the model supports only the legacy `include_reasoning` flag, the pipe toggles it based on the configured effort.
- If the model supports neither, the pipe adds no reasoning field; any the task request itself carries (for example Open WebUI's task-model parameters) goes out as sent.

### Request-field filtering still applies

Housekeeping tasks are still passed through the same OpenRouter request-field filter (only documented OpenRouter Responses fields are retained; explicit `null` values are dropped). That filter runs on the responses branch only: a task routed to `/chat/completions` is translated after the filter has already had its say, so the field it drops is the one the responses endpoint does not take.

### Nothing a housekeeping task does reaches the chat message

Open WebUI launches housekeeping tasks with the metadata of the turn that just
finished, so the event emitter the pipe receives is bound to the **assistant
message the user is already reading**. Anything written to it lands on that
answer: a `chat:message` replaces its text outright, a `chat:message:delta`
appends to it, a `chat:completion` supplies its error banner and its usage
figures, and a `status` is appended to its status history and stored with it.

For housekeeping tasks the pipe closes all four of those channels for the whole
job, from the moment the task is recognised through to the reply. A task that
fails still returns the parseable JSON stub Open WebUI expects — a title, a tag
list, an empty follow-up list — and the answer on screen is left exactly as the
model wrote it. Toast notifications are the one channel still open, because Open
WebUI shows those beside the conversation rather than inside a message.

MOA merged-response synthesis is a visible answer rather than housekeeping, so
none of this applies to it.

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

- Low-latency for short outputs.
- Configured to produce concise strings (titles/tags/summaries) rather than long prose.
- Not dependent on external tools or plugins (task requests do not execute tool loops).

If you need tasks to be as fast as possible, reduce `TASK_MODEL_REASONING_EFFORT` (for example to `minimal` or `none`). `none` switches reasoning off where the model allows it; a model whose reasoning is mandatory answers at the lowest level its catalog entry lists other than `none` instead. If task quality is inadequate, increase it (for example `medium`).

---

## Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| Tasks often return `[Task error] ...` | Provider errors or repeated request failures in the housekeeping task adapter | Check backend logs for `Task model attempt ... failed` (DEBUG gives full stack traces). |
| Task outputs are overly verbose | Housekeeping prompt/model configuration encourages long-form responses | Tune the task prompt/model configuration for short outputs; consider lowering `TASK_MODEL_REASONING_EFFORT`. |
| Housekeeping is running up unexpected spend | Housekeeping runs on every chat, so the task model it targets and the length of what that model produces both drive the total | Confirm the configured task model and review `usage`/cost snapshots (if enabled). What a model charges is on OpenRouter's pricing page. |
| Housekeeping tasks bypass the model allowlist unexpectedly | The request is using the housekeeping task adapter path | Treat this as expected behavior; if you need strict enforcement, control task model selection at the Open WebUI admin/config level. |
| MOA ignores housekeeping task settings | `moa_response_generation` now uses normal chat semantics | This is expected; MOA keeps the selected chat model and normal chat features. |

---

## Relevant valves

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. Housekeeping task behavior is primarily controlled by:

- `TASK_MODEL_REASONING_EFFORT` (default: `low`)
- `USE_MODEL_MAX_OUTPUT_TOKENS` (affects whether the pipe injects a model max for requests, including tasks)
