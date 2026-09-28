# Task models and housekeeping

**Scope:** How the pipe handles Open WebUI “task” requests (`__task__`) such as title/tag/summary generation, and how to operate them safely in production.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md)

Open WebUI can issue two different kinds of requests through `__task__`:

- **Housekeeping tasks** such as generating a chat title, tags, follow-ups, queries, autocomplete, emoji, or image prompts. These should be fast, short, and low-risk.
- **MOA merged-response synthesis** (`moa_response_generation`), which is user-visible and should behave like a normal chat response.

The pipe treats these categories differently.

Housekeeping tasks split three ways by **what happens to the value the pipe returns**, and the three are not the same thing:

| Category | Kinds | What the return value becomes |
| --- | --- | --- |
| **Persisted** | `title_generation`, `tags_generation`, `follow_up_generation` | A card a person reads in a field. The pipe's card is the whole answer, and a refusal is shown as itself. |
| **Content-consumed** | `context_compaction`, `context_summary`, `memory_review` | Prose Open WebUI feeds back to the model. Nobody displays it, and **any truthy value is stored as the chat's own memory**. |
| **Display-string** | `query_generation`, `emoji_generation`, `autocomplete_generation`, and anything else not in the two sets above | A string a person reads; a refusal that is not a card is not useful, so it is `""`. |

The middle category is the one to understand. The `context_compaction` sub-turn's return value becomes `contextSummary` and then the model-visible `[CONVERSATION SUMMARY]`, and `context_summary` and `memory_review` feed the same store. `context_compaction.py:385-395` takes the pipe's answer **if one is non-empty** and only re-derives a summary from the messages themselves when it is not — so returning an error sentence here converts Open WebUI's designed degradation into a corrupt summary that *looks* like a successful compaction and is then served to the model as fact.

So a refused content-consumed kind returns `""`. That is Open WebUI's own degradation, not a new one: an empty summary is declined and re-derived, and the turn continues on full chat history. **The failure is visible only in the backend log**, as `Task model attempt N/M failed` — including for `context_summary`, which does carry a `chat_id` and no `message_id` and would otherwise meet the API caller's no-chat gate. It is caught and answered inside the task adapter, before the request loop can build an HTTP error, so no caller ever sees an envelope for these three. If a compaction silently looks wrong, read the backend log; nothing appears in a browser console.

---

## How the pipe detects a task request

The pipe treats a request as a task when the special `__task__` argument is present (a dict or task name). When detected:

- Housekeeping tasks log a DEBUG message (`Detected task model: ...`) and use the dedicated task adapter path.
- `moa_response_generation` keeps the normal streaming/tool-execution path even though `__task__` is present.

### Task output caps

A task's cap is the caller's: `tasks.py` applies `{'max_tokens': 4}` to the emoji task and 1000 to a title, and an admin can set `task.model.params.max_tokens` to anything. The pipe never raises, floors or drops it. What it does do is bound the thinking budget inside it, because reasoning tokens count against the cap: on Gemini 2.5 the budget becomes `min(budget, cap - 64)`. When the cap leaves no room for any thinking the pipe asks for no bounded budget at all and writes no off flag, so the model thinks at its own default and the cap still governs the answer.

---

## Housekeeping task behavior (what is different vs normal chat)

### Non-streaming request

Housekeeping tasks are forced to **non-streaming** behavior (`stream=false`) and processed as a single request/response. They use whichever OpenRouter endpoint a non-streamed chat turn would use under the same valves — `Default API endpoint`, both `Models forced to …` valves and the responses-to-chat fallback included — because Open WebUI sends a task through the same `generate_chat_completion` as a chat turn. Two request-side inputs fix the endpoint ahead of the valves, and a task honours both exactly as a chat turn does: a request-level `preset`, or a direct video/audio upload that requires chat. A chat reply is converted to Responses shape first, so the extraction below reads the same payload either way. The pipe then extracts plain text from that payload.

### Output extraction rules

The pipe extracts housekeeping task output text from:

- `output[].type == "message"` items containing `content[].type == "output_text"`, concatenated with newlines.
- Fallback: a top-level `output_text` string (some providers return a collapsed field).

The `type == "message"` filter is an allow-list, so a `reasoning` item -- or any other item type, or an item with no type at all -- contributes nothing however its content parts are typed. A `message` item carrying a non-blank `refusal` raises `task_model_refusal` on every arm, `output` and `choices` alike, so a refused task model is never used as a classifier result. The refusal text itself is never carried in the fault.

If the provider returns no usable text, the pipe does not raise. The failure is routed through `_task_refusal_result`: a kind Open WebUI persists (`title_generation`, `tags_generation`, `follow_up_generation`) gets a contextual card naming the task, the model, the attempt count, the error class and the error id; every other kind gets `""`, because Open WebUI hands that return value straight to a consumer that displays it. A warning toast carrying the same card is emitted on the one channel that is still open -- once per chat-or-user and model within the notification window.

### How many requests a failing task makes

A task tries twice. While `AUTO_FALLBACK_CHAT_COMPLETIONS` is on, one attempt is itself two requests where the fallback fires, so a failing task makes at most **three**: either `responses, chat/completions` in the first attempt with one left, or `responses, chat/completions, chat/completions` when the retry goes straight back to the endpoint the first attempt settled on. A retry does not re-run a fallback it has already taken. With the fallback off, the bound is two.

### Model whitelist bypass (task-mode only)

If a `MODEL_ID` allowlist is configured, normal chat requests enforce it. Housekeeping tasks can **bypass** the allowlist so housekeeping continues even when the selected task model is not in the allowlist.

Important nuance:

- Reasoning overrides described below apply only when the task request targets a model that the pipe considers “owned” (i.e., inside the pipe’s allowed model set when an allowlist is configured). When a task bypasses the allowlist, the pipe does not force task reasoning overrides for that model.
- Model catalog filters (for example `FREE_MODEL_FILTER` and `TOOL_CALLING_FILTER`) control which models are shown to users and enforced for normal chat requests. Housekeeping tasks are not guaranteed to respect those filters; choose task models explicitly if you need “free only” or “tool calling required” behavior for housekeeping.

### Task reasoning effort override

For housekeeping tasks targeting models the pipe “owns”, the pipe overrides the request’s reasoning configuration using `TASK_MODEL_REASONING_EFFORT` (default: `low`):

- If the model supports the modern `reasoning` parameter, the pipe sets `reasoning.effort` and keeps reasoning enabled. A `none` on a model whose reasoning is mandatory becomes the lowest level its catalog entry lists other than `none`, and no level at all when it lists no other level. On any other model, an effort of `none` is the one that turns reasoning off, so the pipe also removes the `enabled` key the chat-turn pass left on the request — a task asking for no effort is not sent a request that also insists reasoning is on.
- If the model supports only the legacy `include_reasoning` flag, the pipe toggles it based on the configured effort.
- If the model supports neither, the pipe adds no reasoning field; any the task request itself carries (for example Open WebUI's task-model parameters) goes out as sent.

A model the catalogue marks `reasoning.mandatory` is not asked to stop, whatever this valve says — the pipe substitutes an effort the row supports. Nothing is reported about it, because a task emits no status line of its own; the status line a chat shows does not apply here.

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
fails returns a contextual card for a persisted kind and the empty string for
every other, and the answer on screen is left exactly as the model wrote it. The refusal paths are the exception, and deliberately so: a task
the pipe refuses before it enqueues the job — an open circuit breaker, a failed
warmup, a missing or full request queue, a setup exception — gets the refusal
sentence rather than a stub, for every kind. The same holds for the refusals
that happen in-request, after the job is queued: ZDR enforcement on a model with no
ZDR endpoint, the same refusal when the endpoint list cannot be read at all, an
endpoint-override conflict from a preset on a model forced to `/responses`, and the
six sites inside `_handle_pipe_call` where the job itself fails — an artifact store
that will not open, an API key that cannot be read, an active auth breaker, an
unusable model catalog, and the catch-all tail. A task
on any of those gets the refusal card only if Open WebUI persists that kind's
return value; every other kind gets the empty string, because Open WebUI hands that
return value to a consumer that displays it — a search query, an emoji, a prompt
suggestion — and the sentence would be shown as one. A stub is a write, not
a no-op: Open WebUI parses the reply and persists it, naming the chat from the
returned `title`, **replacing** the stored tag list, and storing the follow-ups, so
a stub handed back during an outage would overwrite the user's own title and tags
with no retry. A value that does not parse leaves the stored tags and follow-ups
untouched, because those two writes sit inside the `try` that wraps the parse.
The title is different: `update_chat_title_by_id` sits outside it, and Open
WebUI's own `if not title:` fallback renames the chat to the user's first
message on *any* title parse failure — for the card, for `""` and for an
outage that happened without this pipe at all. That is Open WebUI's behaviour
for its own failed title generation, the pipe follows it rather than returning a
stub, and the alternative would be renaming the chat to the literal placeholder
"Chat". Toast notifications are the one channel still open, because Open WebUI
shows those beside the conversation rather than inside a message; a
refused task says so in a single plain sentence. The two refusal sites that have
already rendered an error card into the chat do not toast as well — the card is
the report, and a second rendering of the same failure is noise rather than
information.

The adapter's failure tail follows the same rule: a persisted kind gets the card,
every other kind gets `""`, and the toast carries the text. `str(last_error)`
never reaches the returned string — for `query_generation` it would be sent to a
search provider, and for `context_compaction` stored as the chat's summary.

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
| A housekeeping task failed | Provider errors or repeated request failures in the housekeeping task adapter | A warning toast names the task, the model, the attempt count, the error class and the error id; the same id is on the `Task model '<task>' failed after N attempt(s)` **ERROR** record, which carries the model id, the error class and the request id. The per-attempt `Task model attempt %d/%d failed` records are WARNING and do not carry the id; DEBUG adds full stack traces (except on an auth failure, where `exc_info` is deliberately withheld at `task_model_adapter.py:267`). The toast is emitted once per chat-or-user and model per window, so a long outage toasts once rather than on every dispatch. |
| A `[CONVERSATION SUMMARY]` that is a `- role: …` bullet list of truncated message starts | The compaction sub-turn failed: the pipe returned nothing for `context_compaction`, so Open WebUI re-derived a lossy summary from the messages themselves (`context_compaction.py:390-395`) | The pipe's own failure is visible only in the backend log, as `Task model attempt N/M failed`. The summary looks like a successful compaction and is not one. There is no browser-console error and no HTTP envelope for this kind — see the content-consumed category above. |
| Task outputs are overly verbose | Housekeeping prompt/model configuration encourages long-form responses | Tune the task prompt/model configuration for short outputs; consider lowering `TASK_MODEL_REASONING_EFFORT`. |
| A task fails and the model is one that thinks | The task's cap is below what the model needs for thinking | Two halves: the pipe's is `budget = min(budget, cap - 64)` on Gemini 2.5, and under `cap < 65` it asks for no bounded budget at all; the other half is the cap itself, which the pipe must not raise. Raise `task.model.params.max_tokens` (4 for the emoji task, 1000 for a title) or point the task at a different model. |
| Housekeeping is running up unexpected spend | Housekeeping runs on every chat, so the task model it targets and the length of what that model produces both drive the total | Confirm the configured task model and review `usage`/cost snapshots (if enabled). What a model charges is on OpenRouter's pricing page. |
| Housekeeping tasks bypass the model allowlist unexpectedly | The request is using the housekeeping task adapter path | Treat this as expected behavior; if you need strict enforcement, control task model selection at the Open WebUI admin/config level. |
| MOA ignores housekeeping task settings | `moa_response_generation` now uses normal chat semantics | This is expected; MOA keeps the selected chat model and normal chat features. |

---

## Relevant valves

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. Housekeeping task behavior is primarily controlled by:

- `TASK_MODEL_REASONING_EFFORT` (default: `low`)
- `USE_MODEL_MAX_OUTPUT_TOKENS` (affects whether the pipe injects a model max for requests, including tasks)
