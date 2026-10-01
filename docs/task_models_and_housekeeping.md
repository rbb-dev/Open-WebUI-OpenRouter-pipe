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

A task's cap is the caller's: `tasks.py` applies `{'max_tokens': 4}` to the emoji task and 1000 to a title, and an admin can set `task.model.params.max_tokens` to anything. A cap that expresses a number is honoured as that number — an integer as it stands, a numeric string read as the number it spells, a float rounded — while a value that is not a number is left out of the request, as Open WebUI's own cast of the same field does. What the pipe also does is bound the thinking budget inside it, because reasoning tokens count against the cap: on Gemini 2.5 the budget becomes `min(budget, cap - 64)`. When the cap leaves no room for any thinking the pipe asks for no bounded budget at all and writes no off flag, so the model thinks at its own default and the cap still governs the answer.

### Where `task.model.params` is applied, and where it is not

`task.model.params` is an **admin setting, and the pipe treats it as one — but only for calls the pipe itself builds.** The boundary matters, because there are two kinds of task call and they get the params row by different routes:

- **Open WebUI's own housekeeping calls** (titles, tags, emoji, autocomplete). Open WebUI reads the row itself and applies it in `routers/tasks.py:202` *before* the request reaches the pipe, so the pipe sees the effect already in the payload. The pipe neither re-reads nor re-applies the row on this path; it honours whatever arrives.
- **The pipe's own structured-task call** (the video intent classifier). Here the pipe builds the payload and hands it to `generate_chat_completion`, which applies nothing. The pipe is the only place the row can be applied, so it is: the row is read once per turn and merged for keys the payload does not already set, and **where** it lands follows the candidate model's owner. A task model whose `owned_by` in `request.app.state.MODELS` is `ollama` takes the row under the payload's `options` key, because `convert_payload_openai_to_ollama` forwards only `options`; anything else takes it at the top level. The owner is resolved per candidate, since the shipped `other_task_model` fallback crosses internal and external.

Two kinds of key in the row are never applied on the pipe's own call, on either arm of that placement rule, for the reasons Open WebUI gives for dropping them (`utils/payload.py:90-103`, `:118-122`). Keys the call already set stay set — the classifier's own `temperature: 0`, its own `stream: False` and its own `response_format` are correctness constraints, not defaults, since a strict-JSON classifier that samples its own output cannot be relied on to hold the schema, and neither is nested under `options` either, where an admin's `temperature` would reach Ollama while the converter drops the pipe's own top-level `0`. (The *schema envelope* is the correctness constraint; the `strict` flag inside it is decided per candidate from that task model's own catalog row and relaxed to `false` where the endpoint does not advertise `structured_outputs`.) And Open WebUI's request-scoped keys (`system`, `stream_response`, `stream_delta_chunk_size`, `function_calling`, `reasoning_tags`, `compact_token_threshold`, `note_id`, `tool_approval_mode`) are dropped: they configure Open WebUI's request plumbing rather than a provider call, and `system` in particular would replace the classifier's own system prompt, which is where its JSON schema lives. They are dropped from `custom_params` as well as from the top level — a deliberate divergence from `utils/payload.py:101-103`, which deletes them from the top level only. `system` is the key that divergence protects: it is the one whose arrival would replace the classifier's own prompt, and the other seven are dropped with it rather than on their own account. `custom_params` is deep-merged after its string values are JSON-decoded, and lands with the rest of the row: a `custom_params` key overrides a top-level key of the same name, and an object named at both levels arrives as the union with `custom_params` on top. The call's own keys still win over both. An absent or non-dict row leaves the payload byte-identical. See [Video Intent Classifier](openrouter_video_intent_classifier.md).
---

## Housekeeping task behavior (what is different vs normal chat)

### Non-streaming request

Housekeeping tasks are forced to **non-streaming** behavior (`stream=false`) and processed as a single request/response. They use whichever OpenRouter endpoint a non-streamed chat turn would use under the same valves — `Default API endpoint`, both `Models forced to …` valves and the responses-to-chat fallback included — because Open WebUI sends a task through the same `generate_chat_completion` as a chat turn. Two request-side inputs fix the endpoint ahead of the valves, and a task honours both exactly as a chat turn does: a request-level `preset`, or a direct video/audio upload that requires chat. A chat reply is converted to Responses shape first, so the extraction below reads the same payload either way. The pipe then extracts plain text from that payload.

### Output extraction rules

The pipe extracts housekeeping task output text from:

- `output[].type == "message"` items containing `content[].type == "output_text"` parts: the parts of one item are joined with no separator, distinct items with a newline, and an item whose text is empty or all whitespace contributes nothing. That is Open WebUI's own rule, in `open_webui.utils.misc.get_output_text`, and the pipe follows it so a value the provider streamed in parts -- the shape `response.output_item.content_part.added` exists for -- reads as the one value it is instead of as several.
- Fallback: a top-level `output_text` string (some providers return a collapsed field), read **only when the `output[]` walk produced no usable text** -- a walk that yielded only empty or whitespace-only parts counts as no usable text. The two spellings are alternatives, never concatenated: a response carrying both yields the `output[]` answer, once.

The `type == "output_text"` part filter is the one place the pipe keeps its own rule rather than Open WebUI's: Open WebUI takes *any* part carrying a `text` key, and the pipe does not, because widening would splice a `refusal` or `summary_text` part into the JSON the classifier is about to parse. The `type == "message"` item filter is an allow-list on both sides, so a `reasoning` item -- or any other item type, or an item with no type at all -- contributes nothing however its content parts are typed. A non-blank `refusal` is read by `read_task_model_response_json` before it reads anything else, on both of its arms, `output` and `choices` alike: it raises `task_model_refusal` there, and on the `/chat/completions` leg the non-streamed adapter raises `TaskProviderRefusal` at the same point of the fold, so a refused task model is never used as a classifier result and never becomes the value a housekeeping consumer reads. The two arms carry it differently: in `output` a refusal is a **content part** inside `content` (`{refusal: string, type: "refusal"}`, per `create-a-response.md:4288-4302, 7540-7546`), and in `choices` it is a **sibling** of `content` (per `create-a-chat-completion.md:2803, 4831`). Either way the attempt fails and takes the `_task_refusal_result` route below, on either endpoint. The refusal text itself is never carried in the fault.

If the provider returns no usable text, the pipe does not raise. An answer that is one JSON object spread over both spellings, joined to itself, fails to parse for the same reason as no answer at all; the two spellings are alternatives, so the pipe never returns that shape. The failure is routed through `_task_refusal_result`: a kind Open WebUI persists (`title_generation`, `tags_generation`, `follow_up_generation`) gets a contextual card naming the task, the model, the attempt count, the error class and the error id; every other kind gets `""`, because Open WebUI hands that return value straight to a consumer that displays it. A warning toast carrying the same card is emitted on the one channel that is still open -- once per chat-or-user and model within the notification window. A temporary chat is the exception, and deliberately so: it is warned on **every** failing turn, and its id (a `temporary:` or `local:` id — see `temporary_chat_prefixes()`) is never written to the "already notified" record, and never printed in a log line, where it reads `<not retained>`. The toast is *not* suppressed along with the key: withholding the key is what keeps the raw id out of a process-lifetime structure, and it is not a reason to stop telling the person. A `channel:` chat is latched like any saved chat, because Open WebUI does store it, and a call carrying no `chat_id` at all is warned once per process under a `__no_chat_or_user__` sentinel. See [Video Intent Classifier](openrouter_video_intent_classifier.md) for the same rule stated once.

### How many requests a failing task makes

A task tries twice. While `AUTO_FALLBACK_CHAT_COMPLETIONS` is on, one attempt is itself two requests where the fallback fires, so a failing task makes at most **three**: either `responses, chat/completions` in the first attempt with one left, or `responses, chat/completions, chat/completions` when the retry goes straight back to the endpoint the first attempt settled on. A retry does not re-run a fallback it has already taken. With the fallback off, the bound is two.

### Model whitelist bypass (task-mode only)

If a `MODEL_ID` allowlist is configured, normal chat requests enforce it. Housekeeping tasks can **bypass** the allowlist so housekeeping continues even when the selected task model is not in the allowlist.

Important nuance:

- Reasoning overrides described below apply only when the task request targets a model that the pipe considers “owned” (i.e., inside the pipe’s allowed model set when an allowlist is configured). When a task bypasses the allowlist, the pipe does not force task reasoning overrides for that model.
- Model catalog filters (for example `FREE_MODEL_FILTER` and `TOOL_CALLING_FILTER`) control which models are shown to users and enforced for normal chat requests. Housekeeping tasks are not guaranteed to respect those filters; choose task models explicitly if you need “free only” or “tool calling required” behavior for housekeeping.

### Task reasoning effort override

For housekeeping tasks targeting models the pipe “owns”, the pipe overrides the request’s reasoning configuration using `TASK_MODEL_REASONING_EFFORT` (default: `low`):

- If the model supports the modern `reasoning` parameter, the pipe sets `reasoning.effort` and keeps reasoning enabled. A `none` on a model whose reasoning is mandatory becomes the lowest level its catalog entry lists other than `none`, and no level at all when it lists no other level. On any other model, an effort of `none` is the one that turns reasoning off, and the reasoning pass that runs after the task override is what decides it: it rebuilds the `reasoning` object, so the `enabled` key nothing else wrote is not on the request a task asking for no effort goes out with. A task request that only hides the trace (`reasoning.exclude` of `true`) is not an off: it keeps the depth the valve chose and draws no status line about it.
- If the model supports only the legacy `include_reasoning` flag, the pipe toggles it based on the configured effort — except on a row whose reasoning is mandatory, where there is no effort to substitute and the flag goes out as `true` whatever the configured effort says.
- If the model supports neither, the pipe adds no reasoning field; any the task request itself carries (for example Open WebUI's task-model parameters) goes out as sent.

A model the catalogue marks `reasoning.mandatory` is not asked to stop, whatever this valve says — the pipe substitutes an effort the row supports. On a row whose only reasoning channel is the legacy `include_reasoning` flag there is no effort to substitute, so the pipe writes `true` instead of the off. Nothing is reported about it, because a task emits no status line of its own; the status line a chat shows does not apply here.

### Request-field filtering still applies

Housekeeping tasks are still passed through the same OpenRouter request-field filter (only documented OpenRouter Responses fields are retained; explicit `null` values are dropped). That filter runs on the responses branch only: a task routed to `/chat/completions` is translated after the filter has already had its say, so the field it drops is the one the responses endpoint does not take.

### The server-tool switches apply here too

A task turn's request is filtered by the same `ENABLE_*` server-tool switches as a chat turn's: while one is off, the pipe does not send that tool on a task request either, even if an inlet filter or an admin's Task Model parameters row put it there. The switch is about the deployment, not about the kind of request. Nothing else about a task turn changes — the pipe still *injects* no server tools on this leg, so a title generation never gains a web search or an advisor of its own.

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
warmup, a missing or full request queue, a setup exception — is refused per class,
and never with a stub:

| Class | What a pre-enqueue refusal returns |
| --- | --- |
| **Persisted** | the card, as for any other refusal of that kind |
| **Content-consumed** (`context_compaction`, `context_summary`, `memory_review`) | `""`, re-derived by Open WebUI, for the reason the table above gives |
| **Display-string**, and any other task-adapter kind | the refusal sentence |
| A kind whose name carries `follow`, `tag` or `title` but is not one of the three persisted kinds | the stub. No kind in Open WebUI's `TASKS` enum is in this class — `title_generation`, `follow_up_generation` and `tags_generation` are all persisted and checked first — so this row is a guard kept for a future kind, and it is documented so nobody reads the arm as dead code. |

The same holds for the refusals
that happen in-request, after the job is queued: ZDR enforcement on a model with no
ZDR endpoint, the same refusal when the endpoint list cannot be read at all, an
endpoint-override conflict from a preset on a model forced to `/responses`, a provider
`refusal` in the task model's own reply on either endpoint, and the
six sites inside `_handle_pipe_call` where the job itself fails — an artifact store
that will not open, an API key that cannot be read, an active auth breaker, an
unusable model catalog, and the catch-all tail. One more sits outside `_handle_pipe_call` entirely: the `await future` arm of `_pipe_impl`, which refuses a job the pipe has already handed to the dispatch worker. A task
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

The adapter's failure tail follows the same rule, as does the `await future` arm in
`_pipe_impl`: a persisted kind gets the card, every other kind gets `""`, and the toast
carries the text. `str(last_error)`
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
- Not dependent on external tools or plugins (task requests do not execute tool loops, and a housekeeping request builds no tool specs at all).

If you need tasks to be as fast as possible, reduce `TASK_MODEL_REASONING_EFFORT` (for example to `minimal` or `none`). `none` switches reasoning off where the model allows it; a model whose reasoning is mandatory answers at the lowest level its catalog entry lists other than `none` instead, and a request that only hides the trace (`reasoning.exclude` of `true`) is not a `none` at all. If task quality is inadequate, increase it (for example `medium`).

---

## Troubleshooting

| Symptom | Likely cause | What to check |
|---|---|---|
| A housekeeping task failed | Provider errors or repeated request failures in the housekeeping task adapter | A warning toast names the task, the model, the attempt count, the error class and the error id; the same id is on the `Task model '<task>' failed after N attempt(s)` **ERROR** record, which carries the model id, the error class and the request id. The per-attempt `Task model attempt %d/%d failed` records are WARNING and do not carry the id; DEBUG adds full stack traces (except on an auth failure, where `exc_info` is deliberately withheld at `task_model_adapter.py:271`). The toast is emitted once per chat-or-user and model per window, so a long outage toasts once rather than on every dispatch, including when two of Open WebUI's tasks fail at the same time. A temporary chat is the exception: it is warned on **every** failing turn, its id is never written to the "already notified" record and never printed in a log line (`<not retained>`), but the toast is not withheld along with it — a missing toast is therefore not this latch. A `channel:` chat is latched like any saved chat, and a call carrying no `chat_id` at all is warned once per process. |
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
