# task models & housekeeping best practices

**file:** `docs/task_models_and_housekeeping.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:6846-8190`, `Pipe._run_task_model_request`, `Pipe._apply_task_reasoning_preferences`, `Pipe.Valves.TASK_MODEL_REASONING_EFFORT`

This note captures everything operators need to know about “task models” inside Open WebUI—those quick, background invocations that keep chats tidy by generating titles, tags, summaries, and other housekeeping responses. Because these requests fire frequently and should never disrupt the main chat, we treat them differently from user-facing model runs. Use this guide whenever you expose new models through the pipe or tune valves that affect housekeeping.

---

## 1. how task models run inside the pipe

1. **Open WebUI tasks set the `__task__` flag.** When the UI needs metadata (e.g., “generate a title for this chat”), it schedules a request with `__task__` metadata. The pipe detects this at the start of `_process_transformed_request` (see line ~6846) and diverts to `_run_task_model_request` instead of the full streaming loop.
2. **Requests are forced into a non-streaming, single-shot path.** `_run_task_model_request` sends a Responses API call with `stream=False`, aggressive retries, and shorter timeouts so chores finish quickly and do not hold queue slots indefinitely.
3. **Task reasoning effort is governed by valves.** If the Open WebUI model belongs to this pipe, we override its reasoning payload via `_apply_task_reasoning_preferences`, honoring `TASK_MODEL_REASONING_EFFORT` (default `low`). This setting is independent from the main chat’s reasoning knobs, so you can keep chats verbose while making chores snappy.
4. **Allowlists are bypassed but still informative.** Even if `MODEL_ID` exposes only a subset of catalog models, Open WebUI tasks may target the pipe. We therefore treat housekeeping as an internal workload and focus on stability and cost efficiency.

> **Why this matters:** A chat can trigger multiple task runs per turn (title refresh, tagging, search-index metadata). Choosing the wrong model or misconfiguring prompts/tools can multiply latency and spend before the user sees anything.

---

## 2. model selection guidance (pick “mini” tiers)

| Goal | Recommendation |
| --- | --- |
| Latency-sensitive chores | Use OpenRouter’s “mini” or “small” models that deliver first tokens in <1s. |
| Cost containment | Minis cost a fraction of flagship models yet are accurate enough for metadata generation. |
| Housekeeping quality | Favor models that combine reasoning with tool awareness even at the lower tier. |

**Current top choices:**

1. **OpenAI · GPT-4.1 Mini** – Excellent balance of reasoning depth and latency. Handles long chat transcripts gracefully when generating titles or TL;DRs.
2. **OpenAI · GPT-5 Mini** – Slightly higher cost than 4.1 Mini but better multilingual tag quality and summarization fidelity.

> Treat these as the canonical task models unless your organization mandates another provider. Keeping consistency reduces surprises when audits compare cost reports vs. expected workloads.

### Why minis beat flagships for chores

* **Queue health:** Minis finish in milliseconds-to-low-seconds, so they occupy far fewer semaphore tokens (`MAX_CONCURRENT_REQUESTS`) and free resources for user-visible chats.
* **Token efficiency:** Task prompts are short, but full-model completions still incur minimum billing increments. Minis keep per-run costs negligible (< fractions of a cent).
* **Failure isolation:** If a flagship model rate-limits or experiences transient errors, chores stall. Minis are less prone to throttling, making housekeeping more predictable.

---

## 3. configuration checklist for task-friendly models

| Setting | Recommendation | Rationale |
| --- | --- | --- |
| **System prompt** | Leave blank (empty string). | Task prompts already include the context they need. Extra system text often forces the model to answer verbosely instead of returning a short title/tag list. |
| **Tools** | Do **not** attach tools. | `_run_task_model_request` does not execute tool loops. Adding tools risks 400s because housekeeping payloads omit the `tools` array entirely. |
| **Reasoning effort** | Set `TASK_MODEL_REASONING_EFFORT` to `low` (default) or `minimal`. | `low` gives the model enough budget for coherent summaries while keeping latency predictable. Use `minimal` if chores are still too slow; the pipe automatically skips auto-attached web-search in that case. |
| **Web-search plugin** | Leave disabled for task models. | Search is unnecessary for metadata about an existing chat and slows the request. The pipe already suppresses it when effort resolves to `minimal`. |
| **Parallel tool calls** | Irrelevant for chores; keep OFF to avoid sending unsupported flags. | `_filter_openrouter_request` strips unsupported keys, but keeping the model config simple avoids surprises if OpenRouter changes validation rules. |

> **Tip:** When cloning model entries inside Open WebUI, create a dedicated “Task – GPT-4.1 Mini” record with the above settings so operators don’t accidentally edit the main chat model.

---

## 4. wiring models inside Open WebUI

1. **Create dedicated task model entries.** In *Admin → Models*, duplicate your preferred mini model, rename it with a `Task` prefix, clear the system prompt, and remove any attached tools/MCP servers.
2. **Assign the model to housekeeping slots.** In *Settings → Pipes → OpenRouter Responses Pipe*, set “Task model” selectors (title generator, tagger, summarizer) to your new Task model entry.
3. **Verify per-user overrides.** `Pipe.UserValves` defaults keep reasoning enabled for chats but store `PERSIST_REASONING_TOKENS='next_reply'`. Users typically should not change the task model—they only control their chat flow.
4. **Monitor cost and latency.** Use OpenRouter’s usage dashboards. Tasks should remain a flat baseline; spikes indicate someone switched to a flagship or reintroduced a system prompt.

---

## 5. code paths worth knowing

* `_process_transformed_request` (≈ line 6846) detects `__task__` payloads and logs `"Detected task model"` so operators see it in DEBUG logs.
* `_apply_task_reasoning_preferences` applies the `TASK_MODEL_REASONING_EFFORT` valve and toggles `include_reasoning` off when the provider lacks native reasoning support.
* `_run_task_model_request` emits targeted retries (two attempts) and strips unsupported parameters before calling the Responses API.
* `_extract_task_output_text` converts the Responses payload into plain text, returning the first assistant message or the fallback `output_text` list.

Knowing these entry points helps when you trace why a specific task run failed or took longer than expected.

---

## 6. troubleshooting & quick fixes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Task requests time out or queue behind chats. | Using a heavyweight model or enabling web-search/tools. | Switch back to GPT-4.1 Mini / GPT-5 Mini, confirm system prompt is empty, and ensure no tools are attached. |
| Titles are verbose paragraphs. | Residual system prompt encourages essay-like answers. | Clear the system prompt and re-save the model entry; the pipe does not inject extra formatting. |
| Repeated 400 errors referencing tool payloads. | Task model configured with tools that the pipe never declares. | Remove tools from the model definition; housekeeping does not support tool calls. |
| Tagging misses recent turns. | Reasoning effort set to `minimal` while expecting deep summaries. | Raise `TASK_MODEL_REASONING_EFFORT` to `low` or `medium`; minis still perform well at those levels. |
| Costs spike after admins add new models. | Housekeeping switched to a flagship model by accident. | Check admin model assignments, re-point to GPT-4.1 Mini/GPT-5 Mini, and consider locking down permissions. |

---

## 7. next steps for operators

1. **Standardize on the recommended minis** across environments (dev/staging/prod) so QA mirrors production behavior.
2. **Document the model IDs** in your runbooks (e.g., `openai.gpt-4.1-mini`). When OpenRouter updates slugs, update the runbook and the model entry together.
3. **Review task latency weekly.** Use the Open WebUI telemetry or OpenRouter usage logs; aim for <2s p95 per housekeeping run.

By following these guardrails you ensure housekeeping stays invisible, cheap, and reliable—letting flagship models handle the conversations that users actually see.

