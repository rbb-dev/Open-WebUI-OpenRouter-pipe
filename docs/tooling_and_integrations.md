# Tools, plugins, and integrations

**Scope:** How tool schemas are built, how `function_call` items are executed, and how Open WebUI tool sources (registry + Direct Tool Servers) are attached.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [🔒 Security](security_and_encryption.md)

This pipe supports OpenRouter tool calling either via an internal execution pipeline (pipe-run tools) or via Open WebUI pass-through (OWUI-run tools). Tool sources and integrations:

- Open WebUI tool registry tools (server-side Python tools).
- Open WebUI **Direct Tool Servers** (client-side OpenAPI tools executed in the browser via Socket.IO).
- OpenRouter web-search (attached as an `openrouter:web_search` server tool in the `tools` array — not a `plugins` entry or a function tool).

---

## Tool backends (`TOOL_EXECUTION_MODE`)

This pipe supports two tool execution backends. Choose based on whether you want the pipe to run tools itself, or you want Open WebUI to run them.

### `Pipeline` (default)

The pipe runs the tool loop itself:

- Provider returns `function_call` items.
- The pipe executes those tools (Open WebUI registry tools + Direct Tool Servers where available).
- The pipe appends `function_call_output` items and re-calls the provider until the model stops requesting tools or `MAX_FUNCTION_CALL_LOOPS` is reached (at which point the model gets a synthesis turn, and each call abandoned at the cap is shown as a **failed** card).

**You gain:**
- Pipe-level concurrency controls, batching, timeouts, and breaker protections around tool execution.
- Its own copy of every tool round it runs, holding the full arguments and result, so the model learns on later turns which tools it used even with tool cards off (except in a temporary chat, for which the pipe keeps nothing); `PERSIST_TOOL_RESULTS` decides whether later turns get those results or a placeholder (a round that arrived before the chat's first turn is earlier too), and `TOOL_OUTPUT_RETENTION_TURNS` shortens long results from older turns.
- Optional strictification of tool schemas (`ENABLE_STRICT_TOOL_CALLING`) for more predictable function calling.

**You lose / trade off:**
- Tool execution behavior is “owned” by the pipe rather than Open WebUI’s native tool runner (so Open WebUI UX/logs may not exactly match the built-in tool flow).

### `Open-WebUI` (tool bypass / pass-through)

The pipe does **not** execute tools. Instead, it returns tool calls in an OpenAI-compatible `tool_calls` shape and expects Open WebUI to:

- execute tools locally (registry tools and/or Direct Tool Servers), and then
- replay tool outputs back through the pipe as `role:"tool"` messages on the next request.

**You gain:**
- Open WebUI-native tool execution behavior and UI (tool boxes, retries, and tool server flows are handled by OWUI).
- A simpler “adapter-only” path: the pipe focuses on transport translation between Open WebUI and OpenRouter.
- Better compatibility with OpenRouter streaming quirks: OpenRouter `/responses` can emit tool calls with `arguments:""` early; in this mode the pipe will **never** emit `arguments:""` to Open WebUI (it waits for complete args or normalizes to `{}`).

**You lose / trade off:**
- The pipe does not run tool batching, tool timeouts or tool breakers; Open WebUI’s behavior governs execution. The per-user request breaker still applies.
- The pipe runs none of these tools, so it keeps no copy of their rounds; Open WebUI's saved tool cards carry them. `PERSIST_TOOL_RESULTS` still decides what the model is handed on later turns: off, a result from an earlier turn -- or from a round that arrived before the chat's first turn -- reaches the model as a placeholder; on, in full. OpenRouter's own server tools run on OpenRouter in either mode, and the pipe keeps their rounds exactly as in `Pipeline` mode. In a temporary chat those rounds, and the reply's thinking unless `PERSIST_REASONING_TOKENS` is `disabled`, are held only in memory in a streamed reply, for that reply's calls back after each round of tool calls, and dropped when the pipe answers its last call back, or when the provider refuses a call-back the pipe was waiting for, or when the reply is stopped, or after 15 minutes unused (see [Persistence](persistence_encryption_and_storage.md)).
- In pass-through, the pipe does not strictify or mutate tool schemas; Open WebUI’s schemas are forwarded as-is.
- A caller's own tool is offered whatever the model's catalogue says about tool use, exactly as Open WebUI forwards it; a stripped tool is only one Open WebUI or this pipe added.

Every call reaches Open WebUI, including one without arguments or with `null` or malformed ones, and Open WebUI answers it as its own loop does; on `/responses` missing arguments arrive as `{}` and `null` ones as `null`, as Open WebUI's own Responses connection reads them.

---

## Tool schema assembly

Tool *schemas* are assembled by the tool registry builder and attached to the outgoing Responses request as `tools`.

### Preconditions

- A function tool reaches OpenRouter only if the model's catalogue row lists `tools` or `tool_choice`, publishes no parameters, or does not exist, **except** for a tool the request itself declared, which is offered whatever the row says. This holds in every mode, whatever the tool's **origin**, and for Fusion panel models; a model the catalogue rules out for tool use is sent none of the tools Open WebUI or this pipe added — no resolved tool, no direct tool, no OpenRouter server tool. A tool the request itself declared is the caller's business and goes out as sent, and a `tool_choice`, `parallel_tool_calls` or `stop_server_tools_when` left with nothing to point at is cleared. The general rule is the same whatever removed the tool: a `stop_server_tools_when` cap reaches OpenRouter only while at least one `openrouter:`-prefixed server tool is still on the request, so a cap whose every tool was switched off or stripped is dropped rather than sent pointing at nothing.
- The same row decides the per-chat Image Generation and Web Tools switches: a model whose tool use is ruled out is offered neither, so the switch and the request-time guarantee cannot disagree.

### Tool sources (in order)

1. **Open WebUI tool registry** (`__tools__` dict)
   - Converted to OpenAI tool specs (`{"type":"function","name",...}`) by `_build_collision_safe_tool_specs_and_registry` (called from `requests/orchestrator.py`), which builds each registry entry with `_responses_spec_from_owui_tool_cfg`.
   - When `TOOL_EXECUTION_MODE="Pipeline"` and `ENABLE_STRICT_TOOL_CALLING=true`, each tool schema is strictified (with the valve off the schema is not strictified and is sent with `strict: false`, and the executor filters a call's arguments against the names that untouched schema declares, with a root carried by `$ref` or `allOf` resolved for that purpose only; a root that is not an object and is not advertised under the pipe's `value` envelope declares no names, so nothing is delivered for it):
     - Object nodes get `additionalProperties: false`.
     - All declared properties are marked required; a property that was not explicitly required becomes nullable, in whichever of the two shapes it was written in. A property that states a `type` gains `"null"` to it; a property that states none — a bare `$ref`, an `allOf` that unwrapped into one, a typeless `anyOf`/`oneOf` — gains a `{"type": "null"}` branch instead, the whole node becoming the first branch of the resulting `anyOf` (never a `type` beside the `$ref` or the composition, which would constrain rather than union). A node that already admits null is left alone, and the decision is made on the node's post-`allOf` form.
     - `$ref` and `$defs`/`definitions` are preserved: referenced definitions are strictified in place, a required `$ref` node passes through untouched, and single-`$ref` `allOf` wrappers are unwrapped. An *optional* `$ref` property is the one shape that is rewritten: it is wrapped in an `anyOf` whose second branch is `{"type": "null"}` so the model can leave it out, and its `$ref` and `$defs`/`definitions` survive inside that first branch. A node's own keys survive the unwrap, so a property keeps its `title` and `description` beside the `$ref` it resolves through, and a definition a nested node carries is hoisted into the root container under the same key — so `#/$defs/X` and `#/definitions/X` keep their spelling — and strictified there. A name the root already defines is the root's, and the nested body is hoisted beside it under a fresh name, with every pointer inside that same nested container following it. Multi-branch `allOf` is merged (local `$ref` branches are resolved from the schema's own definitions); unresolvable references leave the `allOf` untouched.
     - Keywords OpenAI strict mode rejects are stripped (`default`, `$schema`, `pattern`, length/numeric/array constraint keywords, `format`). The strip is final rather than positional: it runs again after each `allOf` is inlined, so a key that rode in on a branch is not grafted back on, whatever shape carried it and whatever route it arrived by.
     - A missing `type` is filled in so schemas remain valid, at all three places one can be missing: a property, an array's `items`, and an `anyOf`/`oneOf` branch. A node there whose values are pinned by `enum` or `const` and which states no `type` of its own is given the JSON type those values have (`string`, `integer`, `number`, `boolean` or `null`), so a schema stops shipping a sealed empty object that admits no value; a node that pins no values is still given `object`, or `array` where it declares `items`. A heterogeneous (`["a", 1]`), empty (`[]`) or non-scalar (`[{"k": "v"}]`) `enum` pins no single JSON type and keeps the `object` default.
     - If a schema cannot be serialized for strictification, it is sent unmodified (with a warning logged).
     - A small LRU cache (size 512) avoids repeated strictification work for identical schemas.
     - The advertised schema is what the model is sent and the names the executor filters the tool's arguments by are its own top-level property names, so the two name sets cannot drift apart: a tool whose root schema is not an object root is advertised as a single `value` property and its argument is delivered under that name, and arguments the model invents beyond those names are dropped before the tool runs. With the valve off the executor filters by the names the untouched schema declares, resolving a `$ref`/`allOf` root for that purpose only; a root that names no properties is not a filter that keeps everything, it is a filter that keeps nothing.
     - A tool the pipe advertises on the Responses route also carries `strict: true`, so the provider enforces the strictified schema; a tool the pipe does not strictify (Open WebUI tool mode, a hand-back, or the valve off) is sent there with an explicit `strict: false`, so that endpoint's own `true` default never applies to a schema the pipe did not strictify (except where the schema is free-form and cannot be made strict: an array whose `items` node declares no properties is left open, because sealing it could only be satisfied by an empty object); a provider that does not support strict tool calling rejects the request.

2. **Open WebUI Direct Tool Servers** (direct entries in `__metadata__["tools"]`)
   - These are user-configured OpenAPI tool servers that Open WebUI executes client-side.
   - Before calling the pipe, Open WebUI resolves the tools of the servers selected for the chat and adds each tool to `__metadata__["tools"]` as a direct entry carrying its spec and its server, and writes the same specs into the request's own `form_data["tools"]`. The pipe builds its browser-run tools from those entries, so a tool Open WebUI withholds stays withheld (for example, the shell tools of a personal Open Terminal while no shell is connected).
   - This pipe:
     - advertises the tools to the model under the names Open WebUI gives them (OpenAPI `operationId` values). Open WebUI keeps one tool per name, so a direct tool replaces a same-named tool from another source before the pipe sees it; when names collide among the tools the pipe receives, it disambiguates them with a source prefix (e.g. `direct__`), and
     - executes tool calls via the Socket.IO bridge (`__event_call__`) by emitting `execute:tool` so the browser performs the request.
     - filters the call it sends to the browser to the names the pipe advertised for that tool, which with `ENABLE_STRICT_TOOL_CALLING` off are the spec's own, since the pipe then does not strictify.

   The prefix is a provider-facing label only: the Open-WebUI hand-back reads the tool's origin name. The strip that withholds the pipe's own tools from a model the catalogue rules out for tool use matches on that same origin, whatever name the provider saw, so a renamed tool is withheld as surely as an un-renamed one. The tool card does not read the origin: with tool cards on (the default) it is named as the model saw the tool, prefix and all, because that is the name the call was made under and the name a replay pairs with; the one card published with cards off (a file shown inline through Open Terminal) is named with the origin instead. The registry the loop runs against is keyed by the advertised name, so the pipe runs a call under the name the model saw. Citation routing requires the registry entry to be one of Open WebUI's own builtins — a `type` of `builtin` with a `builtin:`-prefixed `tool_id`. That marker, not the name, is the whole test: a direct tool, an MCP tool or a user custom tool that merely carries a builtin's name never has it, whichever name the provider saw. A request-supplied tool is the one case that can carry it, and only because the executor it resolves to *is* Open WebUI's own builtin.
   - Direct tools are only advertised when `__event_call__` is available; without an active Socket.IO session there is no safe execution path, so the pipe skips them. A direct tool is therefore withheld from any request that carries no browser session — an API call, an automation, a background task — on the request-spec side as well as on the registry side.

3. **Extra tools** (`extra_tools`)
   - A caller-provided list of already OpenAI-format tool specs is offered as they arrive: in Pipeline mode an extra tool is offered only when a tool of that name can run it; a spec whose name several registry entries share is left out, as on the request route (non-dict entries are ignored).
   - An `extra_tools` entry is a pipe-local extension: it is a schema only, and in ask mode (`tool_approval_mode: "ask"`, or `function_calling: "legacy"`) the round is handed to Open WebUI, which has no such field and cannot run one. Neither side can run it, so none is advertised there — the same reason the request's own tools are withheld, and for the same reason: a name offered to a model nobody will answer for is a dead end.
   - The `Tool ingest: … extra_tools=%d` debug line counts what was read off the request body on the way into the builder, not what went out on the wire. In ask mode it reads `0` by design, and that is correct; a `0` there in any other mode is the real signal that the body carried none.

### Duplicates and collisions

Two candidates that resolve to the *same* executor are advertised once, under the origin name, first candidate winning. Two candidates with *different* executors, or with none at all, both go out and take a source prefix plus a digest so neither loses its name; a replayed call naming a shared origin is rewritten to the name the model actually saw.

---

## Tool execution lifecycle (Responses API loop)

Tool execution happens in the request loop that follows each Responses API call:

1. The pipe calls the provider (streaming mode for normal chats).
2. When a `response.completed` event arrives, the pipe inspects the response `output` list.
3. Any `output` items with `type == "function_call"` are treated as tool calls to execute locally.
4. The pipe executes the tools and converts each result into `function_call_output` items.
5. The `function_call` items (normalized) and their outputs are appended to the next request’s `input[]`, and the loop continues until either:
   - no more `function_call` items are returned, or
   - `MAX_FUNCTION_CALL_LOOPS` is reached — pending tool calls receive stub responses and the model gets one additional turn after them to synthesize a final answer, so a turn that reaches the cap bills `MAX_FUNCTION_CALL_LOOPS + 2` model requests. Each abandoned call is also shown to the person as a **failed** card, the shape a genuinely failed tool produces, when tool cards are on.

Notes:

- A round that names an offered tool the pipe has nothing to run behind goes back whole after exactly one upstream request: an API caller gets it back directly, and a streamed saved chat hands it to Open WebUI. While the hand-back stands, the pipe never answers `Tool not found` for a name the request itself offered; once the hand-back budget is spent, a caller's own advertised schema with nothing behind it is answered `Tool not found` like any other name the pipe cannot run, because at that point the pipe is adjudicating calls in its own loop rather than sending them back.
- A reply's hand-back budget is charged only against that reply, and only a reply Open WebUI can re-ask is charged at all. The budget belongs to that user's spend as well as that reply: two users naming the same `chat_id` and `message_id` are charged to two budgets, and neither user's turns change the other's. A request that supplies both a `chat_id` and a `message_id` spends one turn of that reply's budget per turn, and once that budget is spent the pipe stops handing back and runs the remaining round itself, against the same tools it advertised. The exception is a round of browser-run tools and Open WebUI's own builtins: those stay out of that registry and only Open WebUI runs them, so such a round is handed back whatever the budget says and the reply is bounded by Open WebUI's own `CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS` instead — which is the point, since the pipe answering such a call is what would produce `Tool not found` for a name the request itself offered. A round mixing one of those with a call nobody can run falls to the pipe's cap, because a call with nothing behind it is not Open WebUI's to run either. Anything else — an OpenAI-compatible API caller, a caller that supplies only one of the two ids, any internal Fusion member, and Open WebUI's own task requests such as follow-up and title generation on the same chat — is handed back once per request, so those requests no longer share a budget with each other or with a live reply. The budget ends when a turn ends the reply; a turn the person continues from — a Continue, or a tool prompt they answered — is a further turn of that same reply rather than the end of it, so it carries the budget on rather than resetting it. The pipe keeps a budget for at least as many replies as it admits requests at once (`MAX_CONCURRENT_REQUESTS`), so a reply in flight is never the one whose budget is dropped to make room. A temporary chat is charged as well, under a key whose chat-id slot is blank: its id is the browser's socket id, and the budget outlives the reply in a process-lifetime map. The blanking costs the reply nothing — `message_id` is still what tells two replies of one temporary chat apart, so the budget stays per-reply and is never a shared one.
- A name nobody offered is answered `Tool not found` inside the loop.
- A schema that will be handed back is never strictified, and it keeps the fields the request put on it (`cache_control`, `strict`). A caller's tool the pipe runs has its `strict` taken from the request - unless the valve is on and passthrough is off, in which case the pipe strictifies the schema and advertises `strict: true` itself. A tool that arrived in Chat Completions shape states its strictness explicitly, `false` when the request named no `strict`, because `/responses` reads an absent one as `true`; that stamp returns on the chat leg too, nested under `function`, where the endpoint already defaults to `false`. A tool that arrived in Responses shape is passed through with exactly the keys it arrived with.
- A tool receives only the arguments **the schema the pipe advertises** declares, as in Open WebUI's own tool loop: anything else the model sends is dropped before the tool runs, so it can never replace what Open WebUI bound into the tool (such as the user a built-in tool acts for).
- For a browser-run (direct) tool the argument filter is the only thing the model's arguments can influence: the operation and the server it reaches come from the tool's own registry entry, are captured in the closure at `tools/tool_executor.py:832-843`, and are never taken from the model's arguments — so the model's arguments cannot point the call at another operation or server.
- Before each call the pipe hands the tool the chat's messages and files as the current request carries them (`__messages__`, `__files__`), as Open WebUI's own loop does; a Fusion panel model's tools see the person's chat.
- The pipe does not “stream” tool outputs mid-request. Tools are executed between Responses calls.
- `MAX_FUNCTION_CALL_LOOPS` applies whenever the pipe runs the calls, whatever the mode: a non-streamed reply, a Fusion panel model, a tool a request declared with nothing behind it, and every call except the ones Open WebUI runs under 'ask' approval, legacy function calling, or a model it holds back. Where Open WebUI runs the calls, it applies the cap of its own.

---

## Adaptive tool output budgeting

This section documents the dynamic context-budget guard. It runs on every request, in either tool execution mode; in `Pipeline` mode it also budgets each round's new results before the next call.

### Problem users observe

In long tool loops, the request can become context-saturated (large replayed artifacts + new tool outputs + reasoning state). A common symptom is:

- tool loops continue, but the model eventually returns no useful assistant text (or an incomplete response) because the prompt budget is exhausted.

### Conceptual fix

The pipe now applies **adaptive, model-aware budgeting** instead of fixed output caps:

- It derives prompt limits from model metadata: `max_prompt_tokens` when the catalog publishes it, otherwise the model's `context_length` less whatever reply allowance the request itself carries, with safe fallbacks, and a routing variant such as `base:nitro` resolves through its base's row. The provider's largest possible completion is not reserved — that is a ceiling the request never asked for, and on most of the catalog it put the budget far below the real window.
- It estimates request/input size and omits oversized `function_call_output` payloads by replacing them with a short model-visible stub that advises the model to retry with a narrower query.
- The model retains full tool access throughout the conversation and can recover from oversized results by retrying with tighter parameters.

This keeps the loop alive, informs the model in-band, and lets the model decide whether to summarize, stop tools, or ask for narrower tool queries.

### User-visible behavior changes

- Some tool outputs may be replaced by an omission stub in the request sent to the model, when the full text would exceed the remaining context budget for that turn.
- The stub replaces the result only in what the model is sent, for the rest of that turn. A shown tool card and the saved message keep the full text, and a warning notification names the tools the model did not receive.
- On later turns the request is rebuilt and budgeted again. With `PERSIST_TOOL_RESULTS` on, an omitted result is handed over in full once it fits -- the context has room, or the chat moves to a larger model; with it off, an earlier turn's result -- or one from a round that arrived before the chat's first turn -- reaches the model only as a placeholder anyway.
- If tool loops complete without any assistant content growth and no actionable continuation remains, the pipe emits a fallback assistant message instead of staying silent.

### Operator guidance

To reduce omissions and improve reliability:

- Prefer tools that support tight server-side limits (`limit`, `top_k`, date ranges, filters).
- Have tools return concise summaries plus references/IDs instead of full raw blobs.
- For bulky outputs (search results, logs, traces), expose pagination/continuation parameters so the model can request smaller chunks.
- `PERSIST_TOOL_RESULTS` is off by default to keep long conversations lean; enable it (site-wide or per user) when chats need to reuse exact raw tool outputs on later turns instead of re-fetching.

---

## Tool execution cards (`SHOW_TOOL_CARDS`)

When `SHOW_TOOL_CARDS` is on (the default), each tool the model uses appears in the chat as a collapsible card, exactly as Open WebUI shows a tool it runs itself:

- **In-progress cards** appear in a streamed reply when a tool starts, showing its name and arguments. A reply that is not streamed shows its cards when it ends. In a streamed reply, Open WebUI's `ask_user` card stays hidden while its question is open, as Open WebUI keeps it, and shows as soon as the answer arrives; a reply that is not streamed shows it when the reply ends.
- **Completed cards** show the result once the tool, and every call before it in its round, has finished, except a call the pipe refused before running it, which is shown as soon as the refusal is made. A call that failed or timed out keeps its real status rather than being recorded as a success. Where its text alone would not show the failure, the text opens with `Error: the tool call did not complete.`; that judgement of the text is Open WebUI's own classifier, or the pipe's copy of it when that module could not be imported so that later turns still read the call as failed. As in Open WebUI's own tool loop, a picture that a tool call returns as image data goes only to the model, with that call's result, and is gated first — scheme, plain `http://`, inline size, and the address an `http(s)` link names — so a link the pipe cannot use is never forwarded and is named in the chat on the turn that round is answered on; a picture Open WebUI has stored as a file, such as an MCP tool's, goes to the model with that result and stays in the chat, as Open WebUI does since its fix after 0.11.4; the call's other files go only to the chat, each stored under a name built from the content type the tool answered with, and falling back to `tool_result.bin` for a type nothing can name. A file the pipe could not store — in a temporary chat, one over `BASE64_MAX_SIZE_MB`, an undecodable payload, or a storage refusal — is not shown, and a warning names the tool that returned it, so a file that vanishes is never mistaken for a tool that returned nothing. The card shows the tool's text.

A shown card is saved with the message and reappears when the chat is reopened, because Open WebUI draws a card for every tool call saved in a message. That holds whether or not the reply was streamed, and when a reply ends in an error: a reply that is not streamed hands Open WebUI the same record a streamed one publishes. The exception is a reply that is not streamed and is stopped before it ends: it keeps no cards, because Open WebUI then receives no reply to save. Open WebUI also hands a saved round back to the model on later turns, and the pipe then sends no second copy of it. In a streamed reply, Stop keeps the calls before the first one still running: Open WebUI saves their cards as finished and hands them back, and with cards off the pipe's own copy does. The pipe writes a round's calls when the round starts and each result when its call returns, and the round comes back to the model in call order on that turn. A call the pipe refused before running it, such as one naming an unknown tool, is answered at once and on the turn it is answered is handed to the model in the order the round asked for it, because its answer is produced before any queued call has run; a round stored with cards on is written as the calls are answered, so a Continue replays that refused call's answer ahead of the results that were still running; its card is shown as soon as it is answered. A call the model sends malformed is taken as Open WebUI's own tool loop takes it on the same route, and is kept with its card like any other: missing or blank arguments, and `null` on `/chat/completions`, count as `{}`; `null` on `/responses`, or any value that is not an object, is answered that the arguments must be a JSON object; arguments that are not JSON are read as a Python literal, or answered that they could not be parsed; several objects sent back to back run as one call each; a call without a name is not kept anywhere, and the model is told nothing about it. Files and embeds a tool returns travel on its card; with no card they appear with the message instead, so they reach the chat once either way.

With `SHOW_TOOL_CARDS` off, no card appears at any point: not while the answer streams, not after it finishes, and not on reload. That is only possible by keeping the tool round out of the saved message, since Open WebUI would otherwise draw it. The model still learns on its next turn which tools it used: the pipe keeps its own copy of each round, written as the round runs, and hands it back, so the model can tell an answer it looked up from one it made up. A temporary chat is the exception: the pipe keeps nothing for it, so with cards off the model does not learn of earlier rounds. The copy holds the full call and result, as a shown card does; `PERSIST_TOOL_RESULTS` decides whether a later turn gets them or a placeholder, exactly as it does for a round that Open WebUI hands back, and a picture comes back in the same place, in Open WebUI's own images message. One card is kept all the same. Open WebUI draws a terminal file inline only from its saved `display_file` card, so for a person whose Open WebUI shows terminal files inline, the pipe keeps the card of a file the model shows through Open Terminal. Whether terminal files show inline comes from that person's own interface setting, or from the site's default when they have none. For everyone else, a file the model shows inline (`display_file` with `inline`) opens in the preview panel instead. Tools that internal Fusion's panel models run get no card, no preview and no file-browser refresh, whatever this setting says. What a tool returns feeds only the answer of the panel model that ran it, where a file it returns shows as a link. A link found in a tool result is **not** listed among the reply's sources: only a link from one of Open WebUI's own five citing tools, from the model's own `url_citation` annotation, or from a Fusion item is — and the five citing tools' own sources are withheld too while that model's `Citations` box is unticked, which is the same verdict Open WebUI reaches from the same model row. An MCP or tool-server web tool therefore keeps no citation chip, and there is no setting to bring the chip back. The pipe is here stricter than Open WebUI in one way: by tool identity. Open WebUI's own gate is by tool *name*, so any tool carrying a builtin's name would get a chip there — a tool-server entry named `fetch_url` does, and a custom tool named `fetch_url` displaces the real builtin in Open WebUI's registry and is cited in its place. On the capability the two now agree, and neither gates the provider's own annotations. The model still receives the tool's full result, and can name the link in its answer.

This setting is available as both an admin valve and a user valve (users can override the admin default).

**Scope:** cards for tools the pipe runs apply only when `TOOL_EXECUTION_MODE="Pipeline"`; in `Open-WebUI` mode Open WebUI runs those tools and draws its own cards, whatever this setting says. Cards for OpenRouter's server tools (web search, fetch, datetime, advisor, subagent) follow this setting in both modes.

---

## Concurrency, batching, and timeouts (per request)

Tools are executed via a per-request worker pool backed by a bounded queue:

- Queue size: 50 batches per request (bounded).
- Worker count: `MAX_PARALLEL_TOOLS_PER_REQUEST`, started when the request's first tool call is queued, so a turn that calls no tool starts none.
- Per-request semaphore: limits concurrent tool executions per request.
- Global semaphore: `MAX_PARALLEL_TOOLS_GLOBAL` limits tool executions across all requests.
- Plain-function thread pool: a tool written as a plain function (rather than `async def`) is offloaded to a pool the pipe owns, `min(MAX_PARALLEL_TOOLS_GLOBAL, 8)` threads wide. The semaphores above are the gate that decides which calls are admitted; the pool is the isolation boundary that keeps an admitted blocking tool from taking the threads Open WebUI's own requests and this pipe's storage work use. Raising the valve above 8 therefore admits more calls without adding plain-function threads.
- Open WebUI's built-in `ask_user` takes no slot from either semaphore, because it waits on a person rather than doing work.

Batching behavior:

- The pipe groups a response's tool calls into batches before any of them runs. Consecutive calls join one batch, up to `TOOL_BATCH_CAP` calls, when they name the same tool -- resolved the way the registry and the tool breaker resolve it, so a name the model padded with surrounding whitespace is the same tool -- when neither the joining call nor any call already in the batch carries a dependency or ordering blocker in its arguments, and when the joining call names the call ID of no call already in the batch. The cap is spent per batch, so one spelling's worth of padding costs no second `TOOL_BATCH_CAP` slot, no second worker and no second batch deadline. A call refused before queueing (an unknown tool, invalid arguments or a tripped breaker) does not break a run of consecutive calls.
- A call whose arguments include any of `depends_on`, `_depends_on`, `sequential` or `no_batch` is never batched. These keys only keep the call out of a batch; they do not make it wait for other calls.
- Batching does not require identical arguments and never deduplicates calls. It does not raise concurrency either: every call in a batch except `ask_user` still waits for a per-request slot and a global slot, and all calls in a batch share one batch deadline.
- Each batch is queued separately, so while slots are free, a slow call never holds up a call to another tool, and a response's calls start together as long as there are free slots for all of them. Each call's result is handed back as soon as that call finishes, even while other calls in its batch are still running. The same holds inside internal Fusion, where each model that calls a tool gets as many tool workers as the chat request it answers, and all of those models share that request's slots.

Timeouts:

- Each tool call has a per-call timeout (`TOOL_TIMEOUT_SECONDS`), measured from when the call starts running, not from when it starts waiting for a slot. When it expires the pipe stops waiting, the model reads `Tool '<name>' timed out after <N>s.`, and the timeout counts toward that tool's breaker. An async tool is cancelled; a tool written as a plain function is not, and runs to its end on a thread of the pipe's own tool pool, whose width is `min(MAX_PARALLEL_TOOLS_GLOBAL, 8)`.
- Each tool call runs exactly once. A tool that raises an error is never retried automatically, because tools can have side effects (an MCP tool that sends an email must not fire twice). The failure is reported to the model in Open WebUI's own form — a JSON object on two lines, `{` / `  "error": "<the exception's message>"` / `}` — which is the same text the person reads on the tool card, so the two never disagree. When the exception carries no message, the class name is used instead.
- If Open WebUI has already closed an MCP tool's client connection, the tool reports "no longer available in this session" rather than a raw error, and this does not count against its breaker.
- Calls grouped into one batch share a batch deadline (`TOOL_BATCH_TIMEOUT_SECONDS`, never shorter than the per-call timeout). Its clock starts when a worker picks the batch up. It is also the ceiling on the time one response may take putting its calls on the queue: a round whose calls outnumber the free workers ends at that ceiling, and the calls it never started are reported to the model as not started rather than as idle timeouts. When the deadline passes, finished calls keep their results; every call still running or still waiting is cancelled and reported as `Tool batch '<name>' exceeded <N>s and was cancelled.` The round then ends as soon as those cancelled calls have had five seconds to acknowledge the cancellation; one that has not is dropped from the round, keeps running, and its result never reaches the model. Of the cancelled calls, only the running ones count toward the tool's breaker, except any that `TOOL_IDLE_TIMEOUT_SECONDS` had already given up on.
- `TOOL_IDLE_TIMEOUT_SECONDS` (unset by default) caps how long the pipe waits in total for one response's tool results, counted once from when the model asked. Every call that had started and whose result has not arrived by then is reported as timed out, however long it has been running. When that time passes, a call that was already running reads `Tool '<name>' timed out after <N>s (idle timeout).`, and one still waiting for a tool worker or a free slot reads `Tool '<name>' was not started: it was still waiting for a tool worker or a free slot <N>s after the round asked for it (idle wait).` Giving up this way does not count toward the tool's breaker. A call that is already running is not stopped: it keeps running and holds its slot until it finishes, another limit ends it, or request cleanup cancels it after `TOOL_SHUTDOWN_TIMEOUT_SECONDS`. The model never receives the late result. Outside internal Fusion, files or embeds the call returns still appear in the chat: a streamed reply waits for the call, bounded by `TOOL_SHUTDOWN_TIMEOUT_SECONDS`, so a call that returns during that wait is read before the reply ends. When they do appear, a file the call shows through Open Terminal opens in the preview panel, or nowhere for a person whose Open WebUI shows terminal files inline: their browser ignores the preview request, and the call's card has already closed as timed out. The call's own later error or per-call timeout still counts toward the tool's breaker, and a later success clears the count. A call still waiting for a slot or a worker never starts. This limit is not what bounds getting calls started: that is bounded by `TOOL_BATCH_TIMEOUT_SECONDS` as a ceiling on the round's own queueing. The tool workers remain for the whole request, except inside internal Fusion, where a model's workers and its running calls are cancelled as soon as that model's answer ends, without the `TOOL_SHUTDOWN_TIMEOUT_SECONDS` wait.
- Open WebUI's built-in `ask_user` -- the built-in specifically, identified by the tool behind it and not by the name it is advertised under -- keeps its question open for the time the model asked for, as normalised by Open WebUI. Its per-call limit becomes that time plus 15 seconds, and the batch deadline and the idle limit are raised to at least that long. Since it takes no tool slot, its question does not wait for other requests' tools, though it still needs one of its own request's workers to be free. Its timeout does not count toward the breaker. It runs alone: an `ask_user` call mixed with other calls, or repeated in the same response, is refused with Open WebUI's error text, and the other calls run. That "only tool call" rule is checked first, so a malformed `ask_user` that shares its turn with another call is refused with Open WebUI's own "runs alone" text, not with the pipe's argument text. The rule is Open WebUI's, applied to the name as the model sent it, so a spelling Open WebUI would not recognise as the built-in — a padded `" ask_user "`, or a collision-renamed `owui__ask_user` — does not get the rule: the pipe still resolves that call to the built-in and runs it, and the built-in resolves that name the same way (see the padding rule below). A call whose arguments Open WebUI's own normaliser rejects never reaches the builtin: alone in its turn it is refused with `Error: <Open WebUI's own message>`, and it neither counts toward the breaker nor starts a worker, so ask_user stays usable.

---

## Breakers (stability controls)

Three per-user breakers share `BREAKER_MAX_FAILURES` and `BREAKER_WINDOW_SECONDS`; each internal Fusion run also keeps one count per tool, shared by all of its models, which uses `BREAKER_MAX_FAILURES` but not the window. The per-user breakers count failures in two ways:

- **Per-user request breaker:** counts each failed chat call to OpenRouter within the trailing `BREAKER_WINDOW_SECONDS`, whether the failure is an error reply; a connection that cannot be opened, drops or times out; an accepted body lost before it finished arriving; an error OpenRouter reports after accepting the call (an error event in the stream, or an error body); or a stream that stops before its final event. A request the pipe retries automatically counts once, however many tries `TRANSIENT_RETRY_MAX_ATTEMPTS` spent on it. A stream that ends as `response.incomplete` (for example, when the answer reaches its length limit) is a finished call, not a failure. A generation on a picture-only image model or a video model counts once, when it fails after being answered by OpenRouter, except a generation whose response arrived whole and was not an OpenRouter document — a body a proxy, CDN or WAF rewrote, or one that is not a JSON object at all, which is not a failure of the request and never counts toward the limit. A request that ends without an error clears the count, but a request to a picture-only image model or a video model clears it only once its result is delivered whole, and a stream that breaks after the finished image arrived clears nothing, and a request the user stops does not clear it. Housekeeping tasks such as title generation neither count nor clear, and a request refused before anything is sent (such as a missing API key, or a model the pipe will not serve) neither counts nor clears; Open WebUI's merge-responses task counts but never clears. At `BREAKER_MAX_FAILURES`, that user's new requests are refused with "Temporarily disabled due to repeated errors. Please retry later." until the oldest failures age out of the window, or until a request that is still let through ends without an error and clears the count: one already under way when the limit was reached, or an exempt one. A request is exempt, and never refused by this breaker, when its last message is a tool result, or is Open WebUI's own message handing the model a tool's images directly after a tool result. That is how Open WebUI calls back after running its tools to finish an answer already under way. Open WebUI's own message consists of its fixed sentence followed only by the images. A question the user types, or a picture they attach, right after a turn that ended on a tool result is a new request, which the breaker refuses like any other. Inside internal Fusion, the run counts once: one failure when no panel model answered, none when one did, however many of its panel, judge or final-answer calls failed at OpenRouter; when the run ends, the count, including that run's own failure, is cleared if the run finishes and any panel model answered, and kept if none did or the user stopped the run.
- **Per-user, per-tool breaker:** counts a tool's failures in a row, keyed by tool type and the tool name it was looked up by, so other tools of the same type, including the rest of an MCP server's tools, keep working. A name the model padded with surrounding whitespace is the same tool and spends the same budget. A successful call clears the count, whichever spelling of the name it was sent under, and so does a gap longer than `BREAKER_WINDOW_SECONDS` between the tool's last failure and its next call. The gap is measured to the next call, not between failures, so a slow tool that keeps timing out still trips. Errors, per-call timeouts, running calls cancelled by the batch deadline (other than calls the idle limit had already given up on), and calls whose tool server cannot be reached or answers with an HTTP error status count; an `ask_user` timeout, a call to an MCP tool whose session already closed, and a call still waiting for a slot do not. A `SystemExit`, `KeyboardInterrupt` or `GeneratorExit` out of a tool is not a failure of that tool: it signals the process rather than the tool, so it shows as failed on its card but neither adds to the count nor clears it, and the tool is not skipped because of it. A failure that the tool reports in a result it returns normally, such as an error message inside a successful response, shows as failed on its card but neither adds to the count nor clears it, whether the judgement is made by Open WebUI's own classifier or by the pipe's copy of it. Each internal Fusion run keeps one count per tool, shared by all of its models: a tool that fails `BREAKER_MAX_FAILURES` times in a row within the run is skipped from then on. Unlike the user's own count, this one is never cleared by a quiet spell, only by a success; once the tool is skipped, only an already-running call to it can supply that success. Calls to that tool inside the run neither raise nor clear the user's own count for it, and are not skipped because of that count.
- **Per-user DB breaker:** counts failed database reads, writes and deletes of stored reasoning, tool results and session logs within the trailing window. A failure to refill the artifact cache after a read that already succeeded is not a database failure and is not counted. A delete that faults counts here like any other database failure, which is what lets a database that has started refusing clean-ups open the breaker rather than being re-asked on every turn. A successful read or write clears the count; where Redis buffers writes, a write succeeds once Redis has taken it. Below the limit, a write the database refuses is announced to the person on the turn it happens, because the turn still completes and the loss is otherwise only visible on a later one. At the limit, the pipe skips that user's database reads and writes and shows the warning "DB ops skipped due to repeated errors." Chats still get answers, but that user's reasoning and tool results are neither saved to nor read from the database until failures age out of the window.

While a tool breaker is open, calls to that tool are skipped and the model is told why; outside internal Fusion, a best-effort status message is also sent to the UI, once per tool per round however many of its calls that round asked for. A turn whose tool calls are all skipped this way does not count as a failed request.

---

## OpenRouter web search server tool

The web-search integration is attached as a server tool (not as a `tools` function or legacy `plugins` entry):

- When the **OpenRouter Web Tools** toggle (the filter's `WEB_SEARCH` user valve) is enabled for the request (per chat, or enabled by default via the model’s Default Filters), the pipe appends `{"type": "openrouter:web_search", ...}` to `tools`. One entry per type per request: a type the request body already lists is not sent a second time, and the filter's parameters are the ones that go out. Image-output and video-generation models are excluded because they do not receive the Web Tools filter.
- Search parameters (max results, engine, allowed/excluded domains, etc.) are controlled by the OpenRouter Web Tools filter’s admin valves.

Important: Open WebUI also has a separate built-in **Web Search** toggle (Open WebUI-native). OpenRouter Web Tools and Open WebUI Web Search are different systems.
See: [Web Search: OWUI vs OpenRouter](web_search_owui_vs_openrouter_search.md).
See: [OpenRouter Server Tools](openrouter_server_tools.md) for the full server tools reference.

---

## OpenRouter response-healing plugin (intentionally not exposed)

OpenRouter offers a response-healing plugin that can attempt to repair malformed outputs. This pipe does **not** expose that plugin on purpose:

- We prefer failing fast when a model returns malformed JSON or invalid structured output.
- Silent repairs can hide real model issues (bad prompts, low token budgets, provider quirks) and make debugging harder.

If you want auto-healing, integrate it explicitly in your own request layer so it is visible and auditable.

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
