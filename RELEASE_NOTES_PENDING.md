# Release notes pending

## Behaviour changes

- **The pipe stops writing into Open WebUI's saved chats** — after an answer in a saved chat the pipe merged the reply's citations, file annotations and reasoning details into the stored message and wrote it back, and a Fusion reply and a tool-call limit wrote their panel and note the same way. Open WebUI's write rewrites the whole chat, and on a large chat the read and the write held the worker's event loop: 315 ms to read and 143 ms to write, 458 ms of hold in all, measured by the maintainers on a 123.8 MB synthetic chat on SQLite on 2026-10-16. Open WebUI saves the citations, the panel and the error the pipe publishes as events, so those writes were a second copy, and they are gone. Citations travel as the `source` events Open WebUI appends to the message's `sources`: on the dev box (Open WebUI 0.12.x, 2026-10-22) Open WebUI saved the same citations with the pipe's write removed, 20 of 20, 5 of 5, 3 of 3 and 5 of 5 across Perplexity, Exa web search, the web plugin and native search. The closing Fusion panel travels as an `embeds` event with `replace`, and the Fusion answer as the output item Open WebUI folds. In a saved chat the tool-call limit note rides the closing `chat:completion` frame as its `error`, the bare sentence Open WebUI stores for its own limit, which Open WebUI saves on the message and the browser shows as the message's error; a reply returned whole, and a streamed reply to a caller with no chat (an OpenAI client raises on a streamed chunk that carries an `error`), carry the note at the end of the answer text. The reads of a stored message that remain (the seeded output and the Fusion panel's embeds) take the one `chat_message` row by its primary key where Open WebUI has that table, and fall back to Open WebUI's getter, which loads every message of the chat, where it does not or the row is missing; a reviewer measured 204-302 ms with a 188 ms event-loop stall for the getter against 7.5 ms and a 5.6 ms stall for the row read, on a 300 MB chat with Open WebUI 0.12.2. With a bounded `Open WebUI stream buffer limit` (the default is unbounded), a citation or Fusion panel frame that a full buffer drops is missing from the saved message, because no end-of-turn write repairs it any more. The pipe no longer saves a copy of a reply's `reasoning_details` or `annotations` onto the message: reasoning stays in the pipe's own artifact rows, and a chat saved earlier keeps the copies it holds. Nothing in an old chat is rewritten. One write remains: a refused automation reply (a tripped circuit breaker, a failed warmup, a missing or full request queue, a failed request setup) still gets its `error` written to its message. Open WebUI's automation runner reads that field to decide whether a run failed, and its chat route stores an error only for an HTTP error response or a raise, while the pipe answers an automation reply with the card's text, so the pipe's write is what carries it.
- **A setting saved in Open WebUI while a model sync pass runs is no longer written over by the pass** - a pass read every model's row once, when it started, and built each model's write from that copy, so a system prompt, a parameter, a name, a profile picture or any other setting an admin saved in the model editor after the pass began and before the pass wrote that model was replaced by the value the pass had read; both writes succeeded, so neither was reported as a problem. A pass runs whenever the catalog's content or a tracked setting changes, and the icon downloads alone are allowed 45 seconds of it. Each model's write now reads that model's row again immediately before it builds the write and builds it from that read, so the write starts from the row as it is then; that is one single-row read per model on top of the pass's batched read. What the pass manages itself - the capability boxes, the filter attachments, the description and an icon it has just downloaded, each under its own sync switch - is laid over that row as before. An icon the pass keeps, because its stamped source is unchanged or because the catalogue did not answer, is no longer handed to the write, so the picture written is the one the row holds at that read; that also means a row that keeps its picture is no longer overwritten with the kept picture of another row sharing its catalogue icon, as a model and its variants do. The start-up removal of stale `openrouter_*` filter ids builds its write the same way, from a read of the row taken just before it, and skips a row that reads as absent. A setting saved in the gap between a row's read and Open WebUI's own write can still be overwritten.
- **The Fusion presets use OpenRouter's current `~…-latest` aliases** - the default preset, `general-high`, listed `~openai/gpt-latest`, an alias OpenRouter no longer lists, so on the internal engine the pipe refused that member before sending it, as not in its model list, and a default Fusion panel ran with two members instead of three. Every member of the three presets is now one of OpenRouter's self-updating `~…-latest` aliases: `general-high` is `~anthropic/claude-opus-latest`, `~openai/gpt-sol-latest` and `~google/gemini-pro-latest`, and `general-budget` and `general-fast` are `~google/gemini-flash-latest`, `~deepseek/deepseek-flash-latest` and `~openai/gpt-luna-latest`; the judges are unchanged. `~moonshotai/kimi-latest` left the budget and fast panels because its output price rose, and `deepseek/deepseek-v4-flash`, a fixed version, became the DeepSeek alias. The ids carry the `~`, and an entry written without it does not admit them, so a restricting `Model allowlist` has to name these with the `~`, as exact ids or as patterns such as `~deepseek/*`, or the members it leaves out fail with the reason shown.
- **The account's presets are read from OpenRouter's presets endpoints** - OpenRouter lists an account's presets in its `/models` answer only sometimes, and a catalog refresh that caught an answer without them dropped every preset from the model picker until a later refresh brought the rows back; a restart forced that refresh, and the pipe's re-fetch on a preset-less answer gave up after three preset-less answers in a row. The registry now serves the account's presets itself: each catalog refresh reads the documented presets list and, for every preset whose designated version it has not cached already, that version's record, and a preset the model list omits is served the way OpenRouter's own answer serves it - the base model's catalog row under the preset's id and name. A failed presets read (the list, or one preset's own record) leaves the picker alone: the presets the last catalog served stay served, whether the model list carried them or the pipe built them, a preset the last successful read found switched off stays out, one warning names the failure, and the next catalog call that lands inside the freshness window re-reads the presets on their own, without the model list, so a cold start whose first presets read failed recovers on the next request; a retry that fails again is tried again after a pause that grows with each failure. A preset whose status is anything but `active` (OpenRouter documents `disabled` and `archived`) is left out of the catalog, the row OpenRouter's own model list carries for it included, on a full refresh and on the presets-only retry alike; a preset the model list does not carry is also left out when it has no designated version, its configuration names no model or its base is not in the model list. The leave-taking is logged once per change, not on every refresh, and the line says the preset left the picker only when the last catalog served it. A preset's file upload follows its base model: the pipe judges the direct-upload blocklist on the model the preset's configuration names rather than on the preset's own `@preset/...` id, which no entry names, so a preset on a blocklisted base offers no file upload, for the row the pipe builds and for the row OpenRouter's own answer carries; a row the model list carries for a preset whose configuration could not be read is judged on its own id until the configuration is read. The re-fetch on a preset-less answer is removed with the behaviour it worked around; nothing else about the refresh moves: same interval and failure backoff, one zero-data-retention read, media-row carry-over as before.
- **The sync takes its own stale media defaults back** - a row that an earlier build's picture/video rule cleared carries `File context` and `Built-in tools` unticked by the pipe itself; on a model the current catalog counts as neither picture- nor video-producing, where the row's whole `builtinTools` entry is the pipe's `Files` seed, the sync now clears those two keys on the next refresh, leaves the row's other capability values alone, takes the seed with them and records the reclaim in the model's `builtin_tool_defaults` note. A model-editor save writes its own full tool object, so such a row is left untouched, and a box ticked after a reclaim stays ticked. A lone cleared box passes untouched.
- **The kept tool turns move in steps, so the provider's prompt cache keeps working** - `Tool turns to remember` used to move its oldest kept turn out on every turn, which changed the start of the request each time and left the cache covering little past the first turns. The kept span now widens by one turn at a time up to twice the setting minus one and is cut back to the setting every that-many-th turn, so between two cuts the request starts exactly where the previous one did. The tool-output shortening (`Tool output history depth`) steps by its own value the same way: its boundary moves every that-many-th turn, and between moves the results of the last up to twice that minus one turns, the one being answered included, go whole. Nothing else about what a request carries changes, and the setting's range and per-user override are as before.
- **The AI can search the tool calls of its chat's earlier turns** - a long chat sends only the last few turns' tool calls (`Tool turns to remember`), and with `Keep tool results across turns` off the model never sees an earlier result again; the data was stored all along. Three pipe-run tools - `search_tool_history` (keyword, exact-text or RE2 search over tool names, arguments and stored results), `list_tool_history` (every call, newest first) and `view_tool_call` (one call's result or arguments, in pages) - let the model look the rest up on request, with no index, no extra stored data and no new component on any database backend: the search works on the conversation the request already carries, plus the pipe's own copy of a round that copy lacks, read through the replay's own loader. The tools appear once a chat has an earlier tool round the current message no longer repeats, whatever the keep-results setting says; with it off, the note left in place of each withheld result names the call's `turn.position` id, and the search returns what was stored. They are offered where the pipe runs the calls itself - in the Open-WebUI tool mode Open WebUI runs them and would not know these names - and an admin or the person whose chat it is can switch the feature off, which returns the request to the previous behaviour exactly. A search or read is one more model round that re-sends the conversation; the ids are positions on the chat's current path and move when a message is edited or deleted or the chat switches branch.
- **Session log assembly is bounded for a turn of any size** — assembling a turn's archive held the whole turn in memory: every staged segment was fetched whole, merged into one list and sorted there, and a large turn's assembly alone could outgrow the worker, which is what produced the production worker deaths during session log assembly that this change answers. The assembly now orders a turn's records in an on-disk scratch database under `Session log directory`, reads every staged row in bounded chunks (on SQLite through incremental blob I/O, where a driver without it is named once in the log and its passes hold more memory; on PostgreSQL through repeated `substr` queries that each make the server read the whole stored value again, so a pass over one very large row takes longer there than on SQLite), and streams the archive members into the zip. An encrypted row is read as its still-encrypted token, copied to a temporary file in the session log directory, and the token's signature is verified over the whole of it before anything is decrypted; the records then leave that file through the decryptor, the decompressor and the parser in chunks. The scratch copy's record text is re-encrypted with a per-pass key when the turn's rows are encrypted, so no decrypted conversation text sits on disk in the clear. On Linux the scratch files have no name (each is either never linked or unlinked as soon as it is created), so a killed worker leaves none behind; on macOS the code creates each file and unlinks it at once. On Windows, where an open file cannot be unlinked, the copy of the encrypted row is deleted by the operating system when it is closed or its process ends and the ordering database, named `.session-log-sorter-` followed by random characters, is removed when the pass ends; a worker killed during a pass leaves that file, and the cleanup sweep removes it once it has sat for one hour or twice `Assembly lock stale timeout`, whichever is longer. If the directory cannot be created or written the pass fails and the turn's staged rows stay in the database. A long pass renews its assembly lock at its progress checks, once the smaller of 60 seconds and a tenth of `Assembly lock stale timeout` has passed since the last renewal, and a stopped or switched-off pass ends at its next progress check with its staged rows kept; the assembler starts no new pass once the stop is set or storage is off. A turn's segment is staged in pieces of at most 1,000 records, terminal piece last. A pass that reaches the turn's rows leaves a `session_log_attempt` row and removes it when it returns - a return whose cleanup fails is logged at WARNING and the row it leaves counts toward the three; a turn whose three most recent passes all left theirs behind - the mark of a worker that died or hung mid-pass - is dropped by the next pass with one WARNING naming the turn, ending the retry loop. A stored row whose signature or structure fails the check refuses the turn at once: its staged rows are deleted and one WARNING names it; a row that merely cannot be read right now (no key configured after a rotation, the row gone) keeps the turn's rows for a retry, as before, and a failure that appears while the artifact encryption key is being changed, after the store has moved to another table during the pass, or after the key changed and came back during the pass, is not read as damage either: those rows stay in the table the previous key named, are assembled again if that key returns, and are otherwise removed with the retired table; if the delete of a refused turn's rows fails, one WARNING says so and the next pass refuses them again. A staged segment the write-behind queue has carried through the full `Rejected write attempt limit` is deleted from the queue rather than dead-lettered, with one WARNING naming its chat and turn; every other row type keeps the dead letter. A `.zip.tmp` a killed writer left behind, and a `.session-log-sorter-*` file a killed Windows pass left behind, are swept once they have sat for one hour or twice `Assembly lock stale timeout`, whichever is longer. Measured once, on one machine, with a fresh Python 3.11 process per case, SQLite set up as Open WebUI sets it (WAL, orjson), PostgreSQL 16 through psycopg2, the real manager and artifact store, encrypted rows and synthetic log text: peak memory above the size at the start of the pass was 361 MB for a 300 MB turn of 103 time-overlapping segments on SQLite where the previous code peaked at 4,729 MB (9.0 s against 8.3 s), and for one 133 MB staged row 218 MB on SQLite (previous code 1,936 MB) and 45 MB on PostgreSQL (1,869 MB), where the pass took 8.7 s against 4.4 s; the archives' log members were byte-identical to the previous code's in those cases, and the session log storage guide lists the measurements. They show the size of the change on that machine and that text, not limits.
- **A round's tool calls run one after another, as in Open WebUI** — the pipe started a round's tool calls together, up to `MAX_PARALLEL_TOOLS_PER_REQUEST` (five by default) at a time, where Open WebUI runs the calls of a round one at a time, in the order the model asked for them, and runs only its sub-agent tool, `delegate_task`, side by side. The pipe now does the same for the calls it runs itself: in Pipeline mode and, in Open-WebUI mode, for the calls it still runs, such as those of a reply that is not streamed or of an internal Fusion model, on `/responses` and on `/chat/completions`, streamed or not. A call starts after the call before it has returned, failed, been skipped or cancelled, or been cut by a time limit, and the round's `delegate_task` calls start together once its other calls have ended; a round with several slow tools takes the sum of their times, not the longest. Results still go back to the model in the order it asked, and a card still fills in once its call and every call before it has finished; in a streamed reply, the cards of a round appear together when the model has named its calls, before the first of them starts. The time limits count from a call's own start, so a call that waited behind earlier calls still gets its whole `TOOL_TIMEOUT_SECONDS`, `TOOL_BATCH_TIMEOUT_SECONDS` and `TOOL_IDLE_TIMEOUT_SECONDS`, and no time limit ends a call that has not started. A call whose body keeps running after the pipe stopped waiting for it (a synchronous body on a pool thread, a body that ignores its cancellation, a call `TOOL_IDLE_TIMEOUT_SECONDS` gave up on) can still be running when the next call starts, and when running calls hold every slot the request has or every worker it has (at `MAX_PARALLEL_TOOLS_PER_REQUEST` `1` one does) the next call waits until one of them ends. Inside internal Fusion each model's calls run one after another while the models still run side by side and share the request's slots. `MAX_PARALLEL_TOOLS_PER_REQUEST` limits the `delegate_task` group, the calls of the internal Fusion models of one request and the calls that run on after the pipe stopped waiting for them. The `TOOL_BATCH_CAP` setting, which capped how many consecutive calls to one tool went into a batch, is removed because those calls are no longer grouped; a value saved for it is ignored, and the pipe logs a warning naming it (at most once in five minutes). In Open-WebUI tool mode a streamed reply's calls keep running in Open WebUI.
- **Web fetch defaults to Exa and caps the page it reads and the page it records** — the Web Tools filter's `WEB_FETCH_ENGINE` defaulted to `auto`, which on Anthropic's models means Anthropic's own fetch: it returns no fetch item, so the chat showed no card, the pipe kept no record, and a later turn did not know the page had been read. The default is now `exa`, which read both a PDF and a long page in a live check on 2026-10-17 (Claude Haiku 5.5, one PDF and one long web page, one run on each engine) and leaves a card and a record; OpenRouter's documentation of 2026-10-08 lists a fee of $1 per 1,000 fetches for it on top of the tokens the page costs, and the page address goes to Exa, as it already did under `auto` where the provider has no fetch of its own. `WEB_FETCH_MAX_CONTENT_TOKENS` changes from no limit to 30000, so a long page is cut before the model reads it (in that check OpenRouter's own engine returned 1,017,070 characters of raw HTML for the long page, 596,228 input tokens for a single question), and the pipe keeps at most 150,000 characters of any fetched page in the card and in its own copy: a longer page is cut at its end and the text ends with a note saying how many characters were left out, the same text in both copies. A valve an admin has set to a custom value stays as set, and a valve left on `Default` takes the new value when the pipe next rewrites the installed filter, which is the next model-list refresh while `AUTO_INSTALL_WEB_TOOLS_FILTER` is on; a filter the pipe does not update keeps the old defaults. An OpenRouter workspace that limits the fetch engines and leaves out Exa rejects a request that names it, so `WEB_FETCH_ENGINE` has to be set to an allowed engine there. The web fetch section of the server tools doc lists what each engine shows and keeps for later turns.
- **A web search on `/chat/completions` reaches later turns** — on the models the pipe sends to `/chat/completions`, OpenRouter reports a web search only as the cited pages with their excerpts (annotations on the answer) and a search count in the usage block, with no search item and no query (four requests to Claude Haiku 5.5 on 2026-10-17, streamed and not, with the tool and with the `web` plugin), so the chat showed no search card and the pipe kept nothing for a later turn to read. For a request that offered the web search tool, the pipe now builds one round from the citations: a card and its own copy that list the cited pages with their titles and excerpts, with the same limits and note as on `/responses`. The call has no query, because the route does not send it back (in a live check on 2026-10-17 a phrase placed in the query was read as the words searched in all three requests, two to Claude Haiku 5.5 and one to GPT-5-mini), and the result opens with a note saying the search ran with a query the model wrote that OpenRouter does not send back on this route; with `Keep tool results` off the model gets one note that says that and that the result is not kept. Streamed and whole replies are handled alike, and `Keep tool results` decides what a later turn gets, as for every other search. One round stands for the whole reply (the route cannot tell two searches apart), a reply with no citations leaves none, the first run of consecutive citation frames forms the round and lists those pages, and frames after that run reach the sources panel only. A reply to a request that offered no web search tool (a model that searches on its own, the `web` plugin, `:online`) is left as it was and its citations reach the sources panel. A stream that fails while the round's first citations are still arriving no longer shows them in the sources panel.
- **Tool calls from earlier turns keep their arguments when results are not kept** — with `Keep tool results` off, the default, an earlier turn's tool call reached the model with `{}` in place of its arguments and the line `[tool result not retained]`, so the model could not tell what it had asked for: in a live check (Claude Haiku 5.5, 2026-10-17) it said its search had no query and returned nothing, disowned an answer it had sourced from that search, and ran the search again in three of seven samples. The call now keeps its name and the arguments it was called with, for every tool and in both tool execution modes, and only the result is replaced, by a note saying whether the call succeeded or failed and that its result is not kept for later turns: `[This call succeeded. Its result is not kept for later turns.]` or `[This call failed. Its result is not kept for later turns.]`. A Tool Search `call_tool` call keeps its `params` too. From now on a later turn gets each earlier call with its arguments whatever `Keep tool results` is set to, and the setting governs results only. With `Keep tool results` on nothing changes.
- **A long chat no longer sends every tool call it ever made (`Tool turns to remember`)** — a chat sent every earlier tool call, with its arguments, on every request: a chat of 220 turns with 3,313 tool calls, measured on production on 2026-10-18, sent about 3.4 million characters (about 800,000 tokens) per round, almost all of it earlier calls. A new setting, `TOOL_TURNS_TO_REMEMBER` (shown to users as `Tool turns to remember`, from 0 to 100, default 5), counts the turns before the one being answered: the most recent ones keep their tool calls, and their results while `Keep tool results` is on, and an older turn that used tools goes back as conversation text only, the person's messages and the assistant's answers, with no tool call, no tool result, no OpenRouter server-tool round and none of the thinking stored for it, because a provider can refuse thinking that is sent without the calls it belongs to. The turn being answered always goes whole, with every round it has made so far; `0` sends no tool call from any earlier turn. Two things are never cut: Open WebUI's built-in `ask_user` question and the person's answer go back in every request whatever the number is (the rest of that turn, its thinking included, does not), and an older turn that used no tool at all goes back exactly as before, with its stored thinking as `PERSIST_REASONING_TOKENS` says. A turn is one message a person sends plus the assistant's whole reply, however many tools it used, the unit `TOOL_OUTPUT_RETENTION_TURNS` already counts in. The cut is made where the request is built, so it applies on `/responses` and on `/chat/completions`, in both tool execution modes, with tool cards on or off, and to the rounds a temporary chat sends back. Shown tool cards and the pipe's own copy of each round are not changed, and raising the number hands the older rounds to the model again on the next request, from the saved cards or, with tool cards off, from the pipe's own copy. A chat that is still in use keeps that copy: each message renews the retention date of the stored rows its conversation points at, the rows it reads by the read and the rows of older turns it leaves unread by one update that loads no payload and rewrites a row only when its date is more than an hour old, and the rows of a deleted message are not renewed and expire after `ARTIFACT_CLEANUP_DAYS` as before. The stored rows of an older turn are not read when the request carries that turn's tool rounds itself (tool cards on). They are still read, and left out, in a chat with tool cards off, where they are the only record of what the turn did and of its `ask_user` rounds, so that chat's database read per message is what it was before this setting while its request shrinks; they are also read for a turn that holds an `ask_user` round, while `PERSIST_REASONING_TOKENS` is `next_reply` or `disabled`, and for the turns the picture reuse window still reaches. Each new turn moves the oldest kept turn out of the window, so the request differs from the previous one from that turn on, and a provider's prompt cache, which reuses only an unchanged start of a request, can reuse at most the part before it. Each user can set their own value, which wins for their chats. With the default of 5, `TOOL_OUTPUT_RETENTION_TURNS` (default 10) shortens no output, because it acts only on a turn that is still sent. Measured on the pipe's request builder with a generated chat of 220 turns, 15 tool calls of 800 characters of arguments in each (not a live chat): the `/responses` input is 3.70 million characters before this change, 0.29 million at 5, 0.21 million at 0 and 1.80 million at 100. The same chat with 6,820 stored rows, run on SQLite and on PostgreSQL 16.2 through psycopg2 and psycopg 3 (generated rows, a local database, not production): with tool cards on, a message builds its request in 68 to 116 ms against 312 to 774 ms before this change, of which the date renewal takes 14 to 43 ms once the rows are fresh and 49 to 296 ms for the first message after they have gone unused for an hour; with tool cards off, it builds in 231 to 429 ms against 287 to 497 ms before.
- **Web search defaults to Exa** — the Web Tools filter's `WEB_SEARCH_ENGINE` defaulted to `auto`, which, for a request whose only OpenRouter tool is web search, means the provider's own search where the model's provider has one; on Anthropic's models that search returns an excerpt only for the pages the answer cites (Claude Haiku 5.5, 2026-10-16, one search), so most results reached a later turn as links. A request that also offered the date and time tool or web fetch, as the filter's defaults do, already searched with Exa under `auto` (Claude Haiku 5.5, 2026-10-17: eight requests with web search alone ran Anthropic's own search, four with one of those tools alongside searched with Exa, at Exa's fee), so the new default changes the engine only where web search is the only OpenRouter tool on the request. The default is now `exa`, which returns an excerpt for every result, so the card shows them and a later turn can answer from them; OpenRouter's documentation of 2026-10-08 lists $0.007 per search with up to 10 results for it, on top of the tokens the results cost, and the search query goes to Exa, as it already did under `auto` where the provider has no search of its own or another OpenRouter tool was on the request. Measured through OpenRouter on 2026-10-17 with a test key (Claude Haiku 5.5 and GPT-5-mini, two questions that needed a search each, so four Exa searches of five results at the filter's default depth `medium`, which stays the default): the citations carried 2,322 to 15,462 characters of excerpt text per search, the most in these four searches being about 15,500, and a search cost $0.0074 to $0.0092 with the $0.007 Exa fee included. With no depth sent (the filter always sends one, so this is only a reference) the larger of the two questions carried about a third less excerpt text and cost $0.0004 to $0.0005 less. Claude's own search, under `auto` with web search as the only OpenRouter tool, cost $0.0114 to $0.0118 per search (no Exa fee: OpenRouter passes the provider's price through), its citations carried at most a few short passages (none for one question, 431 characters in all for the other), and it read more than they show, as its input tokens, the most of any run in the check, say. A valve an admin has set to a custom value (`auto` included) stays as set, and a valve left on `Default` takes `exa` when the pipe next rewrites the installed filter, which is the next model-list refresh while `AUTO_INSTALL_WEB_TOOLS_FILTER` is on; a filter the pipe does not update keeps `auto`. An OpenRouter workspace that limits the search engines and leaves out Exa rejects a request that names it, so `WEB_SEARCH_ENGINE` has to be set to an allowed engine there. The web search section of the server tools doc lists what each engine shows and keeps for later turns.
- **A web search on `/responses` reaches later turns with its query and results** — on `/responses` the pipe stored every OpenRouter web search as an empty call and one fixed sentence, "Search completed. Sources available in the citations panel below.", so with `Keep tool results` on a later turn showed the model a call with no query and no results, and a follow-up could not be answered from what the search had found. The round now carries the query, the result links and the page excerpts the search engine returned. The card shows the query and the links when the search closes and is updated in place with each link's title and excerpt when the turn ends, and the pipe's own copy is written once, at that moment, with the same text, so the model reads the same round with tool cards shown or hidden. The result is a JSON list shaped like Open WebUI's own `search_web` result, with at most 25 results, 1,600 characters of any one excerpt and 8,000 characters of excerpts per search and a trailing note naming anything left out or shortened; a citation attaches to every search of the turn whose result list holds its link, and a citation no search lists stays in the sources panel. A search the model follows with a call to one of the pipe's own tools before it writes any answer keeps the query and the links, because the follow-up request inside the turn does not carry the search round; a citation a later response of the turn brings for one of its links is added. Anthropic's own search returns short passages only for the pages its answer cites (Claude Haiku 5.5, 2026-10-16, one search), so its other pages stay links, and OpenAI's own search is not kept as a round. A reply stopped before the turn ends keeps the query and the links, on the card and in the pipe's copy alike; a reply that fails after something has reached the chat keeps the excerpts seen so far; a reply handed back for a retry before anything reached the chat writes no copy. `Keep tool results` off gives the call with its query and the usual note in place of the result, once. Chats saved before this change keep the stored sentence, and for a search that listed links the card text changes from that sentence to the list.
- **Synchronous tools run on the pipe's thread pool** — Open WebUI hands the pipe every synchronous tool wrapped in an async shell, and the pipe ran that shell on the event loop, so one tool's blocking body froze every chat the worker serves for as long as it waited. The bodies of wrapped-synchronous tools now run on the pipe's own tool pool while `Synchronous tools on threads` (on by default) stays on: the calls the pipe runs overlap on the pool across chats, their progress updates still reach the chat, and the per-call, round and result-wait limits count from the moment a call's body starts and reach it. Inside such a body asyncio reaches the event loop: `asyncio.get_running_loop()` returns a stand-in for it, and `asyncio.create_task()`, `asyncio.ensure_future()`, `loop.call_soon()`, `loop.call_later()` and `loop.call_at()` hand their work to the real one; `create_task` returns a `concurrent.futures.Future` rather than a Task, `asyncio.run()` fails in the body as it did when the body ran on the event loop, and `run_in_executor()` is unavailable. A call that is waiting is never ended by a time limit or a full queue. A call that has not started — in the queue, waiting for a worker, a request slot or a process-wide slot, for its Tools file's share of the pool's threads, inside Open WebUI's re-bind of the tool, or for a thread — runs as soon as what it waits for frees, however long that takes, and its answer reaches the model; before, a round or result-wait limit that ran out first, or a queue that stayed full, ended it unstarted. The one wait nothing ends is a call behind a tool body that ignores cancellation: that body keeps its request slot and process-wide slot until it returns, and only Stop ends a call waiting for one of them. Every limit (`TOOL_TIMEOUT_SECONDS`, `TOOL_BATCH_TIMEOUT_SECONDS`, `TOOL_IDLE_TIMEOUT_SECONDS`) now counts from that call's own start, and a call that is waiting, or withdrawn at a Stop, never strikes the tool's breaker; a call that has started and reaches its limit strikes once. `TOOL_BATCH_TIMEOUT_SECONDS` is a backstop for what `TOOL_TIMEOUT_SECONDS` cannot end: a body that ignores its cancellation, or a call stuck after its body returned. When a request ends — Stop, a streamed API client dropping the reply, or an error; closing the browser tab does not end a chat that Open WebUI runs in the background — exactly the calls that have not started are withdrawn the moment the cancellation reaches the round, and never start, a call a pool thread has taken but not yet entered among them, while calls already running keep `TOOL_SHUTDOWN_TIMEOUT_SECONDS` to finish. One Tools file (the functions that share one Open WebUI tool id) may hold at most `max(pool width - 2, min(pool width, MAX_PARALLEL_TOOLS_PER_REQUEST))` of the pool's threads — 6 of 8 on the defaults, nothing reserved when `MAX_PARALLEL_TOOLS_PER_REQUEST` is at least the width — counted until its bodies return, never until a limit gives up on them, so a Tools file whose calls hang cannot take every thread; its further calls wait for one of its bodies to return and hold no request slot, no process-wide slot and no worker while they wait, so they never keep a healthy tool from running, and after five minutes of waiting one WARNING per Tools file names the file and how many of its calls wait and, when the pool is full of other files' bodies, which Tools files hold the threads. When a call's limit ends while its body still holds a thread, one WARNING per Tools file in five minutes names the file and how many threads it holds. An error raised before a call's body can start, a thread that cannot be started for one, reaches the model at once with its own text and counts as it did before. Two more changes to know: a tool bound to the thread it starts on — a sqlite connection opened at load, a signal handler — can raise where the loop path accidentally worked, and the switch is the escape for such a deployment; and a tool named like one of the pipe's internal frames (`new_function`, the wrapper's own name) no longer takes a breaker strike for a missing argument: one strike before, none now. With the switch off, wrapped-synchronous tools run on the event loop exactly as before, freezes included; the waiting, limit, Stop and breaker rules, the strike-rule correction and the event-loop stand-in a plain-function tool's worker plants (its current loop, with no running loop, so `asyncio.run()` still works there) apply in either position, while the share of the pool's threads governs only the calls that run on it. `async def` tools, arguments, results and cards are unchanged, and in Open-WebUI tool mode a streamed reply's calls keep running in Open WebUI.
- **The pipe's Redis client follows Sentinel and Cluster deployments** — the artifact cache, the write-behind queue and the cost-snapshot writer built their one Redis client from `REDIS_URL` alone, read as a plain standalone address: on a Redis Sentinel deployment that URL names the master's service, which the client read as a literal host, and on a Redis Cluster the client followed no redirection, so every key whose slot another primary owned came back as a moved-key error and the operation fell back or was lost. The client now follows Open WebUI's own topology settings: `REDIS_SENTINEL_HOSTS` (with `REDIS_SENTINEL_PORT`) builds a Sentinel client whose connections re-resolve the current master, so after a failover the client follows the promoted master without a restart, and operations fail only while the sentinels are promoting it; and `REDIS_CLUSTER=true` builds a cluster client that routes every command by key slot. The Sentinel client is assembled from the same URL parsing the installed Open WebUI applies, and authenticates to the sentinels themselves on 0.12.1 and later; on 0.11.4 and 0.12.0 the sentinels are reached without credentials, the same limitation Open WebUI's own manager has there. `REDIS_URL` in the `rediss` (TLS) scheme keeps TLS on standalone and cluster clients; together with sentinel hosts it is refused and the cache is disabled with a warning naming the reason, because that path keeps no TLS flag and the credentials would cross the sentinel and master links in plain text.
- **Zero data retention in a chat someone else owns** — with `ALLOW_USER_ZDR_OVERRIDE` on, a reply followed only the replying person's own `REQUEST_ZDR`, so a participant who had not asked for zero data retention sent the owner's earlier messages to providers that may keep data even when the owner had asked for it. A reply in a chat another user owns now uses zero data retention when either the person replying or the chat's owner asked for it. When the owner's saved answer cannot be read, or the chat's owner cannot be determined, the reply uses zero data retention, and a refusal names that cause. Each such reply makes one light ownership check; the chat and the owner's settings are read only when someone else owns the chat, and a Fusion turn decides once for all its stages. `ZDR_ENFORCE` and replies in one's own chat are unchanged.
- **Session logs for people who reply in a shared chat** — on Open WebUI 0.12 and later a chat's owner can let other people reply in it, and those replies were not archived: the session-log gate admitted only the chat's owner or an administrator. The gate now asks the same question the pipe asks before it writes to a chat (the owner, an administrator, or anyone Open WebUI lets write in that chat), so a participant's turn is archived like the owner's. A person who can only read the chat is still not archived, and below 0.12 nothing changes.
- **A generated-image reply too large to keep is refused while it is read** — the pipe read the JSON reply of an image generation in one piece with no bound, and decoded it on the worker's event loop, so a reply far past the size the pipe would ever deliver was held whole in memory and parsed before the per-picture size limit got to say anything about it. The pipe now reads the reply in pieces and stops once it passes a size derived from `Maximum base64 upload size` (`BASE64_MAX_SIZE_MB`): that limit written as base64 for each picture the request asked for, plus 8 MB for the rest of the reply. A reply that announces a size past the bound is refused before any of it is read, and one that does not announce its size, as a chunked reply does not, is refused at the point it passes the bound. The person sees an error that says OpenRouter generated and billed the image but the reply was more than that many MB and was not read; because the reply is not read to the end, its usage figures are not available and the generation's cost is not recorded. Measured on aiohttp 3.14.3 against a local server, one 5 MB reply, a single read of up to 4 MB returned 93,239 bytes of it when the reply announced its size and 64,081 when it did not, which is why the pipe reads in pieces; at the default 50 MB limit and one picture the bound is 74.67 MB, and a reply of 69.9 MB, the largest a 50 MB picture makes, was gathered in 37 ms and decoded in 101 ms (Python 3.12.2). A reply inside the bound is decoded as before. On the streamed form of a generation, which a request for one picture without an input reference uses on the models that publish streaming, the bound applies to each event of the stream: previews and the stream's total length are not limited, and a finished picture past the bound is refused the same way. The limit's help text names the new bound.
- **The Zero Data Retention owner check no longer reads the whole chat** — in a chat another user owns, the check read the entire chat out of Open WebUI's database only to learn its owner. The JSON decode of a large chat runs on the worker's event loop and grows with the chat. The pipe now reads the owner from the owner column alone. Measured against a 10 MB chat on SQLite with Open WebUI 0.12.2 and 0.12.3, the owner read decodes nothing where the whole-chat read decodes about 10 MB. When the select cannot be used - a database error, or an Open WebUI without the database layer it needs - the pipe reads the whole chat as before, which returns the same owner, and logs one WARNING per kind of error.
- **A row the database keeps rejecting no longer blocks the Redis write queue** — the write-behind flush wrote its batch in one statement and put the whole batch back at the head of the queue when the database refused it, so one row the database could never accept was retried every cycle for as long as the worker ran, the rows behind it waited, and the flusher backed off to five minutes and raised its critical alert although the database was healthy. When the database rejects a batch for the content of one row (a value the table cannot hold, a constraint the row violates), the flush now writes that batch's rows one at a time: the accepted rows are stored, the rejected row goes back to the front of the queue, and the flush counts as a success, so the flusher keeps its normal pace and raises no alert. The rejection is counted per row in Redis, once per flush, shared by all workers. At `Rejected write attempt limit` (10 by default) the original queue entry moves to the Redis list `<function id>:dead` (the newest 10,000 are kept, a WARNING says when the oldest are dropped, and the list never expires) and one WARNING names the row; a set-aside entry is not in the database and nothing moves it back by itself, so the persistence guide gives the commands that re-drive it by hand once the cause is fixed, including the copy-then-remove form that works on a Redis Cluster, where `LMOVE` between the two keys is refused. A database outage still fails the batch as a whole and never counts against a row. The same rule applies to a direct write when Redis is off: the rows the database accepts are stored, the rejected row is dropped with one WARNING and the existing notice that some stored items could not be written, and the database breaker is not charged for it. The new setting `Rejected write attempt limit` sits under `Reliability / Redis`.
- **A stall alarm logs the code a blocked event loop is stuck in and how long the whole worker was unresponsive** — a step that blocks the event loop freezes every chat the worker serves, and a call that holds Python's global interpreter lock for about five seconds can get the worker restarted by its supervisor's health check with nothing in the log. With `Stall alarm` on (the default), a watcher thread and a heartbeat task on each pipe instance's event loop check once a second. An event loop that has not run for `Event loop stall threshold` is logged as a WARNING that carries the loop thread's current stack, with one more line when it runs again. When the watcher itself cannot run for `Whole-process stall threshold`, the next WARNING says how long the whole process was unresponsive. The alarm starts when the pipe first lists its models or serves a request, so a worker whose saved switch is off never creates a thread, timer or file. A thread dump of the whole process is a separate switch, `Thread dump on a whole-process stall`, off by default: a timer that does not need the interpreter lock then writes the stack of every thread to a file in the system temporary directory, and the watcher logs that dump when it runs again; the file is what remains if the worker is killed first. The dump reads the other threads' frames without holding the lock, so it can crash the worker in rare cases when the timer fires while they are still running Python code, as a pause from outside the process also causes: switch it on to hunt a freeze and off again afterwards. One alarm serves the process: a reload hands its state to the next copy of the pipe, so there is one watcher, and the settings of the oldest enabled instance apply. Turning the alarm off, or closing the last instance, stops the watcher, cancels the timer and removes a dump file that holds nothing; a worker stopped by a signal can leave an empty file behind. Five new settings sit under `Reliability / Stall Alarm`: `Stall alarm`, `Event loop stall threshold`, `Whole-process stall threshold` (3 seconds at least), `Thread dump on a whole-process stall` and `Stall warning repeat interval`.
- **A pipe shut down from another thread cancels its background tasks at once** — a pipe that was never closed is shut down by the garbage collector on whichever thread collects it, which is not the thread of its event loop. That shutdown cancelled six of the pipe's tasks (its cleanup task, the dashboard's publisher, update and sweep tasks, the usage store's purge task, and a task a plugin hands back to be torn down) with a plain `Task.cancel()`. That call is not safe from another thread and does not wake the loop, so on a loop with nothing else to do the cancellation was acted on only when something else next woke it. The shutdown now cancels each of them through the pipe's thread-safe task drop, which hands the cancel to the loop that owns the task and wakes that loop. A task cancelled from its own loop is cancelled directly, as before.
- **Gemini thinking budget, `0` is now a real off switch for the whole Gemini family** — setting `GEMINI_THINKING_BUDGET` to `0` used to switch thinking off on Gemini 2.5 only. On Gemini 3 it reached nothing at all: the request went out at the chat valve's effort with reasoning and the summary flag still on it, so thinking was on and nothing said so, while the Config tab promised the `0` switched thinking off. The `0` now reaches every Gemini model, Gemini 3 included; a mandatory-reasoning Gemini row (Gemini 3 Pro, like 2.5 Pro) still cannot stop thinking, so the refusal stands, the row's own floor is substituted and the notice goes out in a chat as before. Nothing else moves: the token budget is still Gemini 2.5's alone, because Gemini 3 is governed by Google's thinking-level API and a budget there buys no precise control, so any other value of the valve is ignored on Gemini 3 exactly as before.
- **A preset Fusion turn is no longer repaired by the chat fallback** — a Fusion turn served
  under its `@preset/` spelling (`openrouter/fusion@preset/<slug>`) whose `/responses` call
  fails used to be re-sent to `/chat/completions` by `AUTO_FALLBACK_CHAT_COMPLETIONS`. That
  retry was a *different* request: the `/chat/completions` converter strips the activating
  `{"id": "fusion"}` entry, because the entry cannot travel there, so OpenRouter answered it
  as an ordinary completion, the pipe had no panel to render, and the turn finished as a
  flattened transcript with the breaker strike already refunded. The fallback guard read the
  raw wire id and the converter read the normalised one — two questions, one shape, opposite
  answers — so the guard now asks the converter's question and the retry does not happen. The
  provider's own refusal is what reaches the person, the breaker strike stands, and the turn
  archives as `error` rather than `complete`. A non-Fusion model still falls back exactly as
  before, and a caller-supplied `enabled: false` Fusion entry is still the per-request opt-out
  it was. The same guard governs the streaming and the non-streaming leg.

- **A per-model advanced parameter now schedules the pass that honours it** — the nine `disable_*` flags Open WebUI's model editor writes are read only by the background metadata pass, and nothing in the pass's schedule observed them: an operator who ticked `disable_web_tools_default_on` on one model changed what the pipe *would* write and nothing told it to write it, so the row kept its old state until a catalogue fetch happened to move the content stamp or an admin moved some unrelated setting. A completed pass now records a digest of the flags it read, per model row, and the next model-list build schedules when that digest differs — so saving the model takes effect within one model-list build instead of waiting for something unrelated to change. With nothing changed the digest settles, so the pass still runs once per real change and not once per refresh, and the scheduler still reads no model rows of its own.

- **Switching the last sync valve off now detaches on the pass after the switch** — the gate that decides whether a metadata pass runs is a disjunction over the sync valves and the routing lists, so it can answer "on" but never "this one just went off": with the last term on, an admin switched something off and the pipe attached no pass, so the panels it had attached stayed attached until some unrelated tracked setting happened to move, which for a valve edit reads to the operator as never. The scheduler now also schedules on an on→off transition of any of those terms, and the pass that runs admits itself through the body gate so it actually detaches. Nothing changes for a deployment with a valve on: turning one on still schedules one pass and then settles, and with every term off the gate still declines on every refresh rather than syncing the whole catalogue each time.

- **Video, Continue Response after a Stop** — after Stop on a video turn, a Continue now picks the job already running and waits for it, rather than submitting a second one. Previously the Stop cancelled the request and that request's cleanup also dropped the in-process claim the still-polling job held, so the Continue started a second poller on the same `job_id`: the clip was downloaded and stored twice and the same generation was billed twice. The wait is the point — it is what makes the job bill once — so a Continue issued minutes after a Stop now waits for the render it already paid for instead of starting a fresh one. Shutdown reaches an abandoned job too, which it could not while its claim was gone.
- **Video turns and the circuit breaker / session log** — a video turn that renders a complete answer now resets the user's circuit breaker, and a video turn the pipe refuses before sending is archived as `error` with the control that refused it, rather than `complete` with no cause. Six of the video adapter's early returns handed back a card and recorded nothing: a stored clip or failure block already on the message, a clarification, the per-user video job cap on both the new-generation and the resume arm, and a message with no prompt. So a turn that succeeded left an open breaker open, and a chat holding nothing but a refusal card was filed as a clean turn. The refusal names the valve — `MAX_CONCURRENT_VIDEO_GENS_PER_USER` — or the missing prompt. A turn the person stopped still archives as `cancelled`, unchanged.
- **A turn's address checks now share one budget** — the `ENABLE_SSRF_PROTECTION` help text has always promised a
  single request-wide `ADDRESS_CHECK_BUDGET_SECONDS` ("that turn's other address checks draw on", "a repeated link is
  resolved once per request"), and the conversion leg — the one that rewrites a chat into a Responses `input` array —
  was the single phase that drew on a private budget and a private verdict memo of its own instead of the request's.
  So a request could spend up to twice the advertised budget (20 s in the transform plus 20 s in the tool round that
  followed it), and a link the transform had already resolved could be resolved a second time by that tool round,
  which is precisely what the text says does not happen. The transform now arms the request's own budget and publishes
  into the request's own verdict memo, as every sibling phase already did. **What a person can notice:** a
  picture-heavy turn can now starve its own tool round's picture check, where each of the two previously had 20 s to
  itself, and the person sees the existing `could not be checked in time, so it was not sent` wording rather than a
  new message. Nothing else about the refusals changes — a check that never answered is still named as unchecked
  rather than as a private address, a completed refusal still reads `could not be fetched, so it was not sent`, and a
  turn that runs with no request context installed at all (automation, the live rig) still draws a private budget of
  its own.

- **The `/responses` leg now gates the media it forwards** — the outbound filter that runs immediately before a
  `/responses` body is POSTed read no media at all: it dropped undocumented top-level keys and copied `input` through
  byte for byte, so every media value the conversion leg already refused for `/chat/completions` was forwarded
  verbatim on the other endpoint. It now puts each one to the same predicate — no cleartext `http://` link unless
  `ALLOW_INSECURE_HTTP` and `ALLOW_INSECURE_HTTP_HOSTS` permit it, no Open WebUI file path, no scheme a provider can
  dial, no non-base64 or over-bound `data:` payload, no `input_audio` URL — drops the block, and names the refusal.
  `ALLOW_INSECURE_HTTP`'s and `ENABLE_SSRF_PROTECTION`'s help text now say both legs; neither adds an address check,
  which still happens at ingress alone. The reach is not a caller's own `/responses` body: `CompletionsBody.messages`
  is required and ingress overwrites `input`, so what puts a media block inside `body.input` is a **filter function
  rewriting `messages`**, or a **`message` item the model itself authored** in a reply — stated here because it is
  narrower and stranger than the item's own text promised. The request sanitizer's `function_call_output`-only scope
  is deliberate and is not the gate for anything else; the outbound walker is. `input_audio` keeps its own rule
  (OpenRouter takes audio as base64 only) and is not widened into this gate. A value one of the two legs refuses is
  still reported as `Images: skipped N (<reason>).`, one line, never twice for the same block.

- **A tool's picture the inliner cannot name is skipped and named, not fatal** — when the file gateway resolves an
  Open WebUI file path to a file id, it inlines it and sends a `data:` URL, because no provider can fetch this
  deployment's own authenticated endpoint. Four spellings of that path yield no id at all — `/API/V1/FILES/abc/content`
  (the id pattern does not fold case), `/api/v1/files/internal.png` (the only segment is a filename),
  `/api/v1/files/%61bc/content` (percent-escaped) and `/api/v1/files/` (no segment). For all four the storage read was
  never attempted and the branch still fell through to `FileUnavailableError`, which is a terminal card: one tool
  picture the pipe cannot resolve cost the person the whole turn, and the sentence it stopped with claimed the file
  was gone — a claim about storage the pipe never made. Those four are now dropped from their block, the body is
  still dispatched, and the skip is named in the turn's usual `Images: skipped N (a picture naming … cannot be read
  from Open WebUI storage).` line. A reference whose id **does** read is unchanged: inlined when readable, terminal
  error card when the record is not there, because in that case the pipe did read the record. A file the *person*
  attached is a different fact and is unchanged, and so is the `input_file` arm. The two vocabularies are distinct on
  purpose: the gate's own causes (`insecure_http`, `unusable_link`, `unencoded_inline`, `oversized_inline`,
  `internal_reference`) describe a link the pipe chose not to forward, while `unreadable_internal_path` describes a
  reference it could not read at all, and the operator log says which is which by naming the cause on every line.

- **Per-user "How long to keep reasoning" now defaults to the whole conversation** — the
  per-user `PERSIST_REASONING_TOKENS` control used to default to `next_reply` while the
  administrator's site-wide valve of the same name defaulted to `conversation`, so a user who
  had never opened their own settings silently got the shorter window. It now defaults to
  `conversation` too: a user who chose a value keeps it, and a user who never set the field
  gets the administrator's site-wide value as before. A user whose stored row the pipe cannot
  read (a rotated `WEBUI_SECRET_KEY`) falls back to this per-user default, which is now
  `conversation` rather than `next_reply`. The cost is input tokens, not correctness: a longer
  chat replays more of the model's earlier reasoning back to it on every later turn. OpenAI's
  reasoning guide is the reason for the default — the GPT-5.6 model family supports
  `all_turns` and uses it by default, which needs the complete history replayed, and that is
  what `conversation` does where `next_reply` keeps only the previous turn's. `disabled` and
  `next_reply` themselves, the terms their cleanups run on, and the administrator's default are
  unchanged; the rows the cleanups retire are cleared to empty markers rather than deleted, the
  marker left for the normal retention cleanup.

- **Tools are no longer sent in strict mode by default** — `ENABLE_STRICT_TOOL_CALLING` now starts off, so a
  tool schema reaches the provider as its author wrote it, with an explicit `strict: false` rather than being
  converted to strict JSON Schema. OpenAI rejects a strict tool whose parameters contain a free-form object, and
  Open WebUI's built-in `ask_user` is exactly that shape (`questions` is an array of objects that declare no
  properties), so with strict mode on by default every request offering `ask_user` was refused by an OpenAI model
  before any tool ran. **Migration:** an admin who wants strict schemas back ticks the valve; the conversion
  itself is unchanged, so turning it on gives today's strict behaviour for every tool.

- **A background task on a picture-only or video-only model is skipped, not sent** — with no Task Model
  set, Open WebUI asks the pipe to title, tag and generate follow-ups using whatever model the chat used.
  On a model whose catalogue row lists no `"text"` among its output modalities, those requests cannot
  succeed: the pipe spent two attempts, logged an ERROR and toasted a task-failure card naming a fault that
  is really a property of the model (GitHub issue #61). The pipe now answers an empty task answer instead,
  before anything leaves, which is the value every Open WebUI task kind already reads as "no task ran" — the
  chat takes its first message as its title, exactly as it does with automatic titles off, and no tags and no
  follow-ups are written. The rule is one predicate on the resolved model's spec, so a variant or a virtual id
  resolves to its base; a model with no spec, or one that lists no output modalities, is treated as able to
  answer and keeps today's behaviour, and a multimodal model that emits text alongside pictures keeps its
  tasks. It is not an error: no breaker record, no card, nothing shown to the person. The first skip in a
  worker logs one WARNING naming the model, what it makes, and where to set a Task Model
  (**Admin Settings → Interface**); every later skip logs the same line at DEBUG.

- **A video model's own instructions now apply on "use my prompt verbatim" turns too** — the
  classifier writes the scene, but a turn where it was asked to hand the person's own words back
  untouched used to hand the video model those words *alone*: the deployment's system prompt, which
  the pipe prepends to every other video turn, was dropped from the wire, and the footer said
  nothing about it. The pipe now composes on both arms, so the model's standing instructions go
  ahead of the scene exactly once whichever arm a turn takes, and the classifier's own prompt no
  longer tells it to fold them in as well. The footer keeps showing the scene alone and adds one
  line saying the instructions were sent too — a system prompt is configuration, not something a
  person asked to read in their own chat. No valve and no configuration change.

- **Video generation, dual models** — a model the chat and video catalogues both list now publishes one verdict
  about pictures instead of two. `capabilities.vision` and `capabilities.file_upload` restate the row's merged
  feature set rather than keeping the chat twin's own answer, so a model whose chat twin takes no picture but whose
  video row takes a first frame is offered the `vision` box instead of being refused the attach it would have sent,
  and a twin that takes pictures keeps the box when the video row takes none. The other seven capability boxes are
  unchanged, so an image-generation answer the chat catalogue published still stands. The Direct Uploads decision
  is unaffected: it is read from the row's features, not from this dict.

- **A tool call the pipe gave up on is now counted as a failure on the dashboard** — a round could
  settle a call without ever reporting it: a call `TOOL_IDLE_TIMEOUT_SECONDS` gave up on after it
  had started. It builds the model's output exactly as before and dispatches to plugins now, so a
  call the pipe answered the model about is a call the operator's Live row shows, and the Live row
  returns to `streaming` instead of being stranded on `status: "tool:<name>"`. Nothing the model
  sees changes: the output list for that arm is byte-identical, and no valve moves —
  `TOOL_IDLE_TIMEOUT_SECONDS` is still unset by default and the plugin system is still off unless
  an admin turns it on. **Not comparable across this release:** persisted usage rows (`db_row`
  writes `tools_failed`, and every Usage window sums it) written before this fix count fewer
  failures than rows written after it, so a `tools_failed` figure that crosses this version is
  comparing two different things. The give-up is rare — it needs an idle limit an admin has set
  and a started call still running when its allowance ends — which is exactly why the shortfall it
  closes survived on the rows an operator is chasing.

- **Pipe dashboard, API-key callers** — the action route behind every dashboard tab now stands behind Open WebUI's own `get_verified_user` rather than a hand-maintained copy of `get_current_user`, so an `sk-` API key is admitted where it was previously decoded as a JWT and refused. Everything else about the route's identity decision is unchanged in substance and delegated wholesale: a request with no usable `Authorization: Bearer` credential is refused before Open WebUI sees it at all, so the session cookie and the `x-api-key` middleware fallback stay unreachable and the route remains CSRF-safe. What changes for an operator is the surface: a key holder Open WebUI permits can now drive the dashboard, bounded by the same read/write grant and admin-role checks as a session, and governed by Open WebUI's own API-key settings — `auth.enable_api_keys`, the `features.api_keys` permission for a non-admin key holder, and `auth.api_key.endpoint_restrictions`, which names `/api/pipe/dashboard/action` in its allow-list. Set that restriction if a deployment wants the old surface. A `401` now carries Open WebUI's own message rather than a bare `{"detail": null}`, so an admin on an expired session is told why instead of being shown "Could not load configuration".

- **Admin's Max Upload Size, no longer gated on the RAG bypass flag** — an install that turned
  `rag.bypass_embedding_and_retrieval` on stopped being held to the cap its admin last saved under
  **Admin → Settings → Documents → Max Upload Size**: the pipe read the bypass flag beside the stored
  size and treated "bypassed" as "no cap", handing that admin a `REMOTE_FILE_MAX_SIZE_MB`-sized download
  from an install whose own upload leg was refusing at the stored number. That flag switches off
  *embedding*, not the admin's ceiling — Open WebUI applies `rag.file.max_size` on its own upload leg
  with no reference to it — so the stored cap now binds either way, and the pipe asks the store for
  that one key rather than two. No valve default changed, and the smaller-wins arithmetic, the
  default-valve carve-out, the 500 MB ceiling and a cleared admin box all behave exactly as before on
  an install that does not use the flag.

- **Plaintext HTTP allowlist, a bare entry names one port** — an `ALLOW_INSECURE_HTTP_HOSTS` entry with no port now
  admits port `80` only, exactly as an explicit `host:80` entry does. It used to admit every cleartext port on that
  host, so an operator who listed an internal service by bare hostname also opened `:22`, `:6379`, `:9200` and the
  rest of its TCP ports to the pipe, which the valve's own "exact match only" help text does not describe.
  **Migration:** if a listed service listens on a port other than 80, add a `host:port` entry for it (for example
  `internal.svc:6379`) — a bare entry does not cover that port and the link is refused until the port is named.
  `https://` links are unaffected: TLS is the control there and the port is the provider's business. The valve's
  help text, the dashboard's `Plaintext HTTP host allowlist` detail and the five refusal messages a refused attachment
  or picture produces all carry the same rule, and the refusal log now names the port instead of blaming a host the
  operator did allowlist.

- **An SVG a person attaches, a tool hands over, or the model wrote is not sent to the provider** — Open WebUI asks two questions before it turns a file into a model image block, is it an `image/` type and is it not `image/svg+xml`, and asks both at both places it does so. The pipe asked only the first, so a vector the interface had declined to draw was typed, converted and forwarded to a third party. The pipe now answers the same two questions, on the resolved media type rather than the declaration, and reports the refusal the way it reports every other picture refusal: the person sees `Images: skipped N (not identifiable as an image).` on the turn that carried it, the picture is not sent, and it is not counted against `MAX_INPUT_IMAGES_PER_REQUEST`. Every leg a picture travels on now runs the same rule: an inline attachment, a remote attachment the pipe downloads, an Open WebUI file, a picture reused from an earlier turn, a live tool round, the request sanitizer, and the `/chat/completions` fallback — the last two reach a gate that checks the scheme, the cleartext valve and the size cap and never resolved a type, which is why an SVG survived them while the same picture was refused everywhere else. Everything else is unchanged: a payload of any other `image/*` type still goes, including the BMP and TIFF spellings this gate has always forwarded under their declaration, and PNG bytes that declare themselves an SVG are still sent as the PNG they are. The picture the pipe *generates* is untouched — a stored SVG is still stored as `image/svg+xml`, and Open WebUI still draws it in the chat. What changes is the next turn: an SVG the model wrote is no longer lifted back onto a later request as a picture. No valve and no configuration change.

- **A file's name can no longer carry the pipe's own transport markers** — every other free-text field the pipe hands the provider has its hidden marker lines removed before the request goes out, and a caller-supplied `input_file.filename` was the one that did not: it was forwarded exactly as it arrived, on both block spellings and on either transport. The pipe's framing is line-based, so a name carrying `[P:final_answer]: #` or a ULID marker line on a line of its own was not an odd string but a second thing — a marker the pipe's own replay then read as its own, which split a reply into several messages and could rebind a stored artifact to a turn the caller does not own. The same line filter as everywhere else now runs on the name, with the same rule that a marker-shaped line only counts on a line of its own, so `[2026] report.pdf`, `quarterly [P:final_answer]: # report.pdf` and a name with an ordinary newline all survive byte for byte, and nothing is renamed or trimmed. A name that is nothing but marker lines loses the `filename` field and the file still goes out on its `file_id`, so a person never loses an attachment to a name. No valve and no configuration change.

- **Pipe Dashboard write gate** — on an install with `BYPASS_ADMIN_ACCESS_CONTROL=false`, a second administrator who is
  neither the dashboard model's owner nor a write-grantee is no longer refused the operator actions; and an
  administrator whose `Pipe Dashboard` model row was never inserted is no longer locked out of a dashboard the picker
  still lists. The write gate now answers Open WebUI's own model-write formula (`routers/models.py:888`, `:956`,
  `:1077`, `:1126`), whose admin term carries no valve. The **read** gate is unchanged and still honours the valve, so
  whether a second admin can *open* the dashboard is still that administrator's setting; only the actions he can run
  once it is open follow Open WebUI now. A model row whose read raises is still undeterminable rather than a verdict.

- **Two installed copies of the pipe no longer share one dashboard** — every dashboard binding is now keyed by the
  pipe's own `Pipe.id`: the socket handler's, the action route's and the publisher's registrations, the socket.io
  viewers room, and the model row a subscribe or an action authorises against. A worker serving two copies of the
  pipe therefore resolves each request against the install that named it instead of whichever copy registered last,
  and one installed copy's off switch, config save or valve event no longer reaches the other's viewers. The action
  route takes the install's id in the request body (`pipe`) and refuses an id this worker has no registration for with
  `404 {"error": "unknown pipe"}`, before any authorization read. **A dashboard panel persisted in a chat message
  before this change carries no such id**, so its Config and Update tabs answer `unknown pipe` until the panel is
  reopened; the panel says so in words rather than showing a raw refusal.

- **Gemini 2.5 thinking, hidden reasoning traces** — a request that carries `reasoning.exclude` of `true` no
  longer defeats the `GEMINI_THINKING_BUDGET` valve. `exclude` is OpenRouter's display preference ("the model
  will still use reasoning, but it won't be returned in the response"), so with `GEMINI_THINKING_BUDGET=0` the
  valve's off now holds against such a request and goes out as `{"effort": "none"}`, and a request's own
  `reasoning.max_tokens` is no longer dropped on that shape at a nonzero budget. On Gemini 2.5 Pro, whose
  reasoning is mandatory, the refusal status line naming the model now fires where it did not, and on any
  mandatory row the repair consumes only the keys it reads as an off, so a chat that hid the trace is not
  shown it. Every path a chat or a background task takes is covered, Open WebUI's tool-execution mode included.

- **Video intent classifier, a retired Task Model is now named rather than left to guess** -- a Task Model the host no longer publishes is already skipped before it is called, and the operator reading the warning had no way to tell that case from a row nobody filled in: both said `no usable task model under task.model.default / task.model.external`, so an admin who had renamed the model had nothing to match the row against. The drop is now recorded at DEBUG by the resolver, naming the id it dropped, and the warning says which case it is and names the id -- `the host no longer publishes 'ext-4o'` -- while an unset row keeps the wording it always had. No user-facing text changed, the warning's per-chat latch is unchanged, and no valve or configuration change. The documentation follows the code: the sentence that still offered a retired id a branch is amended, and the `VIDEO_INTENT_TASK_MODEL_FALLBACK` row now covers a configured row whose model is gone, as the valve's own dashboard detail already did.

- **A housekeeping task that recovers is heard about again the next time it fails** -- the task-failure toast is latched on `(model, chat and user)` so a long outage warns once rather than on every dispatch, and nothing ever cleared a latch: a title, tag or follow-up that failed at 09:00, answered again at 09:05 and failed at 10:00 was told about once, and the second outage stayed invisible until 300 unrelated keys pushed the first record out of the window. A successful task now removes its own chat's record, so a recovery re-arms the toast and the next failure warns again. The release is one key and not the table, so one chat's recovery never silences another's, and consecutive failures with no recovery between them still toast exactly once. A temporary chat is unchanged: it keeps no record, is toasted on every failing turn, and a successful turn writes nothing for it. The latch's shape, its 300-entry window and the anti-storm property for two overlapping failing tasks are all as they were.

- **`logit_bias`, the text an admin typed, is decoded instead of destroying the request** -- Open WebUI writes `logit_bias` as the free text its Advanced Parameters box holds (`"1234:100, 5678:-50"`), and the pipe's own outbound body types the field as a mapping. The string was refused by that body, so a single `logit_bias` in an admin's Task Model Params row -- or in a chat's own advanced parameters -- raised a validation error before any request left the process: the title, tag or reply silently became `### <U+26A0> Unexpected Error` / `Error type: ValidationError` for as long as the setting stayed. A `logit_bias` that arrives as a string is now converted to the value Open WebUI's own `convert_logit_bias_input_to_json` (`utils/misc.py:1129-1144`) produces, with each bias clamped to `[-100, 100]`; a value that is not a string is carried through unchanged; and one that does not parse is dropped from the request rather than forwarded or re-raised, with a warning naming it so the ignored setting is diagnosable. The conversion is the pipe's own, not an import of Open WebUI's, and it runs before the typed body is built, so it covers the ordinary chat turn, every housekeeping task and the video intent classifier alike. On `/chat/completions` the decoded mapping is forwarded to OpenRouter; `/responses` still drops the field, which is deliberate. No valve and no configuration change.

- **Video intent classifier, retired Task Models** — a Task Model the host no longer publishes is now skipped instead of called. When `request.app.state.MODELS` is a non-empty mapping, an id it does not contain is not a candidate, which is the same membership check Open WebUI applies to a task model itself (`utils/task.py:16-27`, `routers/tasks.py:145-150`, and `generate_chat_completion` raises `Model not found` at `utils/chat.py:191-193`). A retired Task Model therefore no longer costs up to four billed attempts per video turn, every one of them answered `Model not found`, and no longer arms the per-user classifier breaker on a configuration state: the turn lands on the already-built `no_task_model_candidates` path, which logs once per chat and opens no outage. An empty, absent or unreadable model list is not evidence and filters nothing, so a host that has not synced its model list still calls the model its admin configured. No chat-model fallback was added, and no new failure code, log line or latch.

- **Tool breaker's out-of-service notice** — the one line that names a tool as out of service is no longer
  lost silently, and no longer costs a round more than that round's own limits allow. Its bound was opened on the
  round's clock, so a round whose per-call announcements had already spent `TOOL_BATCH_TIMEOUT_SECONDS` started the
  notice on an expired deadline and dropped it; and it knew nothing about `TOOL_IDLE_TIMEOUT_SECONDS`, so a browser
  slow enough to cost the notice more than the whole idle allowance pushed the round past a limit the operator set.
  The bound now runs from the moment the notice itself runs and is also capped by whatever the idle allowance leaves,
  so the notice is inside the round's documented totals rather than beside them. When either bound ends it, a
  `WARNING` names the request and the valve that cut it — the loss used to leave no record at all — and the model is
  still told the reason for every call that was skipped either way. Nothing else about a round changes: one line per
  tool per round, an unset `TOOL_IDLE_TIMEOUT_SECONDS` still leaves the round to its batch ceiling, and a notice that
  fits inside the allowance is still delivered.

- **A refused Continue no longer loses the answer it was continuing** — a chat turn refused at the pipe's admission
  gate while it is being streamed now publishes the frames the same refusal publishes when it is not streamed. The
  refusal is published from the stream generator's own cleanup, and that generator's body does not run until the caller
  starts iterating it — by which time the chat id and the continuation prefix the frame shape is decided from had both
  been reset. On a `channel:` chat the reader was left with a bare terminal frame and no error card; on a **Continue**
  the closing frame carried the refusal as `content`, which the browser assigns absolutely, so `Server busy (503)`
  replaced the answer the person was continuing. Both now take the shape every other refusal takes: the card is
  published, the closing frame carries no replacing content on a continuing turn, and the answer stays where the
  browser had it. A saved chat that is not continuing, an API caller with no chat, a temporary chat and a temporary
  answer are all unchanged.

- **A skipped oversized image names the kind of file, not the caller's path segment** — a stored
  Open WebUI picture that is too large to inline is skipped, and the record that says so used to
  carry the file id the caller wrote in the `image_url` as its subject:
  `Skipping an attached image (01JQ8ABCDEFGHJKMNPQRSTVWXYZ01): larger than the 1048576-byte inline
  limit [cause=oversized_inline]`. That segment is the caller's own string — Open WebUI's file-id
  pattern admits any run of letters, digits and hyphens, so a person's name spelled as a path
  segment is as legal as an id — and a subject is not shortened on its way to the log. The record
  now reads `Skipping an attached image (image): …`, naming the kind of attachment the way its
  sibling arms already name `audio` and `video`. Everything an operator triages on is unchanged:
  the kind, the cause, the byte limit, the single WARNING per cause, the
  `Images: skipped 1 (larger than the 1048576-byte inline limit).` status the person sees, and the
  answer. A file that is genuinely gone is untouched — that refusal still names the id, because
  there the sentence goes to the person, not to the log.

- **A file a tool returned is inlined, not forwarded** — a `function_call_output` whose `output` is a list of
  `input_text` / `input_image` / `input_file` parts is now resolved instead of being flattened to a JSON string
  before the request is dispatched, so an `input_file` part naming `/api/v1/files/<id>/content` no longer reaches
  OpenRouter with an Open WebUI path, a host, a userinfo and a `?token=` on it. Two passes were involved and both
  are fixed: the sanitizer decided on part type whether a list was media, and the inliner's `output` arm dispatched
  pictures only. A part type nothing recognises is still flattened, and a caller who writes an Open WebUI path as
  plain prose is quoting it, not handing it over — the claim here is about structured parts.

- **A cancelled attachment read reclaims its copy** — `materialize_owui_file_to_temp` now settles the copy it
  shielded and unlinks the private temp it produced when the call is cancelled, including when it is cancelled
  repeatedly. A Stop used to land the turn while a worker kept writing a full private copy of somebody's attachment
  whose name nothing held, so the file outlived the call and the pipe paid for a copy nobody would ever read. A
  returned path is untouched and an `OSError` out of the copy still reaches the caller unchanged (TODO T637 is
  unchanged and still open: the temp a failed copy wrote is still left behind).

- **Two refusal cards stop appearing for block spellings that carry a real source** — a file block naming its document under `id`/`url` rather than `file_id`/`file_url`, and an `input_image` naming a `file_id` instead of a url, are now sent rather than skipped. Both reached `Files: skipped 1 (has no readable source).` and `Images: skipped 1 (an image carried no picture data).` respectively, so an operator watching refusal counts will see both drop for these spellings even though nothing else was refused. The file block is read through the same address gate, plaintext policy and storage-path rules as any other link (an `id`/`url` naming a private address or a cleartext link is refused exactly as `file_url` is), and the image block's id is forwarded as a file reference to the provider rather than resolved out of Open WebUI storage — it is the provider's own id, and it does not count against the per-request image cap. Open WebUI 0.11.4 itself writes neither spelling into a message, so on a stock deployment these counts do not move; they move where an upstream filter or a direct `/v1/responses` caller sends them.
- **A remote picture whose download the cap aborted is no longer forwarded as its link** — the downloader returned
  the same empty answer for every failure, so a transfer that `REMOTE_FILE_MAX_SIZE_MB` (or the admin's stored Max
  Upload Size, whichever is smaller) aborted on the way in was indistinguishable from a 404, and the inline leg fell
  through to the same rule as any other failed fetch: the link went out, and OpenRouter pulled down the bytes the pipe
  had just refused. The downloader now says which failure it was, and both the inline and the reuse legs refuse a
  cap-aborted transfer in the pipe's own words, `Images: skipped N (could not be fetched, so it was not sent).`
  Nothing else moves: a transfer that failed for any other reason — a 404, a timeout, a slow host — is still forwarded
  as its link on both legs, a link the pipe's own address check refused is still never forwarded, and `BASE64_MAX_SIZE_MB`
  is unchanged, so a picture is still refused for its size once downloaded with its byte count named. The reuse leg's
  warm half is unchanged too: a picture the pipe already held and now declines to send still reports
  `N bytes, over the M-byte download limit, so it was not sent`, naming the count it is giving up. (T1283)

- **Secret valve values, a passphrase typed with the `encrypted:` prefix** — a secret an operator types beginning
  `encrypted:` that is not already a stored row is now sealed like any other, instead of being written into the
  valve row as typed. `encrypt` short-circuited on the bare prefix, so on an install that leaves Open WebUI's
  `ENABLE_VALVE_ENCRYPTION` at its default — which is where the row is plain JSON — that value was the artifact
  encryption key and the session-log archive passphrase, sitting in `Function.valves` in the clear, while two help
  texts promised it was stored encrypted at rest. Nothing about the secret changes: the same value is read back and
  both consumers use exactly the text the operator typed, so the artifact table keeps its name and previously
  written archives still open. What changes is the row, and a value already in one is left alone: a sealed row this
  server can open, a damaged one, and one sealed under a retired `WEBUI_SECRET_KEY` all survive a Config-tab save
  byte-identically, because re-sealing any of them would either rewrite the row on every save or hand the operator
  the damaged blob back as their plaintext — which is what disarms the guard that tells them to re-enter the key.
  Both refusals that a stored value can arm now name the remedy alongside the two causes: re-enter the secret
  without the `encrypted:` prefix. An operator who stored a prefixed passphrase before this change keeps working; the
  row is rewritten as a ciphertext the next time the value is saved through either Config tab, and nothing has to be
  re-entered to make that happen.

- **Session log assembler, one offer per pass** — a turn the assembler already offered, or already failed, inside a
  pass is not offered again by that pass, and a turn whose assembly lock another pass holds is offered once per pass
  instead of once per re-listing round. The pass re-lists itself after meeting a lock-contended turn so the window
  keeps moving, but the exclusion it re-lists with was a snapshot taken before the loop, so a turn that failed in this
  pass — and a contended turn whose last failure was the stale-finalize seal, which the completed-turn listing's
  exemption had released — came back every round until the pass ran out its wall-clock budget, spending the whole
  `SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS` on SQL and never reaching the turns behind it. Nothing else changes: the
  backoff is still `SESSION_LOG_LOCK_STALE_SECONDS`, a turn whose only failure was the stale-finalize pass is still
  offered by the completed-turn listing straight away, and contention still books nothing.

- **Artifact encryption on a schema-qualified deployment** — on a deployment with `DATABASE_SCHEMA` set, the artifact store now finds its
  encryption key and stops writing rows in plaintext. It was reading Open WebUI's `function` row by an unqualified name, could not
  reach the row at all, and read that failure as "there is no unreadable row here" — so after a `WEBUI_SECRET_KEY` rotation the guard
  never armed and every artifact (reasoning traces, tool results, tool arguments) was stored in the clear with nothing said to
  anyone. The row is now read from the same schema the pipe's own artifact table is built in. The existing key-guard WARNING may
  appear there for the first time; follow it (re-enter the key) rather than silence it.

- **The one chat-message write is gated on write access** — the `error` mark of a refused automation reply is written only to a saved chat the asker may write: owner, admin, or the write access Open WebUI grants (0.12 and later); owner or admin below 0.12. The gate matches Open WebUI's own rule for writing to an existing chat and fails closed: if the write check cannot be run, nothing is written and the operator log says why once per cooldown. The session-log gate asks the same question. Open WebUI saves a reply's citations and closing Fusion panel from the events the pipe publishes, and its loop-limit note from the closing frame or the answer text; the annotations and reasoning-details copies are not written at all.
- **Citation notices** — the "this response included a citation type the pipe can't render" notice now names the
  answer the reader is looking at. The notice used to be published the moment a delta carrying an unrenderable
  annotation went past, which is before the retry loop has decided anything: on a turn that was re-sent it
  described the answer that was thrown away, so the toast named a citation from a reply nobody was ever shown while
  the answer that replaced it got no notice at all. The types are now collected per attempt and published once, after
  the retry loop has closed. A turn whose surviving answer carries no such citation gets no notice, and a turn that
  ends in a failure card gets none either — it renders the card. The notice still arrives once per turn, and the
  types it names are the same set; only the attempt they are read off has changed.

- **Stalled accepted bodies** — a request OpenRouter has already accepted and whose body then stops arriving is one
  POST on every path, and it renders the connection card. The carve-out that names a body lost in transit keyed on two
  exception classes, `ClientPayloadError` and `ServerDisconnectedError`, so the three commonest ways a body stops
  arriving — an `Idle read timeout`, a reset socket, and an expired `Total request timeout` — were not in it. On the
  streaming `/responses` leg and on `/chat/completions` each of those spent the full retry budget on a call OpenRouter
  had already begun generating for, and OpenRouter bills on generation rather than on delivery: three POSTs on this
  leg meant three billed answers for one question, and the reader saw whichever attempt won. On the non-streaming legs
  the same faults were reported as a body that arrived and was not an OpenRouter response, which blamed the operator's
  proxy for a fault in the network path and, because that class is charged nothing, passed the breaker as a success.
  The discriminator is now whether the body **finished arriving**, on both streaming legs and both non-streaming ones.
  A body that arrived whole and delivered nothing still takes the full retry budget, a fault raised before the request
  reached OpenRouter is still retried, and an in-band or status-level provider error keeps its own class and its own
  budget. A mid-body timeout now renders the connection card rather than the network-timeout card; the timeout valve
  it was read from is unchanged, and so is every knob an operator can set.

- **Tool pictures, the sanitizer's leg** — a picture link a tool result carries is now refused on
  every leg that carries one, plain HTTP included. The address check behind `ENABLE_SSRF_PROTECTION`
  resolves a tool result's link once per request and memoises the verdict, but the request sanitizer
  read that memo for `https` alone — so with `ALLOW_INSECURE_HTTP` on and the host allowlisted, a
  cleartext link whose address the check had refused was forwarded unexamined. Cleartext links now
  go through the same memo, and a check that reached no verdict inside its budget is refused in its
  own words (`could not be checked in time`, cause `uncheckable_tool_picture`) rather than being
  reported as a failed download, which is what the valve's help text already promised. An `https`
  link that resolves to a public address is unchanged, and so is every link a request never resolved.
- **Tool calling, a caller's own `strict` on a Responses-shaped tool** — a `strict` the caller wrote
  is now forwarded instead of being dropped. A tool already in Responses shape reached
  `/responses` through a branch that only read a `strict` out of a nested `function` block, so the
  key went out with the entry missing it, and that endpoint's own default for an absent `strict` is
  `true`: a caller's opt-in arrived non-strict and a caller's opt-out arrived strict. A tool that
  arrived without a `strict` still goes out without one, and a Chat-shaped tool still states the
  endpoint's fallback explicitly; a tool that arrives in Responses shape now keeps exactly the keys
  the caller sent, which is what Open WebUI's own converter does.

- **Video generation, status-poll errors** — `VIDEO_STATUS_POLL_MAX_ERRORS` now bounds two things instead of
  one. It still reads as "how many status checks **in a row** may come back as an error", and an endpoint
  failing every time in a row is given up on after exactly that many, with the same card and the same single
  circuit-breaker charge. It now also bounds the errors spent **in total across one whole watch**: a second,
  larger budget of the configured value times five, counted however the errors are spaced. Nothing about the
  valve's own meaning changed, and a stored value of it is honoured as stored — but the pipe can now stop
  watching a job whose status endpoint is failing intermittently rather than only one that is failing
  outright, which it could not before: any successful poll resets the consecutive tally, so an endpoint
  answering every other call never ran one out and the job was watched until it rendered or the silence
  window ran out. Nothing is cancelled either way. The card is the resumable still-running one, the job's
  marker stays, Continue Response still picks the same job back up, and the person is told how many times the
  endpoint did not answer across the watch rather than in a row.
  That budget is a **transport** budget, and it now says so on both help surfaces as well as in `docs/`. Only
  connection failures, timeouts and provider-reported statuses count against it. A fault of the pipe's own code
  escaping the status check -- a bug, or the `RuntimeError("Session is closed")` aiohttp raises on a closed
  session -- used to be counted as a flaky poll, retried the full budget over roughly 27 s of the user's backoff,
  and then reported as "OpenRouter's status endpoint did not answer this video's job 5 times in a row": a card
  naming a party that was never asked anything. Worse, it reached the failure card, whose
  `### Video generation failed` heading the resume path reads as final, so a job the user had already paid for
  was never polled again. Such a fault is now neither counted nor retried. It ends the watching on the first
  attempt, is named in the server log as the pipe's own with its traceback rather than blamed on OpenRouter, and
  leaves the same resumable card the transport case leaves -- marker kept, resume offered wherever the marker can
  be stored. It is still charged once against the request breaker, like any other failure after the job was
  submitted. No valve changed, and no real status-endpoint outage behaves differently.

- **OpenRouter Web Tools filter id** — a recurring "the OpenRouter Web Tools function id is held by a row this pipe
  does not own" warning now re-warns once 300 other causes have been armed, instead of going permanently silent after
  its first emission. The latch that backs it carries no timestamps, so the `cooldown_s` it was given was never read;
  it is now a window-bounded latch like the two per-model install latches beside it, and the eviction is what the
  repeat is keyed on. Nothing else about the filter, the id, or the repair changes.

- **Pipe dashboard, viewer payload** — the live dashboard payload no longer carries the data-dir path, so a viewer
  holding only a read grant no longer learns the server's filesystem layout from the System tab. The key was published
  by the system collector as `system.disk_path`, and every key in a payload reaches every socket in the viewers room.
  The Disk card is unchanged — it renders from `disk_free` and `disk_total`, both of which still ship — and the
  operator's own log still names the volume, latched to once per 300 s. An operator who wants the path already has it
  on the admin-only Config tab's `Session log directory` row.

- **Temporary chats, marker records** — three streaming records that named a temporary chat's browser socket id now
  name it `<not retained>`, as every other operator-visible record already did. They fired on the ordinary path for
  a temporary chat that reasons or calls a tool: two when the turn's committed artifact rows are addressed by hidden
  marker lines in the reply (a WARNING when the content was handed back before its markers were added, a DEBUG when
  they were added), and one when a cancelled turn leaves committed rows with no marker addressing them. Because
  `SessionLogger.get_logger` pins the package logger at DEBUG for the session archive, the id reached the
  process-lifetime log buffer — the dashboard's log view — on every such turn. The levels, the marker lists, the row
  counts and the reason strings are unchanged, so the records still say which turn lost rows and how many; only the
  chat id is gone. A saved or `channel:` chat's id is still named in full.

- **Provider errors** — a provider error body carrying a character no text encoding can hold (an unpaired
  surrogate, which `json.loads` accepts and `str.encode("utf-8")` refuses) now reaches the error card with
  that one character shown as `�`, the replacement character. The same body used to end the turn in an HTTP
  500 rather than a card, and to lose the operator's WARNING record and the turn's whole session-log archive,
  because the archive writer's zip write failed on it and lost `meta.json`, `logs.txt` and `logs.jsonl`
  together. Everything else the provider sent is unchanged and still reaches the card verbatim.
- **Tool pictures** — a faulted SSRF address pool now costs the picture instead of the reply. The
  address checks run on a private bounded thread pool, and that pool can refuse to take work: it is
  shut down when the worker closes, and re-made (with the old one shut down) whenever
  `MAX_CONCURRENT_REQUESTS` changes — so a dashboard save mid-turn can leave the next check unable
  to start. The tool-picture arm was the only forwarding path that let that fault escape: the
  person's own picture, file and video links each degrade to a refusal of their own, and the tool
  link instead took the whole turn down with an `Unexpected Error` card. One MCP tool's image link
  was enough to lose the reply. The check now costs that picture and nothing else: the turn is sent
  with the round's text, the link is not forwarded, and the person sees
  `Images: skipped 1 (could not be checked in time, so it was not sent).` — the same sentence the
  arm already used for a check that ran out of budget, because a pool that cannot run has reached
  no verdict, which is what that sentence already meant. The reader itself is unchanged and still
  raises, so a teardown race is never turned into a security refusal on any other path.

- **Temporary chats** — the reply hold no longer keeps a temporary chat's socket id. `ReplyMemory` keys a reply's
  held rounds and thinking on `(user id, chat id, message id)`, and for a temporary chat that chat-id slot is now
  blank, which is the shape the hand-back budget beside it has always used. Nothing observable changes for a reply:
  the same rounds are held for the same 15 minutes, under the same 64 MiB pool ceiling, for the same user, and are
  released at the same point — a reply still reads its own rows back by its own `message id`, which is the slot that
  was left alone and the only thing telling two replies of one chat apart. A saved chat's and a `channel:` chat's
  keys are untouched. What changes is who can read the id: a temporary chat's chat id is a credential Open WebUI's
  own session pool will authorise a request with, and it was the last process-lifetime structure in the pipe still
  holding one.

- **Video generation, a temporary chat** — a video turn that runs the intent classifier no longer puts the browser's
  socket id into that classifier's task-model request. The classifier builds its own `metadata` rather than going out
  over the chat path, so the identifier valves never reached it, and it stamped the id unconditionally: on a stock
  install, with `SEND_CHAT_ID` at its default off, a Temporary Chat's id went out on every video turn that ran the
  classifier. The stamp now answers to the same two rules as the other two writers — a temporary chat's id is never
  sent, and `SEND_CHAT_ID` off means no chat id at all — so an admin's toggle now changes what the classifier sends as
  well. Nothing else about the request moved; the `task` key is unconditional, since it is what makes the call a task
  call. A saved chat keeps its id on the valve, and so does a `channel:` chat, which is not a temporary chat.

- **Video, channel resume** — a video resume in a `channel:` chat is now authorised against the channel before the stored turn is read, against Open WebUI's own predicate (membership for a `group`/`dm` channel with no admin exemption, a `read` grant or the `admin` role otherwise). A requester the channel does not admit reads nothing, so the turn **starts a new, separately billed job** instead of resuming someone else's stored turn. Open WebUI's own channel route refuses such a requester with a 403 before the pipe is called, so this fires only on a request shaped to bypass that route, and it cannot fire for a member resuming their own turn. Nothing from the other member's turn is read or shown. The identity the check uses is the resolved Open WebUI user, never a caller-supplied one.
- **Pipe Dashboard, a second administrator** — on an installation with Open WebUI's `ENABLE_ADMIN_WORKSPACE_CONTENT_ACCESS` off (so `BYPASS_ADMIN_ACCESS_CONTROL` is false), an administrator who is neither the owner of the `Pipe Dashboard` model row nor holds a grant on it now loses the dashboard: the view, the live-feed room and the `read`-permission actions, including `config_get`, which returns the whole stored valve set. The dashboard read gate asked Open WebUI's `check_model_access` and took a non-exception for an answer, and that function puts its entire grant decision inside `if user.role != 'admin':` — so it admitted every admin whatever the flags said, while the Open WebUI gates that decide whether a model is visible at all refused him. Administrators now answer the same question Open WebUI does: `BYPASS_MODEL_ACCESS_CONTROL`, or the admin flag, or owner, or a `read` grant; with the bypasses off a model row that cannot be read grants nobody, administrators included. Nothing changes on the shipped default (`BYPASS_ADMIN_ACCESS_CONTROL` is on), on the row's own owner, or for anyone holding a grant. Note that `whoami` is itself a `read` action, so a refused administrator cannot call the action that would have told them their status — the panel falls back to its access-denied card.
- **Dashboard Update tab, restore on a package or stub install** — Restore is now refused there with the same
  `package_mode` outcome Update already gave, instead of writing an older bundle snapshot over the install and
  turning a pinned install into a self-updating bundle one. The tab no longer shows the Restore button on a
  package or stub install either; its snapshot list and Delete action stay as they were.
- **`file_url` with no scheme** — an attached `file_url` written without a `scheme://` is now refused like any
  other link the provider could not dial, and the person sees the existing `Files: skipped N (served from a link
  that is neither http nor https, which is blocked by security policy).` status. A bare path (`/etc/passwd`), a
  relative reference (`../../etc/passwd`) and a `host/path` written without one (`169.254.169.254/latest/meta-data/`)
  all reached the provider ungated; they are now put to the address gate and refused on their scheme. The gate is
  the same one an `ftp://` link already met, and the refusal reuses that sentence unchanged — nothing is dropped
  silently. `ENABLE_SSRF_PROTECTION=False` does **not** restore these, because the scheme rule is sequenced ahead
  of that valve exactly as it is for `ftp://`. Raw base64 in `file_data`, a `data:` URL in either field and an
  Open WebUI file path are unchanged: they are inline payloads or become a `file_id` before this point, and are
  never checked. Open WebUI 0.11.4 emits no `file_url` of its own, so this only reaches a custom filter, agent or
  API caller that writes one by hand.

- **Multi-worker Redis deployments** — an admin save of the dashboard configuration, or an unattended update,
  no longer closes the Redis client that Open WebUI's own websocket bookkeeping shares. The pipe took its
  cross-worker lease over the cached client and then tore that client down when it released the lease, so on
  any install with `WEBSOCKET_MANAGER=redis` and `WEBSOCKET_REDIS_URL` set, every configuration save and
  every leader-election poll cut connections that `MODELS`, `SESSION_POOL`, `USAGE_POOL` and the two cleanup
  locks were using. The client survives — the cache still holds it and redis-py reconnects — so the visible
  effect was the `Session pool cleanup failed. Retrying.` and `Unknown receive error` lines around a save,
  plus reconnect churn, rather than an outright outage. The lease is still released exactly when it was
  taken, and a deployment without a distributed lock is unaffected: there was never a client to close.

- **Web-Tools filter valves, the default the pipe seeded** — turning *both* Web-Tools filter valves off
  (`AUTO_ATTACH_WEB_TOOLS_FILTER` **and** `AUTO_INSTALL_WEB_TOOLS_FILTER`) now also releases the default the
  pipe had seeded for the panel, on the same pass that takes the panel off `filterIds`. The family-off arm
  cleared the filter list and left the seeded `defaultFilterIds` entry behind against a panel that no longer
  ran, and the entry could not be removed by the pipe afterwards. The three sibling families already
  consulted their own family-off flag here. A blank panel id that means "the installer did not answer" is
  unchanged: nothing is released and the install is retried, and turning the family back on re-seeds the
  default. `AUTO_ATTACH_WEB_TOOLS_FILTER`'s own help text now says the release needs
  `AUTO_INSTALL_WEB_TOOLS_FILTER` off as well, which is what detachment has always keyed on.
- **Native image filters, a superseded panel** — a panel the pipe retires as superseded now leaves the model
  rows it was still attached to on that same pass: the id comes off `filterIds` **and** `defaultFilterIds`,
  with either image valve on. The sweep's verdict reached the filter list only on the branch that computes it
  itself, and never reached the default list on either branch, so a deactivated filter could stay attached and
  default-on. An id the pipe did not retire is untouched.
- **`meta.builtinTools` the pipe wrote from a File-context untick** — the pipe now writes
  `builtinTools.files = false` only on the media models whose own media rule produced the `file_context`
  untick this pass. On a text or vision model the only source of that untick was the operator's own tick in
  the model editor, and the pipe answered it with Open WebUI's own off-switch for `list_chat_files`,
  `query_chat_files`, `grep_chat_files` and `view_file` — the tools `file_context` off is there to turn *on*.
  An operator who unticked the box now keeps the chat-file tools. A media model the pipe really is correcting,
  and a Files default an admin ticked in the editor, are both unchanged.
  **Rows already carrying `builtinTools.files = false` from an earlier build keep it**: there is no valve to
  undo that write and no tick-back, so untick the box in the model editor, or wait for the marker work
  (B682/T971) that can tell a pipe-written entry from an admin's.

- **Pipe shutdown no longer drops a session-log write that was still in flight** — a turn cancelled by `close()` writes its terminal
  segment from its own cleanup, and that write is now given a budget of its own: it is registered when it starts and awaited before the
  artifact store is closed, so a write slower than the 5-second job-drain budget finishes instead of being dropped with the store closing
  under it. The drain's budget is unchanged, so `close()` stays bounded; on the normal path there is nothing to wait for. The log an
  operator most wants — the turn a reload or a shutdown killed mid-answer — is the one that was at risk.
- **Usage collection, a rotated `WEBUI_SECRET_KEY`** — when the stored valve row cannot be read (a `WEBUI_SECRET_KEY`
  rotation with an older worker still serving), the usage writer now refuses and says so in one latched WARNING naming
  the read. Until now it emptied the Usage tab with no pipe-side line. The refusal itself is unchanged — nothing is
  written, and the batch is dropped rather than held — and a row that is merely unset, or plain JSON from an install
  that never turned Open WebUI's valve encryption on, is still read as an ordinary off and does not warn.

- **Video generation, machine callers** — a video turn that fails reaches a caller with no chat as an HTTP error
  instead of a `200` with a Markdown card in it, whether OpenRouter rejected the submission or accepted it and
  the job then failed. A rejected job
  leaves with the status the pipe resolved on the status line and the same number in `error.code` (`502` when a
  proxy rewrote the `/videos` body); a fault the pipe owns leaves as `500` carrying `Video generation failed.`,
  with no Python class name in the body and the full message and traceback at ERROR in the session log as before.
  The two refusals the pipe raises on its own terms — the per-user cap and a missing prompt — still hand that
  caller their cards, unchanged; the gate here is a failure the upstream path produced, not every refusal.
  This now covers the failures that arrive **after** the submission was accepted as well: a video job OpenRouter took
  and then failed on the poll — a status response a proxy, CDN or WAF rewrote instead of OpenRouter's — used to
  reach the same chatless caller as a `200` carrying a Markdown card that quoted the first 200 characters of that
  body's own words, which is the body a provider's operator could put there. It now leaves as the same `502` with
  the same endpoint and `Content-Type` and no excerpt, both on the turn that submitted the job and on one that
  joins a job already running. Nothing else about the accounting moved: a rewritten body is still charged nothing
  against the breaker and still marks the turn failed, because the envelope is built after that bookkeeping
  rather than before it. On a channel chat the same card now withholds the excerpt as the rest of the pipe's cards
  already did — the job is watched in a background task that does not inherit the request's chat id, so the card
  reads the one it was handed.

  **Two carve-outs stay, and they are the same two the submit arm already had.** A **streamed** chatless caller and
  a caller on an Anthropic Messages path (`/api/v1/messages`, `/api/message`) still receive the card with the
  fenced excerpt, on this arm as on every other. A begun stream cannot carry a status line, and Open WebUI's
  Anthropic Messages handler re-wraps any streaming response into a converter with no error branch, so an
  envelope on either path would arrive as an empty `end_turn`. So a machine caller that streams, or that speaks
  the Anthropic format, still sees the proxy's own words: the statement above is about the unstreamed
  `/chat/completions` envelope. A prompt a provider echoes back is not masked either — the pipe cannot tell which
  bytes of a reply are its own request — but it is already withheld on a channel.

- **Provider routing panels** — a model's provider-routing filter is no longer detached from it when two model-list
  refreshes overlap. The pass that could not install the panel reported its verdict on a value shared with the pass that
  overlapped it, so a refusal to write could read as an answer and release the panel; the verdict now travels with the
  answer the pass obtained, and a pass that never obtained one leaves every model's filter list exactly as it was.

- **A model-list refresh that outlives its generation no longer writes** — when a hot reload, a valve save or a deleted
  function row retires a Pipe while its own `/api/models` refresh is still running, the refresh used to go on installing
  and repairing filter rows, running the startup stale-id prune, scheduling the metadata sync and dispatching
  `on_models` after that generation's `close()` had already returned — so two generations could write the same rows, and a
  dashboard could be re-pointed at an instance nothing was serving any more. It already returned its own cached rows
  rather than an empty picker; it now also re-checks after the catalogue load, so a close landing underneath a refresh
  that has already started skips the whole write region instead of only the part before it. A refresh on a live
  instance is unaffected, and nothing a person sees in a chat changes.
- **Media on the `/chat/completions` leg** — a turn resolved to the chat endpoint — by `DEFAULT_LLM_ENDPOINT=chat_completions`, by a model in `FORCE_CHAT_COMPLETIONS_MODELS`, or by the `AUTO_FALLBACK_CHAT_COMPLETIONS` fallback — now judges every media link its converter copies, by the same rule the `/responses` leg applies at ingress: the transport (`http://` only with `ALLOW_INSECURE_HTTP` **and** an allowlisted host), the scheme, a link naming this Open WebUI's own `/api/v1/files/` endpoint, and the inline size bound. Such a link used to be copied verbatim onto the wire, and the cleartext valve was never asked about it on this leg. A refused block is dropped from the turn and named on it as `[An attached item was not sent: …]`, in the request side's own words, beside whatever else survived — except an `input_file` that also carries a `file_id`, which keeps that id and drops only the refused link, exactly as on the other leg. An admin who allows cleartext HTTP sees the same link forwarded as before, and a `https://` link, a `data:` URL and raw base64 in `file_data` are untouched. One asymmetry is worth naming: a `data:` video sent to a chat-leg model is now held to the same size bound as on the other leg (`BASE64_MAX_SIZE_MB`, not `VIDEO_MAX_SIZE_MB`), so a clip over it is reported as not sent instead of forwarded.

- **A refused tool picture on the unstreamed `/chat/completions` route** — a picture a tool result carried that the chat-leg conversion drops (over `BASE64_MAX_SIZE_MB`, a cleartext `http://` link, an unusable scheme) is now named to the person on the turn that carried it. The streamed chat leg already did; nothing in the chain above the unstreamed leaf carried an event emitter, so the refusal was collected and dropped. The sentence is the same `Images: skipped N (<reason>).` every other leg uses, so which leg dropped it is not something a reader can tell. A turn that loses no picture still reports nothing, and a headless caller with no emitter still sends its request.
- **A refused picture no longer re-reports itself on every later turn** — under the default `Image input reuse`, a picture an earlier turn's gate had already refused was put into the reuse pool without being measured, so the reuse arm re-derived that refusal and named it again on each later turn for as long as `Image reuse window` could reach the pool — for a picture over `Maximum base64 upload size`, four turns' worth. The pool is now gated on admission, so a refused picture never enters it, is not re-measured on a later turn, and is not re-reported there; the refusal is logged at the reuse arm's own wording and the person still hears about it once, on the turn that carried the picture. Nothing else moved: a picture within the cap is still measured on the same terms at admission and is still reused on the same window, and `user_turn_only`, which never reaches the reuse arm, is unchanged.
- **A refused file is named on the turn that carried it and on no later turn** — the third disjunct of the notice gate (`or status_files`) was not keyed on the turn, so a refusal filed on any turn satisfied the emit condition by itself, on every later request that replayed that history: measured over a six-turn history, one `Files: skipped 1 (…)` status per turn, unbounded. For a caller with no `chat_id`/`message_id` the same unbounded set was joined into `choices[0].message.content`, so pipe prose sat in the value an automation reads as the model's answer, once per historical turn. The gate now carries a turn-window bound (`_NOTICE_LOOKBACK_TURNS`, keyed on the turn index, not on a message index), and the aggregated `attachment_notices` list rides the same predicate, so the body carries the same bounded lines the status did. The operator decided (N25, 2026-10-04) that the window is `0`: a refusal is reported on its own turn only, which is what the sentence in `docs/multimodal_ingestion_pipeline.md` now says. Nothing else moved — the report is not deleted, the turn that carried the attachment still names it, and a turn that refused nothing still says nothing.

- **Presets** — a request naming a preset in the spelling the pipe itself dispatches (`<base>@preset/<slug>`) is now served instead of being refused as blocked. The pipe publishes presets as `<base>:preset/<slug>` and dispatches them with the `@`; the model-restriction gate read the dispatched spelling as a model of its own, so an id copied out of a request log or a response body came back as the `Blocked model message`. The published row was already enforced under the picker spelling, so nothing new is admitted — the two spellings now name one model. A preset turn also stops being billed to the 128 000-token fallback and picks up its base model's real context window, so a tool result the base's window can hold is no longer trimmed away and one it cannot is no longer shipped whole.
- **image generation, a rewritten response body** — when something in front of OpenRouter answers the image endpoint with an error page or a WAF challenge instead of an OpenRouter response, the failure card now shows that text as a quoted code block instead of as ordinary message text. The provider's words are still on the card in full — nothing is shortened, rewritten or hidden — but they are contained, so a beacon embedded in the page no longer makes the reader's browser fetch it and a phishing link in it is no longer a live link. The two facts the card leads with, the endpoint that answered and the upstream `Content-Type`, are unchanged and remain ordinary prose. The fence is carried by the value rather than by a placeholder, so it holds on an installation that has never opened a template, and it is sized past the payload's own backtick run, so a page carrying its own fence cannot escape one. This affects every caller of a picture-only image model, on a surface that is on by default, and it is the same rule the text legs' error cards already applied.
- **Image generation, caller with no chat** — an image request that fails on a provider rejection, or on a body a proxy rewrote, now answers a caller with no `chat_id` **and** `message_id` with an HTTP error envelope instead of a Markdown card, carrying the resolved upstream status on the status line and in `code` (`502` for a rewritten body, on both surfaces). A caller in a chat keeps its card on the same faults. A fault the pipe owns (a blank prompt, a refused caption, storage refusing the file) still reaches every caller as readable text. A streamed image turn now always delivers its card, so a chatless caller is never handed an empty stream; see [Error handling](docs/error_handling_and_user_experience.md).
- **Session log** — with `SESSION_LOG_STORE_ENABLED` off, which is the shipped default, the pipe no longer builds the
  DEBUG request and response payloads at all. They were redacted and serialised on every request and every
  non-delta SSE event, at any `LOG_LEVEL`, and then dropped by the console; the work is now skipped rather than
  done. What storage off now stops is those payload records, and nothing else: the in-memory session log is still
  filled with every other record the request makes, at any `LOG_LEVEL`, so the dashboard's **Log buffers (RAM)**
  counters count those records, not 0 — they are a live count of what the pipe is holding right now, not a
  session-logging indicator. `Log verbosity level` set to `DEBUG` still prints those payloads to the console,
  which never read the archive valve. With the valve on, nothing changes: the payloads are built and kept, as
  before.
- **Artifact retention sweep** — a cleanup pass whose Redis cache purge could not finish no longer deletes anything. Both purge
  arms returned on the first `delete` fault without telling the caller, so the sweep went straight on to the delete: rows the
  operator had been told were purged were gone while the cache entries naming them stayed readable, for a temporary chat under a
  key that embeds its socket id, until the cache TTL expired. The arms now report whether they completed and one gate acts on it,
  so a faulted cycle logs one warning naming the deferral, deletes no rows at all - retention and temporary-chat alike, since one
  arm's unread pages are the other's evidence - and both the rows and their entries go on the next interval instead. Nothing
  changes on a healthy pass, and a deployment with the Redis cache valve off still sweeps: no client means nothing to invalidate,
  which counts as complete.
- **Filter rows, two pipe copies** — in a deployment with two pipe copies, each copy now reads only its own filter rows: the Web Tools filter's stored per-user toggles and cost ceiling for internal Fusion's panel members, and the image-generation model. A row stamped with another copy's install record is no longer read as configuration or executed, whatever id it sits on. The isolation the documentation already promised, and the second copy gains its own image-generation filter row after the next model refresh, starting from the pipe's own drawing-model default rather than the first copy's admin's choice.

- **Timing profiler** — `ENABLE_TIMING_LOG` now writes only its file. The pipe used to keep a second,
  per-request in-memory copy of every timing event alongside `timing.jsonl`, bounded by a request count and a
  per-record character cap, and three help texts and the storage guide described that copy as something a
  running request could be read from. No production code ever read it: its two accessors
  (`get_timing_events`, `format_timing_jsonl`) had no caller outside the test suite, so the copy grew with every
  event the valve recorded and was bounded only by an eviction cap. That copy and both accessors are gone, along
  with the `clear_timing_events` release that existed to trim it. Nothing observable changes for anyone reading
  `timing.jsonl`: the file the valve exists to produce is byte-for-byte what it was, and with the valve off
  nothing was buffered either way.
- **Video attachments, the size the pipe measures** — a clip whose real video size is below the
  model's published pixel floor is now withheld with a notice naming that size, where before an `.avi` or ProRes
  `.mov` was measured wrong and sent for the model to refuse; and a clip whose cover art came first is no longer
  withheld, because the cover's size is no longer read as the video's. The declared geometry is now the picture
  stream's own, read off the field ffmpeg writes it in, so a cover-art stream is skipped and a sub-floor clip is
  left out with a notice a person can act on.
- **Endpoint valves** — a `FORCE_CHAT_COMPLETIONS_MODELS` or `FORCE_RESPONSES_MODELS` entry that names a base model now also
  matches that model's suffixed spellings: the `:free`, `:thinking`, `:nitro`, `:exacto`, `:floor` and `:online` tags, dated
  forms, and any combination of the two. A pattern that names a tag or a date stamp still matches only that spelling, and a
  `:preset/...` tag is left alone. The valve previously built two candidate keys per request and matched both over the whole
  string, so a base entry silently missed its own `:free` twin; a widened pin means those rows are no longer served, or
  retried, on the endpoint the pin excluded.
- **Open-WebUI tool mode** — a call Open WebUI runs now shows its card from the moment the model names the tool, and the dashboard's live view now shows the running tool on those turns (it stayed empty before).
- **Pipe dashboard, deleted function row** — deleting the pipe's function row now releases the live dashboard on the worker that served the DELETE and lets
  the pipe finish its in-flight requests and close, instead of holding it, its session-log threads and its storage handle until a restart. The stand-down
  happens whether or not `PIPE_DASHBOARD_ENABLE` is on, and it is per worker: on a multi-worker deployment the other workers release their own generation at
  their next hot reload, exactly as Open WebUI keeps its own per-worker function cache. The action route and the socket gate still refuse once the row is
  gone; nothing about the authorization answer changes.
- **Pipe dashboard, usage collection switched on after start** — an install whose dashboard model is off but whose usage
  collection was switched on after the worker booted now records usage, and runs the abandon sweep, without a restart.
  The reaper used to be armed only from the boot path and from the model-list pass *below* the dashboard-off return, so
  a collect-only install that was flipped on later never reached it: sessions were tracked, abandoned ones accumulated
  with nothing to finalize them, and the Usage tab's **Failed** count was short exactly the rows nobody was looking at.
  The reaper is now armed by the first request that starts tracking and by the model-list pass above that return, so it
  comes up within one chat turn or one model-list request, whether or not `PIPE_DASHBOARD_ENABLE` is on. No query is
  added to the request path: every one of those sites reads the valve copy Open WebUI rebinds before the call.
- **Pipe dashboard, session tracker bound** — the tracked-session registry is now bounded. At the bound the stalest entry
  is finalized as `failed` — one usage row, carrying that request's own cost and tokens — and a latched warning names
  the bound that was reached. The **Active** tile and the Live table are unchanged: the tile still describes the whole
  tracked population and the table still renders its 30 newest rows, and a tracker under the bound tracks everything it
  is asked to.
- **`/chat/completions`** — a streaming body the provider accepted and then lost is no longer re-sent.
  The `/responses` leg already stopped re-POSTing one; the fallback leg did not, so the same shape cost
  `1 + TRANSIENT_RETRY_MAX_ATTEMPTS` POSTs for a request OpenRouter had already accepted and begun
  generating for. It now costs one, whatever frames the stream had published — visible or not — and the
  fault crosses as `AcceptedResponseLostBody`, which every handler above the transport still reads as the
  connection-class fault it has always seen. The rule is about whether the **body finished arriving**, not
  about the exception class: a body that arrived whole and delivered nothing a reader could use is still
  retried on the full budget and keeps its own class, and a connection that never reached the provider is
  still retried. `DEFAULT_LLM_ENDPOINT` defaults to `responses`, so this is the fallback leg — reached by
  `AUTO_FALLBACK_CHAT_COMPLETIONS`, by `FORCE_CHAT_COMPLETIONS_MODELS`, and by any deployment whose
  `DEFAULT_LLM_ENDPOINT` is `chat_completions` — rather than the one most turns take.
- **Streaming** — a frame the drop valve discarded no longer closes the retry barrier. With
  `MIDDLEWARE_STREAM_QUEUE_MAXSIZE` above `0`, a full buffer used to discard the delta the pipe had just
  published, and the streaming loop raised the retry barrier anyway — so a recoverable rejection right
  after it ended the turn on a card, and the discarded text was lost with nothing on the wire to say so.
  A frame the reader never received has not been published: the emit and the barrier now go through one
  helper that reads the emitter's own delivery verdict, so such a turn is handed back and retried. The
  direction that matters is unchanged — a frame the buffer *did* take still ends the turn, because a retry
  would splice two attempts into the reader's message — and with streaming off `body.stream` still governs
  the leg, since the non-streaming wrapper drops those frames before they leave the pipe.
- **Session log** — a turn the pipe refused before sending is now archived as `error` instead of `complete`.
  The eight pre-send refusals (Zero Data Retention routing in force with the model off the roster, the ZDR
  endpoint list unreadable, Zero Data Retention requested for a model whose request format cannot carry the
  flag, an endpoint-override conflict from a preset or from Direct Uploads, a Fusion model
  forced to `/chat/completions`, an attachment that would not load, and the operator's model restrictions)
  return a card instead of raising, so the job's future resolved cleanly and the archive recorded a turn whose
  chat holds nothing but a refusal card as a clean completion with no cause. The archived `reason` is the
  pipe's own sentence naming the control that refused it, through the operator's own valve labels — not the
  rendered card, which carries a generated error id and a timestamp that identify one message and nothing else.
  The card the person sees is unchanged, and a turn that answered, that a plugin answered, or that the person
  stopped is still archived as it was.
- **session log, per-record bound** — a record that carries a whole payload is now cut at 16,384 characters, and the
  record names the cut: `...(truncated: 1,234 characters omitted)...`. This covers all five payload sites — the
  streamed SSE event, the request headers, the request payload, the response payload and the error response — so one
  1.8 MB frame or one 200 KB reply can no longer become a single 1.8 MB record. The bound is on the serialised record
  after redaction, so a payload that is large because of many small fields is bounded as well as one large field, and
  a payload below the bound produces a record byte-identical to today's. It is the same 16,384 and the same marker
  the tool-result record already used, so there is now one number rather than two.

  This is what `SESSION_LOG_MAX_LINES`'s help text already promised and did not do; the text is now true rather than
  corrected. The cut is in the archive exactly as it is on stdout, and nothing else is cut by it: an exception block
  keeps its whole traceback, the tool-result bound is unchanged, and the error object a card is rendered from keeps
  its own values whole.

  One help text moves with it. `OPENROUTER_ERROR_TEMPLATE` said "the complete values stay on the error object and in
  the session log"; after this change no session-log record carries them for a payload above the bound (the error
  record's own excerpt is cut at 8,192), so the sentence now promises the error object alone and says where the
  session log's copy is cut.

- **artifact retention, retired tables** — a `response_items_*` table that a previous `ARTIFACT_ENCRYPTION_KEY` left behind
  is now swept on the same `ARTIFACT_CLEANUP_DAYS` window as the table in use. Nothing in the pipe addressed those tables at
  all: after a rotation every row it had written — reasoning traces, tool results, temporary-chat rows, and staged
  session-log segments carrying whole request and response content — stayed in the database past every window meant to
  bound it. The retired tables get the age filter and the temporary-chat delete, but no bookkeeping deny-list: no owner
  can address a retired table, so a staged segment there has nothing behind it but this window.

  The pass is scoped by name to this pipe's own fragment, each table is reflected and swept in its own session, one
  table's failure warns and skips without touching the others, and the live table's statements, order and single
  `Retention removed` line are unchanged. Each retired table gets its own INFO line naming it and the row counts.

  **It does not run while `ARTIFACT_ENCRYPTION_KEY` is set to a value the application secret can no longer open.** On that
  arm the worker's own table name is derived from the sentinel placeholder, so its live table is a third table and every
  other sibling it can see belongs to a key it does not hold — including the table of another configuration of the same
  pipe that has simply never set a key. Nothing in the name tells those apart, so the whole pass is skipped: an unreadable
  key costs retention until the key is re-entered, rather than deleting another installation's rows. The live table is
  swept as before throughout.

- **`SERVICE_ERROR_TEMPLATE`** — an admin who has customised this template now sees it on a picture-only image
  generation whose reply a proxy, CDN or WAF rewrote. That card previously carried its own fixed sentence. The fence
  and the withheld-on-a-channel behaviour come with it, unchanged; the video leg keeps its own card shape.
- **Video and image catalogs** — a failed catalogue fetch now backs off instead of waiting out the whole
  refresh interval. Both media lists already stamped their attempt clock on every outcome; what was missing is
  that a failure and a success were indistinguishable to the gate, so a thirty-second provider blip cost the
  picker an hour of missing media rows. The first retry now costs 5 seconds rather than the full hour, and
  consecutive failures double from there — 10, 20, 40, 80, 160 — capped by `MODEL_CATALOG_REFRESH_SECONDS`
  whenever that is set below 160, which is the chat model list's own ladder. The count is kept per OpenRouter
  account and per media catalog, so one key's outage never paces another's media list, and a successful fetch
  of either resets it. A fetch that returns no models is treated as an answer rather than a fault: it stamps the
  clock exactly as before and leaves the count alone, so a proxy that is merely quiet is not paced like one that
  is down. At the shipped default the healthy path is unchanged.
- **Update tab** — an apply or restore is now refused when the pipe's source was hand-edited inside
  the same second the tab was drawn. The tab's revision was Open WebUI's whole-second
  `Function.updated_at`, so a paste in Workspace ▸ Functions that landed in that second left the
  revision exactly where it was; the update then overwrote the paste and still reported success. The
  revision the tab sends and the guard checks is now `<updated_at>:<content digest>`, so the write
  that could not move the revision cannot slip past it. A client that sends only the stamp is still
  guarded on the stamp alone, so a tab drawn by a worker on the previous version keeps working, and
  the refusal is the Update tab's existing "this view was out of date" message. The window is narrow
  and it needs an edit made in the same second as the tab's own load, so this closes a race an
  administrator had to be holding Apply for; updates are not otherwise riskier — nothing else about
  the apply path changed.
- **Provider routing panels** — every installed routing panel's code is replaced on the next model-list refresh, and
  what changed in it is entirely in the log. A stored choice the panel cannot map (an `ORDER`, `ONLY` or `IGNORE`
  dropdown value with no entry in the routing map) now warns **once** at `WARNING` and repeats at `DEBUG` on every
  later turn, naming the field and the value; before, one such choice meant three WARNINGs on every request for as
  long as it stayed open, with no throttle of any kind. The choice itself still goes out unconstrained exactly as
  before — nothing about the request changes. This entry also covers the two valve-healing warnings in the same
  generated panel, whose latch declarations were renamed to the spelling the pipe's own warn-once inventory reads, so
  that they are reset between runs instead of staying armed for the life of the process.

- **inline video and the tool budget** — a turn carrying an inline video clip on `/responses` no longer shows the
  "this conversation needs about N tokens" warning, and no longer drops a tool result, because the context budget
  now rates the clip under the name the wire gives it. The two spellings of a video block — `video_url` before the
  responses rewrite, `input_video` after it — were priced from different rows of the rate table, and only the first
  had one, so every pass that ran after the rewrite charged the clip's base64 at its literal length, some 125× the
  clip. The turn then read as hopeless and the budget stopped trimming tool output at all. Reach is occasional:
  inline `data:` clips only, and only on `/responses`; a linked clip was never affected, and neither was a turn
  without a tool result to lose. What it cost was the model answering without a tool result it had been sent.

- **Update tab after a refused write** — the dashboard now follows the rebuilt instance on every surface, so the
  Update tab, the Usage tab and the Live card stop disagreeing about which generation is live. A refused write no
  longer leaves the Live card on the retired generation's sessions, and no longer leaves Update and Usage answering
  "unavailable" once the rebuilt instance itself retires: the panel's registrations are taken back down with the
  generation that made them, instead of being pinned for the life of the worker.
- **Provider routing** — the per-model endpoint sweep behind the routing dropdowns is now bounded as a whole at 45 s
  instead of running one per-read cap at a time. With 50 slugs listed against a slow or unresponsive OpenRouter
  endpoints host, the old fan held a metadata pass for roughly 25 minutes while it ran its legs in ten-wide waves,
  and the metadata-sync key stayed committed the whole time, so every later schedule was refused. The pass now comes
  back, logs how many of the listed models were still unfinished at `WARNING`, and carries on. An abandoned slug
  keeps its provider data from the last cycle and has its routing row left exactly as it is — for an admin-enforced
  routing filter, that is the difference between a stale dropdown and a setting the pipe switched off on its own.
  Nothing is announced as a spelling problem: a slug whose read *finished and failed* still is, exactly as before.
  The cost is freshness on a slow host, never correctness, and `HTTP_TOTAL_TIMEOUT_SECONDS` remains the lever for how
  long a single read may take.
- **provider routing** — the two WARNINGs a generated routing filter raises when it heals a
  stored row no longer name the stored value. They name the field and what the filter kept
  instead, because a valve row holds whatever was typed into it — a `data:` URL, a signed
  link, a paragraph of prose — and none of that belongs in a log. What the warning was for
  is unchanged: it still says which field went stale and what it was replaced with, and a
  value that is still on the option list still produces no record at all. Open WebUI's own
  filter warnings name the filter and never the payload, and these now match that.

  **Every installed routing filter is rewritten on the first model-list refresh after this
  update.** The two strings live in the generated Python source, and the pipe compares each
  row's stored source against freshly rendered source byte for byte, so the first refresh
  rewrites all of them — once, for this change, and silently. Nothing about the rows'
  scope, ownership or behaviour changes with it; the rendered code differs only in the two
  format strings. If you have hand-edited a routing row, that edit is reverted by this
  rewrite for the same reason any other regeneration reverts one.

- **Reasoning summary blocks** — on a `/chat/completions` turn carrying two or more `reasoning.summary`
  blocks in one round, the thinking box and the status line now read them as separate blocks separated by one
  space rather than as one glued word (`Name the tradeoff.Then weigh it.`). The blocks are joined the same way
  by every producer the loop folds together — both chat-completions arms, the non-streamed chat leg and the
  `output_item.done` snapshot — so the round still reaches the person once, as two blocks. A block's own text is
  untouched: a block that already ends in a space does not gain a second one, and internal whitespace is left
  exactly as the model wrote it. The stored reasoning row and the closing item are unchanged.
- **Config tab** — a save the database refuses to write now raises a durable banner instead of a toast alone.
  The banner names the fault, carries no Reload control, and leaves your staged edits in place, so nothing the
  refusal preserved can be discarded from the tab that preserved it. It clears on the next successful save or
  the next configuration load; today's toast is unchanged.
- **Auto-update** — the pause line now names *why* a release was paused, not just the code. The Auto-update
  row carries the version it stopped on and the refusal's own sentence, versions included; the
  `(apply manually or restart to re-arm)` hint is now shown only for a pause reason one of those can
  actually clear, and withheld where a manual apply re-runs the same guard.
- **frame reuse** — a frame the pipe extracts from a prior video is now bounded to 1920 on its **long edge**, not on
  its width alone. A portrait 4K phone video keeps its continuation frame instead of losing it with a ⚠️ note: a
  2160×3840 source now yields a 1080×1920 frame where it used to hand back a 1920×3414 one, past
  `VIDEO_FRAME_IMAGE_MAX_BYTES` at the shipped default, which dropped the frame with a note in the chat. Landscape
  sources and portrait sources at or under 1080×1920 are unchanged — the bound was already inside them.

  Detail quality on 9:16 sources is what this trades: 2160×3840 loses 3.2× its pixels, which is the point of the
  change and still leaves the anchor frame above 1080p-equivalent detail.

  Reach is common but content-dependent, not universal. A PNG's size follows entropy, not geometry, so the same
  shape keeps or loses its frame depending on the picture: busy, grainy, low-light or fine-detail 4K portrait clips
  lose it, clean or smooth ones keep it. What made it worth fixing is that the boundary is a knife edge — one stop
  brighter or one notch sharper flips it — and nothing beyond a ⚠️ line said so.

- **frame reuse, memory** — with the long-edge bound in place, the extractor asks no decoder for a frame larger than
  1920 on either axis, whatever the source, so its per-frame footprint is now bounded at about 3.7 Mpx and 10.6 MiB
  instead of scaling with the source's short side. A geometry the pipe cannot measure at all — a video line carrying no
  size, a header read that fails, or zeros arriving from the probe rather than from a second read — is now treated as
  unbounded and bounded at the decoder anyway, instead of turning the scale filter off and decoding the whole frame at
  native size first; the filter is `min(1920, iw)`, which cannot invent pixels, so applying it to a source of unknown
  size cannot make that source look smaller than it is. The ffmpeg arm's own pre-child gate is gone: it compared the
  scaled size the header predicted against the 25-megapixel budget, a comparison that could never fire at the shipped
  values (the largest product it could reach is 1920×1920, 3.7 Mpx, against a 25 Mpx cap), so what it cost was a
  container-header read on every extraction whose probe had measured no geometry. The pre-decode refusal and the
  decoded-size checks behind it stay in the code for a header that lies about the source.

- **Pipe Dashboard, Config tab** — when two administrators save at the same moment on *different* workers, one
  save is now refused with "nothing was saved" instead of being silently lost. Until this, both saves were
  answered "Saved 1 setting": the per-pipe lock only ever serialised saves on one worker, and the revision it is
  checked against is a whole-second timestamp, so two saves inside one second are one revision apart by nothing.
  The refusal appears as the Config tab's own save-failure toast, names the other save as the cause, and leaves
  your edits staged. Installs without Redis (`WEBSOCKET_MANAGER` unset) keep today's behaviour exactly.
- **dashboard** — the System tab's **Readiness** panel now reports how much memory the session-log buffers hold, not only how many there are and how many records they keep. The `Log buffers (RAM)` row reads `2 buf / 40 000 events / 12.4 MB`, and a request whose records rolled off the top under the record cap says so on the same row. The cap counts records, so the counts alone cannot distinguish a quiet afternoon from a request holding a gigabyte of provider payloads in RAM.

  The figure is the number of characters the held records actually carry, and it is counted where a record is appended rather than recomputed by the panel — so opening the dashboard on a busy worker costs the same whatever the buffers hold. A record dropped by the cap is subtracted as it is dropped, a change to `Archive record cap` is recomputed from the records that survive it, and a request's figure goes when its records are released.

- **thinking boxes** — a reasoning box that grows in the middle of an answer no longer cuts that answer in two. The pipe used to flush the pending text into a message item of its own ahead of the box, so the reply a reader got back carried a line break the model never wrote, and a Fusion turn got a second copy of its answer. The answer is now one segment across a growth; only a **tool card** still ends a message item.

  This changes what the next turn is **replayed**: Open WebUI rebuilds the stored answer from the output array and hands that string to the model, so a turn that grew a box mid-answer now replays without the break that the flush used to add.

- **reasoning summaries** — a provider that fragments one `reasoning.summary` now keeps every fragment. The chat adapter kept a single scalar per summary key, so on a model that sends disjoint fragments under one `index` all but the last were overwritten before anything downstream saw them, and a cumulative snapshot was added on top of what had already been delivered. Every fragment now reaches the thinking box, the closing record and the replayed `reasoning_details`, exactly once and in arrival order.

- **API callers, streamed** — a 502 that used to reach a streamed API caller as assistant text now reaches it
  as an in-band error frame. An API caller with no chat to write a card into used to receive the error card as
  the model's own answer on a streamed call: an HTTP 200 whose body was Markdown, indistinguishable from a
  reply, with no status to branch on. A streamed turn cannot carry an HTTP status at all — Open WebUI hands a
  pipe's non-2xx back inside a response with no `status_code`, so it would arrive as a bodyless 200 — so the
  failure is now delivered as a terminal `{"error": {"message": …, "code": …}}` frame carrying the upstream
  status, byte-for-byte the envelope the non-streamed leg already sent. Chats, temporary chats, internal Fusion
  members and both Anthropic Messages endpoints keep their card exactly as before.

- **address checking** — a request's `ADDRESS_CHECK_BUDGET_SECONDS` is now spent by address checks and by nothing else.
  A remote picture's **transfer** used to come off that budget, so a turn carrying a few slow pictures could leave a public
  video link or a file link behind them with no time to resolve, and the link was refused with the sentence for a check
  that never finished. The download's own address check still draws on the budget; the bytes it moves afterwards do not,
  and the valve help text, the dashboard entry, the atlas and the security and multimodal documents now say so in the
  sentence that already promised it.
- **pictures** — one request now resolves at most 16 addresses, whatever the picture count. `Maximum images per request` only ever counted
  what was **forwarded**, so a picture the address gate refused consumed no slot and a message carrying two hundred unfetchable remote
  pictures paid two hundred lookups on top of the one the pipe downloaded each. The ceiling applies to every forwarding arm — the
  cold picture, a picture reused from an earlier turn, the download's own check, a picture a tool result carries, a file link and a
  video link — and a link past it is refused without being looked up and without being downloaded, with the count named in the refusal.
- **fusion** — a Fusion panel member's tool file is no longer filed against the outer chat. The file is still
  stored and still rendered in the panel, but it no longer appears in the chat's file list: the tool executor
  re-checks `fusion_inner` before it hands `chat_id`/`message_id` to Open WebUI's upload, so a member's file
  cannot produce a `chat_file` row naming a real chat in a message that does not exist.
- **image contracts** — repointing `BASE_URL` now drops the shared published image contracts immediately, and not
  only when a contract sweep happens to run. The drop sat below the sweep's staleness gate, so a pass with nothing
  stale to do returned before it, and a gateway admin running with the four image filter valves off kept being
  served the previous gateway's aspect-ratio, resolution and seed limits until some later sweep ran. The identity
  is now adopted before the gate, so one stored value (`API_KEY`) has one reader and one answer.
- **provider error row** — on a failure reported inside a reply the pipe has already started, the **Provider error**
  row now names the provider's own error type wherever the body carries one, read from `error.metadata.raw.error.type`
  or from the failure's own `error.type` — the same two places the rejection path already reads — instead of an HTTP
  status number or the literal word `error`. Nothing else on the card moves: the resolved status, the typed kind, the
  template choice and the retry decision are unchanged for every shape, and a body that names no provider category
  still falls back to the numeric code and then to the stream marker. Custom templates using `{upstream_type}` are
  affected, and on a channel chat it is now withheld along with the rest of the provider's prose about the request: a
  card that reaches a room renders it as though empty, so a **Provider error** row guarded by `{{#if}}` is left out
  there, and an unguarded one omits its whole line. A saved chat, a temporary chat and an API caller are unchanged.
- **API key** — a stored `API_KEY` the key gate refuses now produces the authentication card on the streaming and the
  housekeeping-task legs too, instead of a 401 or an opaque "Unexpected error in streaming loop". Both of those legs
  read the stored field with a bare decrypt instead of through `Pipe._resolve_openrouter_api_key`, so an encrypted
  value that cannot be decrypted went out as an empty `Bearer ` and an encrypted non-`sk-` value the gate refuses went
  out working; a stored value with padding was sent with its padding. Both legs now go through the gate, so one
  misconfiguration has one answer on every leg.
- **Video turns on the dashboard** — a video turn now books its real usage on the dashboard, and a failed one counts as Failed rather than Completed. The video adapter reports its own terminal state, so a Usage row carries the job's real tokens and cost where it used to carry zeros, and a video that failed, stored no clip, or stalled no longer reads as a free success. A second request that attached to an already-running video job reports under its own id without that job's usage, so one billed clip is one row's spend.
- **model icons** — a model icon is now stored the way it displays. The icon sweep applies the orientation the
  source published before it writes the PNG, so a logo stored sideways is no longer stored sideways; an icon
  already stored keeps its pixels until its source URL changes, which is when it is downloaded again.
- **reasoning budget** — a `reasoning.max_tokens` written as a number in a string (`"2048"`, `" 2048 "`,
  `"2048.7"`) is now read as that number, on both endpoints: it is reserved against the request's own output
  cap like an integer budget, and on the Gemini 2.5 leg it overrides `GEMINI_THINKING_BUDGET` instead of being
  replaced by it. A value that expresses no number (`"abc"`, `""`, a list, a boolean, `"inf"`, `"1e400"`,
  `NaN`) used to reach OpenRouter verbatim as the budget and produce a failed turn; it now produces a turn with
  **no thinking budget**, which on a wide model is a slower, more expensive turn rather than an error. Nothing
  is logged for a dropped value, as the carried-field convention has always been.
- **responses streaming** — every SSE frame on `/responses` is parsed exactly once per turn now, by the
  reader that decodes it, and the workers forward that parsed frame instead of parsing it a second time. The
  pre-first-output buffer therefore holds parsed frames rather than byte strings, so a stream whose opening
  frames a fallback has to discard no longer keeps the raw bytes of those frames in memory.
- **integer request fields** — a non-finite value in `seed`, `max_tokens`, `max_output_tokens`,
  `max_completion_tokens` or `top_logprobs` no longer ends the turn. `"inf"`, `"-inf"`, `"nan"`,
  `"1e400"`, `inf`, `-inf` and `nan` all raised `OverflowError` out of the field validator, which pydantic
  does not convert, so the request died before dispatch and the chat got an *Unexpected Error* card plus a
  failure-budget strike. The field is now simply sent unset, which is what the float fields beside it
  (`temperature`, `top_p`, `top_k` and the rest) have always done with the same spellings. A bool, a list, a
  dict and a non-numeric string are still refused with the same message: those are mistakes, not infinities.
- **per-model filter installs** — a single model's image or video panel failing to install is now named once
  per model per kind of failure, at WARNING, and repeats at DEBUG with the same message and traceback. A
  catalogue of a few hundred models against a database that is refusing writes cost a few hundred identical
  lines on *every* refresh, which buried the one line that names the model. A model that starts failing
  *differently* warns again at WARNING. The whole-pass lines are a separate latch, and the image one has
  now moved with it: `OpenRouter Image filter ensure failed` warns once per cause — per exception class,
  for as long as the fault lasts — and every later failure of the same kind is recorded at `DEBUG`, where
  the Fusion, Web Tools, Image Gen and Direct Uploads lines already warned that way.
- **`Files` boxes** — the pipe's own untick of the `Files` box inside `Built-in tools` is now recorded on
  the model, under its `openrouter_pipe` metadata as `builtin_tool_defaults`, and it is unticked only on a
  pass where the pipe unticked `File context` itself. A `Files` box you unticked by hand is no longer changed
  as a side effect of that, and is never recorded as the pipe's. A model row written by an earlier release
  carries no such record and keeps the box it has; an older pipe ignores the unknown key.
- **temporary chats in the log** — a temporary chat's browser socket id is no longer written to the pipe's
  log on the Direct Uploads path or on the dropped-row path; both records now read `<not retained>` in its
  place, the same literal the video-intent path already used. The artifact-replay records join them, so the
  pipe has one spelling for a withheld chat id rather than two. A saved chat's id is still named in full on
  every one of them, and the record's counts and row fields are unchanged.
- **`ModelFamily.capabilities`** — removed. It was an accessor no production code called, kept only for tests;
  `spec["capabilities"]` is still written, still read by `list_models()`, and still merged into Open WebUI's
  `meta.capabilities`, so nothing about a model's checkboxes in the model editor changes. A third-party
  Open WebUI plugin importing the pipe and calling this one accessor would break; nothing inside this package
  does.
- **Log redaction no longer copies the payload it removes** — a record or a debug payload carrying a
  multi-megabyte picture now costs a few kilobytes to redact instead of a copy of the picture. The redaction
  answers the same way as before — a run of 1024 or more base64 characters is still cut to 64 characters
  with the same marker, and the prose around it is still left alone — but it reaches that answer by
  reading spans and indices instead of materialising what it is removing: the `sub` replacement was built
  from `match.group(0)`, which is the whole run; the guard asked `"data:" in text.lower()`, and `str.lower()`
  copies; and `url_scheme` handed a multi-megabyte string with no colon in it to `urlsplit`, which copies it
  to build a `path`. A record's redaction runs on whatever thread emitted it, so on a busy worker that copy
  landed inside whatever else that thread was doing. Nothing an operator sees changes.
- **One user's `Set-Cookie` is no longer replayed onto the next user's request** — every
  `aiohttp.ClientSession` this pipe builds is now given an `aiohttp.DummyCookieJar()`, so a
  `Set-Cookie` on any response is discarded rather than queued. This was the batch's only
  cross-user exposure, and it was live: `openrouter.ai` returns one `Set-Cookie` on both
  `/api/v1/models` and `/api/v1/key`, named `__cf_bm` — Cloudflare's bot-management cookie,
  measured against a test key on 2026-10-02, values not recorded. Every user's later requests
  therefore carried the same Cloudflare cookie and Cloudflare saw them as one client. It is
  **not** an authentication cookie, and no credential left one user and reached another through
  it. Behind a gateway at `BASE_URL` that sets a session cookie of its own (an auth proxy,
  `oauth2-proxy`, an nginx `auth_request`), one user's cookie could have reached another user's
  requests; whether your own gateway does this was not measured. The three affected sessions are
  the pooled per-event-loop session every concurrent request shares, the vetted transport's
  session (which fetches model icons, maker pages and admin-configured release assets — addresses
  somebody else chose), and the one `_ensure_async_subsystems_initialized` opens and hands to
  plugins as `ctx.pipe._http_session`. Nothing else changes: no valve, no timeout, and nothing
  in the package read a cookie before or reads one now. Operator-configured gateways and
  catalog icon hosts were not probed.
- **Temporary Chat error cards and records** — a Temporary Chat's socket id no longer appears in its error card or in
  the log records the fix names. Open WebUI mints a temporary chat's id by prefixing the browser's own socket id, and
  that id is what `/api/tasks/chat/{chat_id}` and `…/stop` resolve back to the socket and then to the owning user as
  their ownership check — so printing it on a card, or in the operator's own log line, handed out a live handle rather
  than a correlation string. `session_id` is now withheld on a `temporary:` or `local:` chat across every error path
  (the templated card, the provider-rejection card, the non-streamed auth card, the card returned to an API caller with
  no emitter, and the three queue-backlog warnings at both `WARNING` and the demoted `DEBUG` repeat), reduced to the
  empty string rather than hashed or truncated. `error_id` remains the correlation handle and `user_id` is not
  withheld — it is the account GUID, present in every chat. A saved chat and a `channel:` chat are byte-identical to
  before, and a channel card still withholds the ids while the operator's channel log line still carries them.
- **session log storage** — an archive written under the pre-digest path component is now found and merged by the
  next assembly pass for that turn, which keeps the outcome the older archive recorded and republishes the merged
  turn under the digest name. Only when the older file's `meta.json` names that turn's exact `ids`, so an archive
  belonging to another turn that used to share the bare stem is left alone, and the file that was read is left
  where it is for the retention sweep to reap.
- **session log storage, an over-budget id with no task name** — a caller-supplied message id longer than
  the 64 characters the archive key is written to is now reduced on the **answer** archive too, not only on a
  task archive. The answer arm used to return the id verbatim, so on PostgreSQL — where that column is
  `String(64)` and the row is refused outright — the turn's staging write failed and fell through to the
  queued-zip fallback, which composes a still longer name, so the condition persisted for that turn. The
  reduction is the same one a task key already got: the stem the budget allows plus a short digest of the
  exact id, never a cut, so two ids that differ anywhere still reach two files. Every id Open WebUI mints is
  a `str(uuid4())` at 36 characters, so no browser-path archive changes; a caller-supplied id past the budget
  changes its key and its filename (and, until the exact id is carried in `meta.json`, its `ids.message_id`
  too), and an archive already written under the raw id is found and merged into the digest-named one.
- **pipe dashboard** — the usage retention window now covers every `dashboard_*` table this pipe published and a key
  rotation left behind, de-identifying its temporary-chat rows as it does the current table's, and the purge runs
  whether or not `Collect usage records` is on. A table whose fragment another installed function id also sanitizes
  to is skipped and named once in the log, and no table is ever dropped — only emptied.
- **Pipe Dashboard, action route** — the dashboard's `POST /api/pipe/dashboard/action` route now answers
  `401` for a request that Open WebUI itself would refuse. The route validated the bearer token, re-read the
  user row and checked the role, but never compared OWUI's `WEBUI_AUTH_TRUSTED_EMAIL_HEADER`: on a deployment
  behind a proxy that authenticates by trusted header, a request whose header named a **different** person than
  the token's own user was admitted, and the admin-only write actions behind it (`config_set`, `update_apply`)
  ran. It now compares exactly as OWUI's `get_current_user` does — the header value lowercased against the
  stored `user.email`, and only when the setting is on *and* the header actually carries a value — so the
  comparison is presence-optional and an API-token caller that sends no such header is unaffected. The
  property the route is built on is unchanged: the comparison reads a request header and never the session
  cookie, so the route stays CSRF-safe regardless of OWUI's CORS/SameSite settings. With the setting off —
  the default, and every deployment that does not run one — nothing changes at all. The header's *presence* is
  not required and an empty value is not a mismatch — both are skipped exactly as Open WebUI skips them, so a
  proxy that stamps the header on `/` but not on `/api/pipe/dashboard/action` does not lose the admin surface.
  The check is read from the header name Open WebUI itself bound at startup, so it names the header your
  deployment is actually using. The same branch now also refreshes the caller's "last active" time, which
  Open WebUI's does, so an operator reading that field to spot idle admins no longer sees dashboard-only
  admins as inactive. Both live in the route's own glue and do not follow a hot reload: they take effect at
  the next worker restart. The socket leg is
  untouched: OWUI performs no such comparison there either.
- **Video, prior-video frames** — the frame extracted from a prior video is now **read** as the person who
  asked for it, not as the pipe's storage service account. The read that copies a prior video out of Open
  WebUI storage to extract a frame was authorised with the storage account — an `admin` by default — so a
  turn whose user could not be resolved read any file in the deployment, including videos belonging to other
  people. Reads now run as the request's own resolved identity, which is what Open WebUI itself gates file
  content on; a turn with no signed-in user gets no prior frame and is told so (`⚠️ Previous video could not
  be loaded.`) rather than quietly sent one. The **upload** of the extracted frame is a write and is
  unchanged: the storage account still owns the file the pipe publishes. One consequence worth naming: a
  video turn driven by API automation, with no signed-in user, loses its prior-video frame. That is the
  intended outcome — Open WebUI gates file content on the requester and never on a service account — and the
  only alternative would be a pipe-owned service identity for prior-video reads.

- **Unencodable characters in a provider's reply** — a provider's unencodable characters reach the card and the
  log record as U+FFFD instead of losing them and the archive.
- **Fusion, Continue of a Fusion answer** — the closing Fusion panel is an `embeds` event that Open WebUI saves,
  not a write by the pipe, and the Fusion answer travels as a native output item that Open WebUI folds. On a
  Continue the stored answer is therefore never cut back to this generation's text: the panel replaces only the
  earlier Fusion panel on the message, and the half the Continue is continuing from stays with it.
- **A refusal that opens a Continue** — the refusal now starts a block of its own, so the separator Open WebUI's
  join needs is emitted before it. A refusal-first Continue had none: the refusal arrived glued to whatever followed
  it, and when the stored reply ended on a hidden marker line the two fused into one line the marker parser no
  longer read -- the continuation's own segments were lost with it. A stored reply ending on ordinary text gets the
  blank line it needed for the same reason. Nothing changes on a fresh turn, and nothing changes on a Continue
  whose stored reply is the empty string: there is no stored line to separate from, and no separator is added.
- **Unrecognised content block, a link in any letter-case** — an opaque block type (`vendor_widget` and friends)
  is forwarded verbatim, and the one thing that stops being forwarded is a block that carries a *location*. The
  test for that was a prefix comparison and a substring search on the raw value, both case-sensitive in a way the
  URL grammar is not. `HTTPS://169.254.169.254/latest/meta-data/`, `FTP://…`, `Gopher://…`,
  `HTTPS://OWUI.EXAMPLE.COM/API/V1/FILES/abc/content` and a `//host/x` behind a leading newline are all the same
  links as their lower-case twins, and every one of them — credentials included — went to the provider inside a
  block the typed arms would have refused. The pipe's own URL parser now decides, at whatever nesting depth and
  under whatever key but a prose key, so a block that is refused today stays refused and the mixed-case spellings
  join it. Two shapes are pinned as staying dropped rather than left to a parser's accidents: a `data:` URL and a
  `javascript:` URL whose *payload* names the file endpoint, neither of which is a link the provider would dial
  and both of which were dropped before. One value is deliberately left alone, because it names no address and no
  endpoint: `Api/V1/Files/abc/content`, with no leading slash.
- **A tool's picture that names a host** — an absolute URL a tool result carries is now put to the
  plaintext and address valves like any other link. The pipe has always had a classifier that answers "is
  this one of my own file endpoints?" from the *path* alone, and the six forwarding gates were reading it
  as "and therefore this is not a link" — so a path a tool wrote into a URL skipped the checks: a
  cleartext `http://169.254.169.254/api/v1/files/abc` skipped the plaintext valve, an `https://` one on a
  link-local or private address skipped the address check entirely, and one written as
  `https://user:pw@host/api/v1/files/abc/content?token=SECRET` skipped both and reached OpenRouter with
  the tool's own credentials in it. A path is not an origin, and `ENABLE_SSRF_PROTECTION` and
  `ALLOW_INSECURE_HTTP` promise to hold on every forwarding site, so the gates now exempt only a
  **relative, host-less** reference whose parsed path *begins* with `/api/v1/files/` — the spelling Open
  WebUI itself mints, which stays exempt on every valve and is asked about zero times, as does any
  letter-case of it. Everything else with that path in it (`file:`, `//host/…`, `\\/\/host/…`,
  `https://host/…`) goes through the cleartext valve, the scheme gate, the address check and the verdict
  memo on every leg, and a refused picture is skipped with a status; the turn is not lost. **What this
  costs:** a deployment whose Open WebUI is on a private or link-local address loses an absolute self-link
  from a tool (`https://10.0.0.5:3000/api/v1/files/<uuid>/content`) — it is skipped rather than inlined,
  because the pipe has no configured Open WebUI origin to compare against and comparing paths would reopen
  the hole above; that is filed as its own item. Five relative spellings that hide the path behind a dot
  segment, a query value or a fragment (`./api/v1/files/abc`, `/proxy?path=/api/v1/files/abc/content`) are
  now refused `not a link the pipe can resolve into an image` on the tool gate; they still resolve on
  every attachment arm, where the host-agnostic classifier is what runs. `/chat/completions` is now part
  of the same promise: an `image_url` block naming a file reference is resolved from local storage before
  the request leaves instead of being forwarded as a link. Nothing about the attachment arms, the video
  arm or the host-agnostic resolve classifier changes.
- **Task-model failure notices in shared rooms** — when a task model (titles, tags, follow-ups) fails in a `channel:`
  chat or a saved chat several people share, each member now gets their own one-time notice. The notice used to be
  latched per conversation, so the first member's notice silenced everyone else in the room for the window. The DEBUG
  lines that record a suppressed or undelivered notice (`task-failure toast suppressed: …` and `task-failure toast not
  delivered; latch left open: …`) now name the user id beside the chat id for a saved or `channel:` chat; a temporary
  chat's line still names no chat.
- **error templates** — a stored operator row that fences a value on its own lines now renders a code block that
  closes on its own opening run. When the provider body carried a fence run long enough to widen the block, the
  template's own closer stopped being recognised and the card's closing guidance was swallowed into a block that ran
  to the end of the card; an indented opener row was rewritten from column 0 and emitted a run the closer did not
  match. No shipped template fences a value, so a card from one of the built-in templates is byte-identical.
- **attachments** — a content block whose conversion raises is now dropped and named in the `Files: skipped N (…)`
  status on your turn, as every other refused attachment already is, instead of being forwarded in the shape
  Open WebUI handed it over — a block the provider does not accept, carrying text the pipe had not finished
  preparing. The turn itself is unchanged: the model goes on with whatever else it carried, and a turn whose only
  block faulted still says so in-band. A block the pipe has no converter for is still passed through untouched.

- **Video attachments and the file host** — three changes to what a turn says and does when an attachment goes to a
   public file host. A **reference picture** over `MEDIA_FILE_HOST_MAX_SIZE_MB` — over the cap on its own, or over what
   one request may publish put together — is now left out with a notice naming it and the cap it broke, and the rest of
   the turn is generated as asked, instead of refusing the whole request; a **clip or a sound file** over the same cap
   still stops the request, naming the cap, because a clip the person attached is the thing being edited. The durable
   record's **"may have been uploaded"** line now speaks for each attachment that reached each host rather than being
   suppressed whenever another attachment of the same kind landed on the same host, so an ordinary two-clip turn records
   both, in one block; and it names that host's own retention, hedged on the file being there at all, instead of
   promising permanence on a host that deletes its own files. The **second-host failure card** no longer says "Nothing
   was uploaded" when an earlier attachment in the same turn already reached a host: it says that nothing further was
   uploaded, that an earlier attachment is already on a public host, and that sending the turn again will not take that
   one down. All four need the opt-ins below, which ship off.
- **Tool shutdown at `TOOL_SHUTDOWN_TIMEOUT_SECONDS=0`** — the per-request `WARNING` an operator who chose that
  documented quiet value saw on every turn stops appearing. A tool context is built for every request, whether or not
  the turn called a tool, and shut down at the end of the job, so the valve's own "there is no grace wait" path was
  the one path that always fell through into the wait-timed-out handler and wrote `Tool shutdown exceeded 0.0s;
  cancelling workers.` — a line describing a stall that could not have happened, on precisely the installations where
  cleanup was instant and deliberate. Nothing was waited for before either, and nothing changes now: the workers are
  still cancelled at the same instant, the reply is unchanged, and log-based alerting on `WARNING` stops firing on a
  configuration that is working as documented. The valve is now a positive guard rather than a synthesised timeout, so
  a wait that genuinely expires still warns exactly once and still names the valve's own value; and a DEBUG line
  records that the grace period was off, which is what tells an operator why the workers died instantly (genuinely
  abandoned tool tasks are left behind on this path).
- **Empty provider-error cards** — the card a reader gets when the operator's own template renders to nothing now
  names their configured support contact, on a `channel:` read as well as an ordinary one. The stub that replaces an
  empty card named the model and an error id but no way to reach a human, so on exactly the path where an admin's
  wording had been withheld from a channel reader, the card they were left holding had no support handle either. The
  two rows appear only when `Support email` or `Support link` is set, in the same guarded shape the shipped templates
  already use; with neither set — the shipped default — the card renders exactly as before, byte for byte, and the
  shipped 400 card is untouched. A non-streaming API caller reads the same text off `choices[0].message.content`, so
  they gain the same handle in the same string.

- **Model metadata, a stored grant the access field cannot spell** — on an Open WebUI whose `ModelForm` carries
  `access_control` instead of `access_grants`, a metadata refresh used to drop every stored grant whose principal type
  is neither `user` nor `group` (`anyone` today — "anyone, including people who are not logged in, may read this
  model"), and a wildcard read took every other grant in the set down with it on the way out: the converter answered
  the public wildcard before it read the rest, so `user:* read` plus `user:alice write` came back with neither. Both
  answers the dict can give move the row, which is why neither is used. `{}` replaces a shared model with a private
  one; `None` is Open WebUI's own public wildcard (`user:* read`), so a grant that conferred read on nobody would
  confer it on every signed-in session instead. The pipe now refuses the write and says so. The row keeps exactly the
  grants its admin set, its icon and its description stop being refreshed, one `WARNING` per row names the model and
  the principal type it could not carry, and on a sync pass the refusal is counted with the pass's other unwritten
  models — that count is the `WARNING` line to look at, with the per-model detail at `DEBUG`. Nothing changes on a
  stock install: Open WebUI 0.11.4's `ModelForm` carries `access_grants`, so the conversion never runs there. Where it
  does run, a wildcard read keeps the rest of its grant set instead of discarding it, and a set the dict can spell
  still round-trips unchanged.
- **Video intent, disclosure on long chats** — a chat past the classifier's window (24 conversation rows or 9 prior
  videos) no longer opens the Intent Disclosure Block on its own. Those two window notes change what the classifier
  *read*, not what was *sent*, and on a turn that lost nothing they were putting a full block on screen for a note
  about nothing that was dropped. People see fewer disclosure blocks on long chats, and the loss is disclosure: the
  note about the classifier reading only part of the chat still appears whenever the block shows for another reason —
  a rewritten prompt, a reused prior-video frame, or any recorded loss — and always under
  `VIDEO_INTENT_CONFIRM_MODE='always'`. A turn that *did* lose something opens the block exactly as before, and
  `VIDEO_INTENT_CONFIRM_MODE='never'` still renders nothing at all.

- **A `data:` header that is not a media type names nothing, on either reader** — both of the pipe's header readers took
  everything between `data:` and the URL's first comma, cut it at its first `;`, and printed up to 64 characters of it.
  A 64-character truncation is not a validation, and the header is the caller's own text: a remote server answering
  `Content-Type: Q4 Payroll - Margit Okonkwo` is enough to put that text in the header, because Open WebUI interpolates
  a remote server's raw `Content-Type` into the data URL it builds (`backend/open_webui/utils/files.py:96-97`). Both
  readers now answer through the pipe's own `media_type_or_empty` — the same predicate the read path already used, which
  accepts a header only if it really is `type/subtype` — so a header that names no media type contributes nothing
  rather than a prefix of itself. **This changes one thing on the chat leg, deliberately:** an inline `input_audio` block
  whose `data:` header is not a media type is no longer refused as "a format the pipe will not rename" (a refusal about
  the caller's spelling rather than about a format); it now declares nothing, so the block's own `format` decides, and
  a block with no `format` at all keeps the `mp3` default — which is what the tree already did for an *empty* header.
  A well-formed type is unchanged, and a media type that was echoed verbatim is now lower-cased (`data:IMAGE/PNG` reads
  `data:image/png`), which is RFC 2045 case-insensitivity and the same normalised form the read path and the free-text
  log leg already return. In a log a refusal subject for such a URL now reads `data:` — the scheme is known and named,
  the payload is not — and a file refusal with no other source reads `no source`, as it already does when nothing else
  can describe the link. No valve and no configuration change.

- **Image generation, the retired size spelling** — an image request that carries the output size under the pipe's own retired spelling `image_size` no longer fails with a 400 when the model's published settings cannot be read. `image_size` is kept alive for valve rows written by a filter that has since been deactivated; it was aliased onto `resolution`, which the OpenRouter schema documents as a closed enum of `512`, `1K`, `2K` and `4K`, so a pixel value written that way was refused by the whole generation. The retired spelling now meets the same gate as the modern one, `size`, whether the contract is readable, unreadable or a model that publishes a tier list of its own, and both spellings produce the same key, the same value and the same note. One behaviour does change and is worth stating: `image_size: "banana"` used to go out unchecked on an unreadable contract and is now refused as outside the contract, naming `size`, exactly as it already was where the contract could be read.
- **Open-WebUI-owned tools** — a round only Open WebUI can run is handed back to Open WebUI on every turn, whatever
  `MAX_FUNCTION_CALL_LOOPS` says. The registry entry under such a name is a marker recording who owns the tool, and
  the pipe read its presence as "the pipe can run this"; at the cap that made the pipe answer a call Open WebUI was about
  to run, and for a browser-run tool server, run it itself inside the pipe over Open WebUI's own Socket.IO bridge. The
  three provenances that reach the hand-back are a browser-run tool server, Open WebUI's own builtins — which carry a
  callable the pipe must not use — and a name the request withheld from the model before it ever saw it, under `ask`
  approval in a streamed saved chat or under `function_calling: "legacy"`. A reply that keeps asking for such a tool now
  goes back every time and is bounded by Open WebUI's own `CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS` — which means such a
  reply runs longer and consumes more of the operator's Open WebUI iteration budget than it used to. The skipped-call
  card the pipe emits for a tool Open WebUI owns now names the reason that is true of the request it came from: the
  spent-budget wording only where a hand-back budget really was charged and spent, which is the round that mixes an owned
  call with one the pipe can run, at the cap; and a wording that names no budget at all on a reply that could not be
  handed back, because it is not streamed, so no hand-back could have carried the round back and no turn of one was ever
  spent. A round that mixes an owned call with a call nobody can run still falls to the pipe's own cap, because a call
  with nothing behind it is not Open WebUI's to run either.
