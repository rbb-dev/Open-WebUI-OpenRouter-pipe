# History Reconstruction & Context Replay

Open WebUI stores chat history as a list of heterogeneous message objects. OpenRouter’s Responses API expects a structured `input` array containing messages and (optionally) structured tool/reasoning artifacts.

This document describes how the pipe builds that `input` array, how it replays persisted artifacts referenced by hidden ULID markers, and how retention/pruning valves affect what is sent upstream.

> **Quick navigation:** [Docs Home](README.md) · [Persistence](persistence_encryption_and_storage.md) · [Multimodal](multimodal_ingestion_pipeline.md) · [Valves](valves_and_configuration_atlas.md)

---

## 1. Entry point: `transform_messages_to_input`

The core history conversion is implemented by `transform_messages_to_input(...)`.

Inputs (high level):
- `messages`: Open WebUI-style messages (each with `role` and `content`).
- Optional context for artifact replay:
  - `chat_id`
  - `openwebui_model_id`
  - `artifact_loader(chat_id, message_id, ulids)` (async) — called once per rebuild rather than
    once per message: the markers of every message sharing a `message_id` are gathered first and
    asked for together. **The replayed path is the id-less shape.** `load_messages_from_db` builds
    each message from `MESSAGE_REPLAY_KEYS`, which includes `id` and `model`
    (`utils/middleware.py:2221`, `models/chat_messages.py:386`), but `process_messages_with_output`
    pops both before the pipe runs (`utils/middleware.py:2296-2298`) — and the `output`-bearing
    assistant branch replaces the dict wholesale through `convert_output_to_messages`, which writes
    neither. Both of its callers (`process_chat_payload` at `:2523`, `drain_approved_tool_calls` at
    `:3574`) run before `chat_completion_handler` (`main.py:1666-1671`), so no saved chat reaches
    `pipe.pipe(body=form_data)` carrying either key. Every message therefore groups under the same
    key and a saved chat of any length costs **one** call, not one per turn.
    `tests/test_b591_the_replayed_path_is_the_id_less_shape.py` drives the real host function and
    pins this; re-check it first on any Open WebUI upgrade.
    The id-carrying shape is the pipe's **own** OpenAI-compatible gateway
    (`api/gateway/chat_completions_adapter.py`), where an external caller supplies messages that
    keep their own ids. There `message_id` scopes the store's own SELECT, so it is never widened:
    a history carrying distinct ids costs one call per id, each asking only for its own group.
    Those calls are issued **together** under a fixed concurrency ceiling
    (`_ARTIFACT_GROUP_CONCURRENCY`), so their cost tracks the slowest group rather than their sum,
    and a longer history adds work rather than overlap.
- Retention/pruning:
  - `pruning_turns` (from `TOOL_OUTPUT_RETENTION_TURNS`)
  - `replayed_reasoning_refs` (for the `PERSIST_REASONING_TOKENS="next_reply"` and `"disabled"` cleanup)
- Runtime context for multimodal conversion:
  - `user_obj`, `event_emitter`
  - `valves` (or defaults to `self.valves`)

Output:
- A list of input items (messages plus any replayed artifacts) suitable for OpenRouter’s Responses API.

---

## 2. System and developer messages

Messages with `role` of `system` or `developer` are preserved as separate message items:

- Content is converted into `input_text` blocks without merging or whitespace normalization. A `system` or `developer` turn whose text resolves to nothing produces no message item at all — which is what Open WebUI’s own converter does (`if system_content:`, `routers/openai.py:1386-1387`) and what the list form of this pipe’s own guard already did.
- The pipe emits them as:

```json
{
  "type": "message",
  "role": "system",
  "content": [{ "type": "input_text", "text": "..." }]
}
```

---

## 3. User messages (content blocks → `input_*`)

User messages are converted into a single `type: "message"` item with a `content` list. The pipe transforms certain known block types; unknown block types pass through unchanged, except that a network or internal-storage URL they carry is treated as untrusted input and the block is dropped, and except that a `data:`-scheme string leaf inside one, at any depth and under any key, is measured against `BASE64_MAX_SIZE_MB` on the same terms a typed block's payload is and the block is refused when one of them is over it. That reading is case-invariant: it is the pipe's own URL parser that decides, so any letter-case of `http`, `https`, `ftp` or `gopher`, a scheme-relative `//host/…` behind leading whitespace, and any spelling of an Open WebUI file path are all the same link. That is the live arm (`requests/transformer.py`). The `/chat/completions` conversion leg (`_responses_input_to_chat_messages`) has its own rule, and it is a different one, and so does the `/responses` outbound walker (`_gate_responses_input_media`), which reads the media values inside `body.input` on the way out for the same reason: neither leg is a place a person can send a link from, but a filter function rewriting `messages` and a model authoring a block into its own reply both can. Its strict arm — the only arm production reaches, since no caller passes `allow_unknown_fields` — drops a block whose type it does not enumerate and names nothing, because a block it cannot read is not evidence of a void attachment. The permissive arm, reached by no production caller, forwards such a block as itself; that fidelity is its contract, and the two arms are not made to agree — with one exception, which is the only place they do agree: a block naming a link only this server can read (an Open WebUI file path, a bare path on this server's disk, or a `file:` URL) is refused and named on both arms, because the backstop sits in the replay guard every block passes rather than in either arm's readers.

### 3.1 Text
Open WebUI may provide user content as a string or as block objects. Text is normalized into:
- `{"type":"input_text","text":"..."}`

### 3.2 Images (vision gating + storage)
Image handling is described in detail in [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md). Key behaviors relevant to history reconstruction:

- Vision gating: if the target model is not vision-capable, the person's attachments and reused images are skipped and the pipe emits a status message saying so. A picture left out of an earlier turn by that gate is reported on the turn that lost it, not only on the turn that skipped it. Pictures a tool returns still go to the model, whatever it accepts, as in Open WebUI's own tool loop (section 5.4).
- Image forwarding policy:
  - `MAX_INPUT_IMAGES_PER_REQUEST` caps the images one turn forwards, its own or reused, and every one of the person's messages in that turn shares that one budget;
    pictures a tool returns are never capped.
  - `IMAGE_INPUT_SELECTION` controls fallback behavior:
    - `user_turn_only`: only user-attached images are forwarded.
    - `user_then_assistant`: if the user turn has no images, the pipe may reuse the most recent image already in the conversation - an assistant image extracted from Markdown image syntax, or one the user attached on an earlier turn - bounded by `IMAGE_REUSE_MAX_TURNS`. An image returned by a tool is never reused this way: it belongs to its tool round (section 5.4), and a tool round ends the window for pictures from before it — once a tool has run, nothing older is reused either, whether or not that round returned a picture. The pool that window reuses from holds only blocks that name a picture source: a block that names none never enters it, so it can neither be reused nor displace a picture that could be. A round that asked you a question is not a media round and does not end the window, and neither does a picture the model shows you in its own reply to a round.
- An image the person attached is never written to Open WebUI storage, in any chat and by any request of a turn. A `data:` URL within `BASE64_MAX_SIZE_MB` is sent as it came apart from the scheme, which is lower-cased to `data:`, and one over the limit is not sent, the picture being skipped and the person told in a status on their latest message; a remote image is downloaded and its bytes sent inline, or its link is forwarded when it cannot be downloaded **unless the pipe's own address check refused the host**: a host that does not resolve, or resolves to a non-routable address, fails closed and is not sent, and the person sees `Images: skipped N (could not be fetched, so it was not sent).` -- that is the completed refusal's sentence, and a check that ran out of its budget instead is refused as a picture left with no time, never as a fetch that failed; an Open WebUI file URL is read with the requester's access and inlined, so providers never need to fetch from your Open WebUI host. That is decided on the **path**: any URL whose path names `/api/v1/files/<id>`, in any scheme, host, letter-case, userinfo, query or encoding, is the same reference, and one that yields no readable id is refused with the person shown the reason rather than sent to the provider as a link. Both an attached remote image and one reused from an earlier turn are inlined as a `data:` URL, under a media type the pipe resolves from the bytes; a payload that settles on a non-image type is refused and reported rather than inlined, and a declaration the bytes do not corroborate is forwarded under that declaration. Images the model generates are stored by the output path.

### 3.3 Files, audio, and video
The pipe includes transformer functions for:
- `input_file` / `file` → `input_file`
- `input_audio` / `audio` → `input_audio`
- `video_url` / `video` → `video_url`

The security/size/SSRF rules (including HTTPS-only defaults) are documented in [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md).

---

## 4. Assistant messages (plain text vs marker-based replay)

Assistant turns are handled in two modes:

### 4.1 Plain assistant messages (no markers)
If the assistant text does not contain any embedded markers, the pipe emits:

```json
{
  "type": "message",
  "role": "assistant",
  "content": [{ "type": "output_text", "text": "..." }]
}
```

### 4.2 Marker-based replay (ULID markers)
If the assistant text contains embedded marker lines, the pipe splits the text into:
- text segments (emitted as assistant `output_text` messages), and
- marker segments (used to replay persisted artifacts).

When a marker split produces several `output_text` items for one assistant message, `annotations` and `reasoning_details` go on the **last** of them, once each, and on no other item. When a turn emits no text item at all — a round that reasoned and called a tool without writing prose — the two lists go on one assistant `message` item whose `output_text` is empty, emitted immediately before that round's tool calls, and only when there is something to carry.

Marker detection and splitting is performed by helper functions (for example `contains_marker(...)` and `split_text_by_markers(...)`); an assistant message's spans are computed once per message per request and read by every consumer of them, and a message carrying no `]: #` is not scanned at all. The markers one group hands the artifact loader are deduped through an insertion-ordered mapping, so the sequence it is handed is first-occurrence order with each marker once, and the number of comparisons the dedupe costs does not grow with the length of the chat. The marker format is:

```text
[<20-char-ulid>]: #
```

Three families of line count as hidden marker lines, all of them valid CommonMark reference definitions and so
invisible in the browser:

```text
[<20-char-ulid>]: #                 the artifact reference — a replay marker
[P:<phase>]: #                       the phase label
[openrouter:v1:<kind>:<body>]: #     the kind-marker transport line — NOT a replay marker
```

Only the first is a **replay marker**: it is what a marker segment resolves, and what the loader is asked for. The
kind family carries the pipe's own transport state (video intent and relay disclosure blocks) and is *not* replayed
and *not* an artifact reference. A kind line is recognised as a hidden line, dropped from the text segments, and
never reaches the provider: not as assistant text, and not as a string value of a content block the conversion leg did
not recognise either — a transport artefact of this pipe is not the caller's to send, whichever arm forwards the block.
A message whose only markers are kind markers takes the marker branch here and produces no `missing_artifact_markers`
warning.

Tool **results** are stripped of hidden marker lines before the model sees them; tool **arguments** deliberately are
not, because a JSON argument value is not a line of the message and a caller may legitimately pass text that looks
like a marker — stripping it would corrupt the payload it means to send.

A caller-supplied `input_file.filename` is stripped too. The name is free text on a block the caller builds, and the
pipe's framing is line-based, so a name carrying a marker on a line of its own would otherwise choose the framing of a
turn the caller does not own — the same line filter as everywhere else, the same carve-out for tool **arguments**
untouched. A name that is nothing but marker lines loses the `filename` field and the file still goes out on its
`file_id`; the person keeps the attachment and the provider is not told the file is called nothing.

The strip covers every content block type and every string field of that block, on both conversion arms, not only the
ones the converter's own loop recognises. A block type the loop names no arm for is judged by the usable check's
catch-all and is then forwarded **as itself** — its other fields intact, because preserving unknown fields is that
arm's contract — but with the pipe's own marker lines removed from its string values first, on whichever arm forwards
it. The block is never dropped to achieve that, and a marker-shaped line inside a sentence is prose and survives.

The strip is a **line** filter, so it removes a line by its whole content rather than by the marker inside it. A
marker-shaped line a user typed **on a line of its own** — in any of the three families above — is removed too, while
the same text inline is preserved. Only whole lines are affected; a marker that is part of a sentence is left alone.

Because the strip works on whole lines, the pipe's own side of this is enforced at the renderer rather than here:
model-authored and user-authored text interpolated into a rendered block is whitespace-flattened onto one line
before it is written, so no user or task-model text can be read back as a marker line. The renderer is the only
place that knows which lines it owns; leaving it to the strip would mean the strip had to guess.

For each marker segment:
- the pipe looks up the referenced persisted artifact payload (via `artifact_loader` when available;
  the lookups are batched across the whole rebuild — on Open WebUI's replayed path, whose messages
  arrive without ids, that is a single call for the whole history rather than one per message; on the
  pipe's own id-carrying gateway it is one call per distinct `message_id` rather than one per message,
  and those calls are issued together under a fixed concurrency ceiling rather than one after another),
- normalizes it to the schema expected by upstream (`normalize_persisted_item`) — once per marker, and the orphan guard below is run over exactly the payloads that normalization **kept**, not over the raw rows. A row the normalizer rejects is therefore not an artifact the guard ever sees: its own output is left orphaned and is dropped and named in the transformer's own `missing calls` warning, while a marker with no row at all stays the separate `missing_artifact_markers` case. A row the normalizer gives a minted `call_id` to is paired against that minted id, so a call and its output stored without ids cannot pair;
- and appends it directly into the `input` array as a structured item.

**Artifact loader preconditions (important):**
- The pipe only attempts to load artifacts when all are present:
  - `artifact_loader`
  - `chat_id`
  - `openwebui_model_id`
  - at least one marker in the message

If any of these are missing, marker segments will not be replayed.

The same four preconditions decide whether the batched lookup runs at all, and they are load-bearing for it: a
fusion member carries no `chat_id` and so never reaches the loader, and the batch is grouped by the same
`message_id` the store scopes its read by. That scoping is unchanged by the groups being read concurrently:
each call still asks only for its own group's markers, and each result is written back under the group id
it was asked for, so the rebuilt `input` is identical to the one a serial pass produces. Artifacts are still classified per message, not over the group, so a
`function_call` in one message whose output lives in another is dropped as the orphan it is rather than paired.

---

## 5. Replay filtering and pruning

### 5.1 Non-replayable tool artifact types
Some tool artifacts are intentionally never replayed back to the provider (to avoid wasting context window and to reduce provider-side errors). The pipe filters these by type during history reconstruction.

The filter is pure set membership over the artifact's `type`, so a family is refused whole or not at all: the shell
family is `shell_call`, `local_shell_call`, `shell_call_output` and `local_shell_call_output`, and all four are in the
set. An output type kept while its call type is stripped would replay half a round — an answer the provider has no
call for — so the two spellings of each half are listed together rather than one of each.

### 5.2 Orphaned function call pairs
When tool calls are persisted, the pipe attempts to keep tool call/request and tool output/response pairs consistent.

During replay, the pipe pairs the normalized artifacts of one assistant message — the same items it replays — and may drop:
- `function_call` items with no matching output
- `function_call_output` items with no matching call

This prevents sending half of a tool interaction back to the model. A call that was still running when the user pressed Stop is exactly that: a call with no result. Dropping it is expected and is logged only at debug level.

The pairing is **per round**, not per call id. The k-th call on an id is paired with the k-th output on that id, in the order the markers are written in the message -- the same notion of a round the replay already uses to name each result -- so a `function_call` with no k-th output is dropped however many other rows share its id, and symmetrically for outputs. One assistant message can hold two rounds on one id (any provider that numbers calls per request produces that shape, and a reply the person stopped with the tool cards off persists `call X, output X, call X` into one message), and a per-id answer would let those two rounds cancel each other's evidence and replay the starved one. This is the rule §5.4 states for the copy selection and the `ask_user` exemption, applied to the same notion of a round.

### 5.3 Tool output pruning by turn age (`TOOL_OUTPUT_RETENTION_TURNS`)
When `TOOL_OUTPUT_RETENTION_TURNS` is set, the pipe computes turn indices across the conversation and treats messages older than the retention window as “old”.

For old turns, it can prune very large `function_call_output.output` strings by:
- preserving a head and tail,
- inserting a note indicating the output was pruned,
- and leaving markers intact.

When an output carries a picture, only its text is shortened; the picture stays.

This keeps replay payloads smaller while preserving recency and high-level context.

### 5.4 The pipe's own copy of each tool round

What is remembered between turns, for a round replayed from that copy: nothing about the round's rows themselves. A stored picture's well-formedness verdict is remembered by the digest of its payload -- a 16-byte digest and a yes or no, never the picture -- so a replayed picture is not re-validated on every turn, and a temporary chat, which keeps nothing, is not remembered at all.

Every tool round the pipe runs itself, and every round of an OpenRouter server tool, is also stored by the pipe: the
call and its output as a pair, behind a hidden marker placed in the answer where the round happened. Four exceptions:
with results kept, a server tool whose own item OpenRouter takes back unchanged -- the advisor, the subagent, model
search -- is stored as that item instead, while every other server tool, including one the pipe does not know, keeps
the pair; image generation, whose picture is already part of the answer, is not stored; a
and a request that belongs to no chat message -- a call that carries no `chat_id`, the plain API route --
keeps its copy in memory for the length of that request only, keyed on the request id, and never writes it to the
database; the request ends and it is gone, and no marker line is added to the caller's response, so such a call's
records last for the request, not the conversation. A temporary chat
keeps nothing, so with tool cards off its rounds reach no later request. (In Open-WebUI tool mode the rounds of a
temporary chat's reply are held in memory until that reply ends or the provider refuses a call-back the pipe was waiting for, so Open WebUI's calls back after each round of tool
calls still hand them to the model. A reply dropped under the pool's byte ceiling is reported as holding none of the rows it was
offered in that flush, so no marker is written for a round that is no longer in memory and that round is not replayed; see [Persistence](persistence_encryption_and_storage.md).) The calls are written when
the round starts and each result when its call returns, and the round comes back to the model in call order on that turn. So in a streamed
reply, Stop keeps a round's calls before the first one still running. A call refused before it ran is answered as soon as it
is refused, and on the turn it is answered it is handed to the model in the order the round asked for it, because its
answer is produced before any queued call has run. A round stored with tool cards on is written as the calls are answered, so a
Continue replays a refused call's answer ahead of the results that were still running. A call the model sends malformed is answered as Open WebUI's own tool loop answers it on the same route and is kept like any other call; several argument objects sent back to back become one call each.

A **provider-native shell round** is kept in the same place and in the same shape, and is not a special case of the
pipe's own loop: the model emitted it, the pipe never ran it. Its `shell_call` or `local_shell_call` item, and the
`shell_call_output` or `local_shell_call_output` item that answers it, are stored as the `function_call` /
`function_call_output` pair every other round is stored as — named `shell`, carrying the commands the item asked to
run as its arguments and the item's own stdout as its result. The pair is keyed on the item's `call_id`, so a call and
its output are one round on one id rather than two, and the round is committed once however many of the family's items
the turn carried. What the model is handed on a later turn follows the ordinary rule: with `PERSIST_TOOL_RESULTS` off
the arguments and the result are withheld and the model is told the tool name and whether it succeeded; with it on
both are handed over whole, since a round the model ran is a fact about the turn whether or not the commands are
retained. The raw native item is never what is replayed, and never what is stored: the four types are in the
non-replayable set (§5.1), so a `shell_call` the caller echoes back into a request is stripped before it reaches the
provider, and the pair behind its marker is what stands in its place.

A call can also be kept in this copy and still be withheld from one request: when the replay budget is applied to that
turn, the result is replaced in `input` with a model-visible stub, and the tools whose results were replaced are named to
the person in a warning notification — once per call id, on every path that applies the budget except the task-model
adapter, which never sees the tool round. The copy above is untouched, so a later turn that fits still replays the full
result.

The copy carries the call's arguments and its full result, pictures included, whatever `PERSIST_TOOL_RESULTS`
says, as a shown card does in the message Open WebUI saves; that setting decides what later turns receive, not
whether the round is stored (see below). A call that
did not complete keeps its failure text; where that text alone would not read as a failure, it opens with
`Error: the tool call did not complete.` With no `ARTIFACT_ENCRYPTION_KEY` set, or with `ENCRYPT_ALL` off, the copy
is stored as plain JSON.

On replay each round reaches the model exactly once, except in a temporary chat, where a later turn gets a round
only through the card Open WebUI keeps for it in the browser, and none with cards off:

- Open WebUI hands back the rounds saved in the message -- the shown cards of a reply -- as ordinary tool messages.
  Where it has handed back a call, the pipe's copy of that call is dropped. Where a call id appears more than once,
  the drop is per round, not per id: each round the pipe has a result for keeps it.
- A copy the pipe wrote for a round Open WebUI was never given -- with tool cards off -- is marked as such and always
  kept. Call ids alone cannot decide this: chats saved before the pipe made the ids it invents unique hold repeated
  ones, because the chat-completions route used to number calls without an id per request. This mark is what keeps the
  rule exact.
- After Stop in a streamed reply, Open WebUI saves as finished the cards of the calls before the first one still
  running (the pipe fills a round's cards in call order) and marks that call and every later call unfinished; its
  converter hands back only the finished calls, and the pipe's copy of them is dropped as a duplicate. With cards
  off, or where no card was saved, the pipe's copy carries the round's calls before the first one still running.
- Where the pipe replays OpenRouter's own item for a server tool unchanged (the advisor, the subagent or model
  search, with results kept), that item
  wins over the card pair Open WebUI saved for it.
- An earlier turn's results are withheld by the same rule whichever copy carries them, and so is a round that arrived before the chat's
  first turn -- which is what an API caller, an imported or reordered chat, or a filter posts -- since such a
  round is never the current turn. With `PERSIST_TOOL_RESULTS` off
  the model gets `{}` in place of the arguments and a placeholder result -- `[tool result not retained]`, or
  `[tool call failed; result not retained]` when the call did not complete. The withholding holds on the list shape,
  and on the flattened shape whenever the hand-off sits next to its result, which is how Open WebUI 0.11.4 builds it
  (`flush_tool_outputs()` and `flush_tool_images()` run back to back). On a flattened history whose hand-off is
  separated from its result -- a shape only an API caller, an imported or hand-edited chat, or a filter can build --
  the round's picture is forwarded: the pipe cannot tell that history apart from a person's own message carrying
  Open WebUI's fixed hand-off sentence and their own photo, and treating it as the round's hand-off would withhold
  the person's picture too. What the caller built is the caller's own posted bytes, so nothing the pipe stores is
  exposed by it (TODO T1163, to reopen only if Open WebUI ever builds the gapped shape). A round of Open WebUI's built-in
  `ask_user` is the exception: its question and the person's typed answer are always handed over, since the answer
  is the person's own words. The round is recognised by the tool behind it, not by the name it was advertised under,
  so a third-party tool that happens to be called `ask_user` is a tool like any other and is withheld -- and that
  is now true on the **write** side as well as the read side, not only where a stored row is judged. The
  identity is carried two ways, and both are read before the request's own tool set: the pipe stamps **every**
  ask_user-shaped round it records, on the run path and on the hand-back path alike, with `True` for Open WebUI's
  builtin and `False` for any other tool carrying that name, so a stamp is never absent and a later turn reads a
  record rather than re-deciding it,
  and the round is read back by **its own record rather than by its name**: the lookup is keyed on the round's
  position -- the message it sits in, its `call_id`, and its ordinal among the rounds sharing that id -- so a
  renamed round is found by whichever spelling of its name it happens to carry, whether the advertised
  `ask_user__<digest>` or the origin name it was stored under. The bare name
  `ask_user` is reserved for the built-in, so a user's own tool of that name is advertised (and,
  with tool cards on, displayed) as `ask_user__<digest>` and a round recorded before this change stays
  grandfathered until it ages out of `TOOL_OUTPUT_RETENTION_TURNS`. A round the pipe answers back to Open WebUI
  instead of running -- which is what `Open-WebUI` mode does with the built-in, since the pipe never executes it --
  is recorded the same way, as a stored call carrying no result, because Open WebUI writes that round into the chat
  under the name the pipe handed back and that name is the bare one. A record is readable only through its marker
  line on the assistant message and a store that answers, so a round whose rows are gone is judged by the request's
  tool set as before. The identity is read from the registry **as this request received it**, even on a turn that
  withheld the built-in from the model before the model ever saw it -- ask approval, and `function_calling: "legacy"`,
  both empty the registry on their way to the model, and neither of them changes what a name in it is. The set
  those two record is also what the hand-back predicate consults, which is why the fix can read it: a round of
  names Open WebUI alone can run is a round the pipe cannot be the judge of, so a streamed reply hands it back
  rather than answering it, and only a reply that could have been handed back and spent its budget is told the
  budget is what stopped it.
  The exemption is also **per round**, not per call id: a model may reuse one `call_id` across two rounds, and on the
  replay path the exempt round is the one whose own stored call is the built-in, while the other round on that same
  id is withheld. Because each output is paired with its own call, a tool round that shares an id with a built-in
  round still counts as a tool round for the picture-reuse window, and so still closes it. It applies inside an
  internal Fusion step too: the built-in is not *offered* there, so a panel member cannot run it and is handed
  no `ask_user` tool, but a round the outer turn already ran it for keeps its exemption, and a panel member is
  told what the person was asked and what they answered. Only a round the markers name reaches a panel member,
  and only that round: no other stored artifact of the outer chat is replayed into a panel member's prompt.
  A round whose stored rows are gone is judged by the request's tool set as before; that
  fallback is the whole purpose of a missing row and this change does not touch it. The stored rows are left
  alone, so turning the setting back on hands the full results over again.
- An image returned by a tool comes back as a separate message right after the round's results ("Here are the
  images from the tool results above"): Open WebUI builds it from its own record, and the pipe builds the same
  message when its own copy carries the round, so the request is the same whatever the card switch says. It is
  adjacent only where the pipe builds the message itself; on replay, a `system`/`developer` message that Open WebUI or a
  filter placed between a round's results and a handoff Open WebUI itself stored is emitted where it sits and the stored
  handoff follows it, because the pipe recognises that handoff across those roles and therefore adds no copy of its own.
  It is part of that round's result: handed over in full where it sits, whatever the attachment limit, even when it is
  not the last message; withheld with the round on an earlier turn while results are not kept; never stored again;
  unlike an image the person attached, never reused on a later question; and it ends the reuse of any older
  picture. Each of its pictures is gated before it is sent, exactly as a picture the person attached is - scheme, and with `image_url` spelled as an object read rather than stringified -
  plain `http://`, inline size, and the address an `http(s)` link names - and a refused one is named on the turn
  that round is answered on. The message itself is sent only when at least one picture survived: a round of
  nothing but refused pictures goes to the model as its text alone, because a message carrying the sentence with no
  picture would no longer read as the round's result, and that reading is what lets a person mid-tool-round past the
  concurrency breaker. That message is recognised as the round's only when its pictures are the round's own, so a person who
  repeats the sentence with a picture of their own keeps their turn.

The copy does not depend on reasoning. Tool rounds were accepted without the reasoning around them when measured with
Claude Opus 4.8 on `/responses` and `/chat/completions`, so the copy stays when reasoning is dropped from a request and outlives reasoning
cleanup. It is never published as an output item, so Open WebUI neither draws it nor runs it again.

---

---

## 6. Reasoning replay and `PERSIST_REASONING_TOKENS`

When replayed artifacts include reasoning items, the pipe can optionally record references in `replayed_reasoning_refs` so the caller can delete those artifacts after replay when reasoning retention is limited to a single turn. Under `next_reply` and `disabled` alike, the cleanup runs only on a generation that finished the reply: it must not have been cancelled or errored, and it must not have handed its tool calls back for another request to answer. The two ways a reply is not finished are both guarded by the same clause. A Continue keeps the rows of the message that request is still writing, so continuing an answer does not delete the reasoning of the generation it continues; a hand-back keeps them, because the request that comes back to answer the tool results is the one that will need them, and when that request arrives it is the one that deletes them; if it never arrives -- the user stops, or the sender errors -- the rows wait for the housekeeping sweep, because no generation finished the reply and `next_reply` has nothing to bind to. A delete that fails does not fail the reply: the turn has already been answered, so the store reports the failure, keeps the rows and the references, and the next turn retries the delete. That retry is the next request rather than the sweep -- the sweep ages rows out by `created_at`, which every read refreshes, so a chat that keeps being read never ages out. It is also said once, on the turn that could not delete them, so the rows left behind are visible rather than silent. The tool-round copies of §5.4 are not reasoning and are not deleted with it.

`disabled` gates the replay as well as the write. A chat that still holds a reasoning marker from a turn run under another setting reaches the provider with that reasoning withheld: on `/responses` no `reasoning` input item is built from the row, and on `/chat/completions` the message's own `reasoning_details` are not re-attached either. The row is still recorded in `replayed_reasoning_refs`, so the delete above removes it on exactly the terms `next_reply` uses; a row that is never deleted that way waits for the housekeeping sweep, whose `created_at` clock was refreshed by the read that resolved it, so a withheld row keeps ageing from the last turn it was resolved in and is reaped within `ARTIFACT_CLEANUP_DAYS` as usual. The gate is a `continue` over one marker, not a `break` over the message, so the round's own `function_call` / `function_call_output` markers still replay under `PERSIST_TOOL_RESULTS` - an Anthropic tool round therefore goes out with its tool call and result and no thinking block.

System default is `PERSIST_REASONING_TOKENS="conversation"`; see [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the exact semantics and defaults.

A reasoning row keeps the `signature`, `format` and `encrypted_content` the provider sent, on every `/chat/completions` route, streamed or not: an unsigned row is one the next turn cannot replay, and on a later Anthropic turn that costs the whole span rather than the one block. A row the provider reports only in the completed response is taken from there, which is the one shape on which a signature can exist at all — a provider that streams its thought unsigned and signs only the closed block never emits a `response.output_item.done` for it to be read from.

### 6.1 Where a replayed reasoning item goes

A model produces reasoning at a particular moment: before a tool call, after its result, or between two
sentences of an answer. Open WebUI stores the answer as text, so that position is lost unless the pipe records
it. Each persisted reasoning item therefore carries an anchor - which call it preceded or followed, or which
assistant message it sat before - and replay puts it back in that place rather than appending it. An ordinal at
or beyond the turn's message count places the block after the last message, in that turn: a continued turn whose
carried turn ends in reasoning produces exactly that one-past ordinal. That ordinal counts **every** assistant
message in the turn, on the writing side exactly as on the reading side: a message the model sent without a
`phase` marker counts like any other, and a `phase`-emitting model and a non-`phase` model produce identical
ordinals for the same event sequence. A turn that wrote no prose is one such message: the `/responses` reading
leg emits the same empty-text assistant `message` item the write side emitted for it, in the same order, so
the two sides still count one message for it. An API call (no `chat_id`) is the one exception, because none of its
messages are written to the database and are replayed, so its counter does not advance.

Anchors are **scoped to a turn**, where a turn is the region between user messages. Tool `call_id` values are
not guaranteed unique across a conversation: in chats saved before the pipe made the ids it invents unique, the
same id can appear in several turns, because the chat-completions adapters used to number calls per request.
Binding an anchor only within its own turn is what keeps a reasoning item from
attaching itself to an unrelated call with the same id. Open WebUI's own "Here are the images from the tool
results above" message, which follows a round's results and carries that round's pictures, is not a user message
for this purpose: it stays inside its round's turn on both the generating and the replaying side, and a thought
the model had after the turn is replayed after it.

On `/chat/completions` a replayed reasoning item rides on the assistant message that carries the tool calls, as
`reasoning_details`, matching Open WebUI's `convert_output_to_messages(raw=True)`. A message that names a
different `model` has its whole `reasoning_details` block dropped before the request is built, the way Open WebUI
does at `utils/middleware.py:2516-2521`; a message that names no model, or a blank one, keeps it, because the
pipe's own gateway puts `model` on the chunk and never on the message and a missing id cannot be attributed.
**On Open WebUI 0.11.4's replayed path that host guard cannot fire**: `process_messages_with_output` pops
`model` from every message before the pipe runs (`utils/middleware.py:2296-2298`), so the message-level
`same_model` predicate always takes its absent-means-unknown arm there. It is load-bearing on the pipe's own
OpenAI-compatible gateway, where the caller supplies its own messages; on a saved chat the **stored row's** guard
(the next paragraph) is the one that carries the rule. That replayed block is
byte-identical to the concatenation of the streamed `reasoning.text` deltas plus every `reasoning.summary` fragment the provider sent for each of its keys: never that run plus the terminal
`message.reasoning_details` echo a provider may close the stream with, which carries the same bytes a second time.

The `Chats` write at the end of a turn reads `annotations` and `reasoning_details` off **every** round of the
turn, in round order, and not off the round the loop happened to end on. A turn whose middle rounds reasoned and
whose last round did not would otherwise store neither field at all, and a turn that ran three reasoning rounds
would store only the third. The per-round scan sits on the path that completes a round, so an attempt handed back
for a retry never fills it and still writes nothing over the winning attempt's message.

The anchors are internal ordering metadata, and they are stripped from every item the pipe sends — including a
turn-opener, not only reasoning — so none of the keys in `REASONING_ANCHOR_KEYS` is ever part of a
request.

A stored `reasoning` row is also withheld **whole** when the store records that another model produced it:
never per detail block, never with its summary kept and only its signature removed. A partly filtered reasoning
sequence is exactly what a provider rejects, so the row either replays intact or not at all. The producer is
the row's own `model_id` column, carried out of the store's read beside the payload and never inside it, and the
predicate is the message-level one from §6.1: a non-blank producer that differs from the model answering this
request withholds; an absent or blank one replays, because cannot-attribute-then-keep is the safe direction (a
dropped signature costs one reasoning block, a wrong one costs the turn). A row withheld this way is **not**
added to `replayed_reasoning_refs`, because that list is the delete queue and the row is not dead — the person
switches back and it is good. Rows written before the column was read back have no producer and keep replaying.

This is the pipe compensating for a host behaviour rather than mirroring it, and deliberately so. Open WebUI
0.11.4's own `strip_reasoning_details` guard (§6.1) is unreachable on the replayed path, and the store is the
only component that still knows which model signed a block. The value never reaches the wire.

### 6.2 An answer continued across more than one request

"Continue response" adds a second generation to the same assistant message, and Open WebUI's own tool loop can
call the pipe several times within one turn. Ordinals are counted per request, so without care the continuation
would number its first call `0` again and its reasoning would bind to the first generation's call - placing two
thinking blocks side by side, the shape providers reject.

The pipe therefore offsets a continuation's ordinals by what the turn already contains: the calls and the
assistant messages the request carries before this generation starts. A turn that ends on reasoning gets one
further step, so the continuation's first block is placed after its own text rather than beside the block that
ended the previous generation. The offsets apply whether or not the Continue streams, since Open WebUI keeps the
earlier generation on a non-streamed Continue as it does on a streamed one. The two `/chat/completions` transports agree on that anchor whichever one carried the
generation: the non-streamed one flushes each round's reasoning before its tool calls, exactly as the
streamed one does, so a thought the model produced before a call is replayed before it on both.

Within one request the same question arrives by a different route, because a tool round re-sends the previous
round's reasoning item verbatim — `id` included — the pipe having normalised only `function_call` and
`function_call_output`. So a second round whose thinking arrives under an upstream id already used is the
ordinary shape of a tool turn with a reasoning model, not a provider quirk, and the reuse is the pipe's own making.
Merging those two blocks on the id would be wrong for a reason that has nothing to do with ordering: every reducer
downstream replaces by id, so the second would overwrite the first in the record instead of standing beside it. The
pipe therefore treats the round, not the key, as the unit: a fragment that recurs under a key in a later round is
published as a block of its own under a fresh id — the first block keeping the provider's, because that is the id
Open WebUI merges the stored record by — while two fragments under one key inside one round stay the one block that
they are, which is also the shape providers reject. What makes this a display question rather than a replay one is
that the pipe sends the reasoning to the model unchanged and mints a display id only for the block it publishes;
the offsets above govern what is sent, and this governs only what the person is shown.

The message's `sources`, `annotations` and `reasoning_details` accumulate the same way, and the pipe's
end-of-turn write is the reason they do. Open WebUI maintains `sources` by its own read-modify-write — its socket
handler appends every `source` frame it receives (`socket/main.py:1250-1265`) — and the pipe's write was the only
*replacing* writer, built from one request's accumulated citations, so it removed whatever was on the message that
this request did not itself produce. The stored value of each field is now the union: the entries already on the
message followed by this turn's entries whose identity is not among them, with the stored copy of a repeat kept so
a chip the person already saw does not change underneath them. The union is unconditional. Gating it on a Continue
would be an inference from a client-supplied key (`assistant_message_id`), and the clobber it would leave is the
ordinary one: a builtin tool's citation is emitted as a `source` frame without being accumulated by the pipe, so on
a plain RAG turn the message already holds a source this request knows nothing about. `annotations` and
`reasoning_details` are read from the chat blob for the same reason — `chat_message` has no column for either, so a
per-message read would merge the citations and silently drop the other two. See
[Streaming pipeline and emitters](streaming_pipeline_and_emitters.md#41-the-stored-output-array-and-who-owns-it)
for how many times that write happens.

The continuation's first output also has to bring its own line break. Open WebUI joins that output onto the
stored reply's last line. A hidden marker line that gains text after it is no longer a marker: its row stops
replaying, and the line either shows its id as text or, when the added text contains no space, hides the added text
along with the id. So when the stored reply ends on a marker line, the continuation's first model output starts on a
new line, whether it is text, a marker line or a generated picture. When the stored reply ends on ordinary text
instead, model text and a generated picture carry on the sentence, while a hidden marker line that comes first
starts a new paragraph, so it stays a line of its own. A card
or a refusal that opens a continuation - the provider's error card when a model refuses prefill, the breaker's
refusal, a busy or startup notice, the card saying the API key is missing, or the notice that the model catalog
could not load when the Continue is not streamed - starts a block of its own on any Continue; on a streamed Continue
the catalog notice shows as Open WebUI's error instead, and the stored reply stays as it was. A reply written by
internal Fusion is not continued: a Continue on one leaves the reply as it was, and a notification says that Continue
is not available for Fusion replies and that regenerating runs Fusion again.

A continuation's closing frame carries no replacing `content`. The stored array is right and stays right; it is the
live message that would be shortened, because the browser assigns `message.content` absolutely from a frame that
carries text and no `output`. So the turn still ends with its `chat:completion` frame, carrying `done` and `usage`,
and carries no `content` for the frontend to write over the prefix with.

---

## 7. Failure modes (what happens when artifacts are missing)

- If the artifact loader fails (DB errors, network issues), the pipe logs a warning and continues without replaying artifacts for that assistant message — or, since the lookups are batched, for the whole group that one lookup covered, every message sharing its `message_id`, which on the replayed path is the whole history — and the person is told: the store emits a warning notification saying that earlier tool results could not be loaded, so the model did not receive them. The notice names no cause, because the store cannot tell a database blip from a decryption failure, and it promises no retry. A read that *succeeds* but returns fewer rows than it asked for is a different state and is announced differently: it says how many rows were unreadable and which artifact kinds they were, and it names no id and no cause. That is deliberately not folded into the notice above, because the two are told apart by the decrypt set rather than by a shortfall — a row that was legitimately deleted is missing from the result too, and reporting it would announce every retention sweep as a lost round. It is a notification rather than a status because the read-side arms report something the person can act on in the moment, and a progress status is a transient-degradation surface: `StatusHistory.svelte` shows `history.at(-1)` as its collapsed header and `expand = false`, so the newest status is what a reader sees without clicking. The write-loss site differs and is deliberately not folded in: it keeps the toast *and* adds the status, because the fact it reports is not over — the rows are gone and the next turn will be short of them. That is safe where the rejected reading was not, because Open WebUI's socket emitter **appends** each `status` to the message's `statusHistory` rather than replacing the entry, and the component renders the whole history behind the toggle, so the notice is one click down on a surface that survives a reload rather than lost. The cache refill that follows a successful read is a separate case and is not this one: a fault there returns the rows the database gave, clears the breaker window, and is logged as a cache fault, because nothing was lost and announcing a lost round would be false. A third arm reaches the same state without the failure: if the store never initialised at all there is no database to ask, and it announces the same missing round in the same cause-free words and logs that the store is not configured, without charging the per-user database breaker — that one is a state rather than a fault, so charging it would make the next turn report a repeated-error skip that never happened. Each of these four arms keeps its own wording, and each fires once per degraded episode rather than once per turn, re-arming on the first read that reaches the success arm: the loader is called once per group, a long turn can carry many of them, and Open WebUI fires a toast per notification frame without deduping, so an unlatched notice repeats the same banner on every turn of a broken chat. The fault and breaker arms share one latch because both say the same thing; the not-configured arm keeps its own. The unreadable-rows arm is not latched: its text carries the row count and the artifact kinds, so a second turn's copy would say less than the first.
- If an individual marker cannot be resolved to a payload (for example after key rotation or cleanup), the pipe logs a warning and skips that artifact, and says nothing to the person. That stays log-only on purpose: at this layer a row consumed by `PERSIST_REASONING_TOKENS=next_reply`, a row lost to a rotated key and a row lost to a failed read are one state, and the first is a normal steady state on every turn of every reasoning conversation whose retention is `next_reply`. In a temporary chat a later turn's markers resolve to nothing by design, since the pipe keeps none of its rows, and are logged only at debug level. Neither of these records names the chat when the chat is a temporary one -- a temporary chat's id is the browser's socket id, so the record reads `chat_id=<not retained>`, the one spelling the pipe uses wherever it has to name a temporary chat in place of its id, and a saved chat's names its own, as at every other site in the package. A call that carries no `chat_id` never gets markers at all, so there is nothing for it to resolve.

Operational implications:
- Conversations may still render in the UI, but upstream requests may lack some historical tool/reasoning context. When the store itself failed, the gap is announced on the request that lost the round rather than left to be discovered later.
- If you rely on long-lived replayability, validate your retention and key rotation procedures in [Persistence, Encryption & Storage](persistence_encryption_and_storage.md).
