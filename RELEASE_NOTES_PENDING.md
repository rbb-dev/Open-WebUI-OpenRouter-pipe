# Release notes pending

## Behaviour changes

- **Open-WebUI tool mode** — a call Open WebUI runs now shows its card from the moment the model names the tool, and the dashboard's live view now shows the running tool on those turns (it stayed empty before).
- **Pipe dashboard, deleted function row** — deleting the pipe's function row now releases the live dashboard on the worker that served the DELETE and lets
  the pipe finish its in-flight requests and close, instead of holding it, its session-log threads and its storage handle until a restart. The stand-down
  happens whether or not `PIPE_DASHBOARD_ENABLE` is on, and it is per worker: on a multi-worker deployment the other workers release their own generation at
  their next hot reload, exactly as Open WebUI keeps its own per-worker function cache. The action route and the socket gate still refuse once the row is
  gone; nothing about the authorization answer changes.
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
  instead of scaling with the source's short side. The pixel-cap gate that used to refuse an over-cap portrait source
  from its header before decoding is no longer reachable — the bound keeps the geometry inside the cap — so a tall
  over-cap source such as 1920×20000 is now decoded rather than refused from a header read, and the pipe pays that
  decode. It already paid it for every wide over-cap source (10000×3000 is the documented rescue), so what is given
  up is only the tall-and-narrow half of an asymmetry with no principled basis. The gate and the decoded-size check
  behind it stay in the code for a header that lies about the source.

- **Pipe Dashboard, Config tab** — when two administrators save at the same moment on *different* workers, one
  save is now refused with "nothing was saved" instead of being silently lost. Until this, both saves were
  answered "Saved 1 setting": the per-pipe lock only ever serialised saves on one worker, and the revision it is
  checked against is a whole-second timestamp, so two saves inside one second are one revision apart by nothing.
  The refusal appears as the Config tab's own save-failure toast, names the other save as the cause, and leaves
  your edits staged. Installs without Redis (`WEBSOCKET_MANAGER` unset) keep today's behaviour exactly.
- **fusion** — a Fusion panel member's tool file is no longer filed against the outer chat. The file is still
  stored and still rendered in the panel, but it no longer appears in the chat's file list: the tool executor
  re-checks `fusion_inner` before it hands `chat_id`/`message_id` to Open WebUI's upload, so a member's file
  cannot produce a `chat_file` row naming a real chat in a message that does not exist.
- **image contracts** — repointing `BASE_URL` now drops the shared published image contracts immediately, and not
  only when a contract sweep happens to run. The drop sat below the sweep's staleness gate, so a pass with nothing
  stale to do returned before it, and a gateway admin running with the four image filter valves off kept being
  served the previous gateway's aspect-ratio, resolution and seed limits until some later sweep ran. The identity
  is now adopted before the gate, so one stored value (`API_KEY`) has one reader and one answer.
- **API key** — a stored `API_KEY` the key gate refuses now produces the authentication card on the streaming and the
  housekeeping-task legs too, instead of a 401 or an opaque "Unexpected error in streaming loop". Both of those legs
  read the stored field with a bare decrypt instead of through `Pipe._resolve_openrouter_api_key`, so an encrypted
  value that cannot be decrypted went out as an empty `Bearer ` and an encrypted non-`sk-` value the gate refuses went
  out working; a stored value with padding was sent with its padding. Both legs now go through the gate, so one
  misconfiguration has one answer on every leg.
- **model icons** — a model icon is now stored the way it displays. The icon sweep applies the orientation the
  source published before it writes the PNG, so a logo stored sideways is no longer stored sideways; an icon
  already stored keeps its pixels until its source URL changes, which is when it is downloaded again.
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
  *differently* warns again at WARNING. The whole-pass lines (`OpenRouter Image filter ensure failed` and its
  siblings) are unchanged.
- **`ModelFamily.capabilities`** — removed. It was an accessor no production code called, kept only for tests;
  `spec["capabilities"]` is still written, still read by `list_models()`, and still merged into Open WebUI's
  `meta.capabilities`, so nothing about a model's checkboxes in the model editor changes. A third-party
  Open WebUI plugin importing the pipe and calling this one accessor would break; nothing inside this package
  does.
