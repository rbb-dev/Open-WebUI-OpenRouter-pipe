# Pipe Dashboard Plugin — Operations Guide

An admin dashboard and configuration editor exposed as a virtual model in Open WebUI.

1. [Overview](#overview)
2. [Access](#access)
3. [Opening it](#opening-it)
4. [The dashboard](#the-dashboard)
5. [Editing configuration](#editing-configuration)
6. [Live configuration updates](#live-configuration-updates)
7. [Requirements](#requirements)
8. [See Also](#see-also)

---

## Overview

Selecting the **Pipe Dashboard** model in Open WebUI turns the chat box into an admin console. The console offers two things at a glance: a live operations dashboard — requests, resources, and storage tracked in real time — and an editable configuration panel for the pipe's admin valves.

The dashboard is part of the pipe's plugin system, which ships off. Two valves switch it on, in order:

1. `ENABLE_PLUGIN_SYSTEM` — the master switch for the plugin system (default: off). Set it on before any plugin loads. The action route reads it from the persisted valve row on every request, so committing it off closes the route on every worker without a restart, and a row the store will not hand back refuses rather than serving the in-memory copy.
2. `PIPE_DASHBOARD_ENABLE` — adds the Pipe Dashboard model to the selector. Read from the persisted valve row on every request, for the same reason.

Three admin valves control the feature. They appear in Open WebUI's Settings once the plugin system is enabled.

| Valve | Type | Default | What it does |
|-------|------|---------|--------------|
| `PIPE_DASHBOARD_ENABLE` | bool | `False` | Shows or hides the Pipe Dashboard model in the model selector, and closes the console behind it: with it off the action route answers 404, new dashboard subscriptions are refused, and viewers already watching are dropped so the live feed stops. The dashboard's own row in Open WebUI's Models table is switched off and on with this valve as well, and claims the off in its metadata, so Settings -> Models and Settings -> Interface never list it as enabled while every gate behind it answers 404. Read from the stored row at the model list, the route, the subscription and the emit, so a toggle takes effect on the very next request on every worker, with no restart -- including on workers that never served a chat; a row the store will not hand back refuses rather than serving the worker's in-memory copy. A row that cannot be read is treated as off, and so is a row that omits the key: the Config-tab writer drops any field equal to its declared default, and that default is `False`. Viewers already connected are evicted as part of the same switch, from the dashboard's own Config tab as well as from Open WebUI's function editor. |
| `PIPE_DASHBOARD_USAGE_COLLECT` | bool | `False` | Records one usage entry per completed request (user, model, tokens, tools, cost) to power the Usage tab. Read from the stored settings at write time: turning it on starts recording without a restart, and so does turning it off -- on every worker, including one that has served no request, and for the background sweep's rows as well as the request path's. A stored configuration that cannot be read at all counts as off, as does one that does not carry the key. |
| `PIPE_DASHBOARD_USAGE_RETENTION_DAYS` | int | `30` | How long usage records are kept. A background purge removes older records. Read live, from the stored valve row rather than the worker's in-memory copy, so the purge and the Usage tab always report the same window, falling back to this default when they cannot be read at all. |

---

## Access

Access is governed by Open WebUI's model access control on this overlay model — there is no separate admin flag, and the switch that follows `PIPE_DASHBOARD_ENABLE` is about *enabled state*, not about who may reach it. Assign users and groups in the model's **Access** editor:

- A **read** grant makes a **viewer**: open the dashboard and watch the live feed.
- A **write** grant makes an **operator**: everything a viewer can do, plus use the action buttons. Editing configuration is separate — see [Access](#access) below.

Owners and admins hold both grants. A user with no grant receives an access-denied message.

---

## Opening it

Select **Pipe Dashboard** in the model dropdown, then type a command and send it.

| Command | What it does |
|---------|--------------|
| `dashboard` | Opens the live console. |
| `help` | Lists the available commands. |

An empty message opens `help`. An unrecognized word returns a short notice pointing back to `help`.

---

## The dashboard

The `dashboard` command opens a console organized into tabs. Each tab covers one area of the pipe's operation.

### Live

![Dashboard walkthrough](images/dashboard_video.gif)

The Live tab shows the in-flight and recently-completed requests across every worker, as a capped view: the **30 newest** in-flight rows per worker, then up to 300 completed rows per worker, and up to 300 rows across the cluster once the workers are merged (finished rows are the ones dropped first). The **Active** tile is *not* capped — it counts every tracked, not-yet-finished request across all workers — so above the cap the table header says how many of them the rows below represent. `Done`, `Cost` and `Tokens` are totals over the rows shown, so above the cap they cover the newest requests rather than all of them. Each row carries:

- User and model. The model column shows display names and toggles to model slugs.
- A status badge: `queued`, `streaming`, `tool:<name>`, `completed`, `failed`, or `cancelled`.
- Elapsed time, tool success and failure counts, tokens (in → cached → out), cost, and the worker PID.

Cost updates live as the request runs; the completed row shows the final cost. Task-model calls — titles, tags, follow-ups — fold their cost into their parent chat's row: the turn that was already running when the task started.

Completed rows stay visible, dimmed, for the **Keep completed** window (5 minutes to 3 hours, default 10 minutes), set in the table itself.

The table sorts on any column and has a filter box.

### Usage

The Usage tab needs `PIPE_DASHBOARD_USAGE_COLLECT` on. Without it, the tab shows a hint to enable collection.

With collection on, the tab presents:

- **Metric cards** — Sessions, Tokens, Cost, Tools, Errors, and Cached input, each with a change chip against the previous period of equal length and a per-bucket sparkline. On Errors and Cached input the sparkline plots the card's own rate — the bucket's error rate and the bucket's cached-input percentage — so the line and the headline are the same quantity; a bucket with no sessions in it plots 0%.
- **Resource cards** — live CPU, Memory, and Disk.
- **Usage trend** — a chart with two lines per bucket: tokens (left axis) and cost (right axis), with a hover tooltip and a timezone-aware range caption.
- **By model** — cost share per model. Each task-model appears as its own `model (tasks)` row with its own cost.
- **By user** — sorted by cost, showing the top 10 with an "N others" roll-up. Search and column-sort reach every user, including those inside the roll-up. A pinned **Totals** row at the bottom sums the visible rows (sessions, tokens, tools, cost).

Select a range: 1h, 6h, 24h, 7d, or 30d. Ranges longer than the retention window are disabled. A footnote shows the collection-start date, the retention window, and the record count.

**Invoice note.** Task models configured outside this pipe never reach it, so they are absent from these totals. Expect a small gap against the OpenRouter invoice when such task models are in use.

A request that is still running and producing liveness signals when the two-hour sweep passes is recorded by whoever really ends it, so its status, duration and token counts are the real ones. Streaming requests refresh the liveness stamp on every chunk. A non-streaming request that runs a tool call, or whose provider call retries, refreshes it too and is likewise spared: the sweep abandons a non-streaming request only when it has produced no liveness signal at all in two hours. A non-streaming request with no tool activity and no retry is still abandoned as a failure after two hours — that case is not covered here. The sweep runs on a timer of its own, so this happens whether or not the dashboard is open.

The Usage tables sort on any column and have a filter box.

### Health

The Health tab tracks the pipe's live load:

- **Concurrency** — active requests and tools, in-flight calls, and active video generations when the video pool is configured.
- **Queues** — the pending-request count, and the log and archive queue depths, each with its bound. The row's headline number is `waiters + queued`, where `waiters` counts coroutines blocked on a semaphore permit and `queued` counts requests still in the request queue, so the backlog itself is the `requests` / `requests_max` field rather than the headline: because a single dispatcher performs the waiting, the request waiters term reads 1 for any real backlog and stays 1 however deep the queue is. The tool pool parks one waiter per tool call instead, and its depth is in the Concurrency row's tool cell tooltip.
- **User Circuit Breakers** — request, tool, and auth breaker counts. "Seen" counts the request-breaker records whose newest failure is inside the breaker window (the Tools row counts the same for user+tool pairs), and a record is created by a failure. The RECORD is a separate thing from the count: the Requests row's record is released when that user's next request succeeds, when a later check finds its window empty, or on the next breaker request write once its newest failure is older than the breaker window, so a tripped user who stops sending requests keeps a record that is no longer counted until then. A Tools record is released the same way: when that user next invokes that tool and the call succeeds, when a later check finds the window empty, or on the next breaker tool write once its newest failure is older than the window — a tripped pair for a tool never invoked again lingers the same way. "Users w/ fail" counts records with recent failures, which after that change is the same set, so on each row the two numbers are equal.
- **Models** — the model catalog with a per-type breakdown (text, image, video), the ZDR-capable count, and per-type fetch clocks. A fetch clock moves whenever that catalogue was read, so an extra "Last attempt" row appears for a type whose last read found nothing; a metadata sync, by contrast, runs on a catalogue or settings change and stays quiet on a read that changed nothing. The status badge tracks the chat-catalog fetch loop and shows the consecutive-failure count of the OpenRouter accounts **this deployment is still using** — the worst of them — so it cannot read healthy while an account in use is failing, and an account you have rotated away from stops counting. A failure is remembered per credential for the most recent 4 credentials, so that window is the most the badge can ever fall back on when it cannot resolve the key in use.

### System

The System tab covers readiness and infrastructure:

- **Readiness** — initialization state, HTTP session, the session-logging worker, log-buffer RAM usage, and pipe-level Redis with a liveness ping. The session-logging worker reads **Idle** until the first record is persisted, which is its normal starting state. The HTTP session is the address-vetting transport the model-icon, maker-profile and self-update fetches share; it reads **Idle** until the first of those runs, **Active** once its pool is open, and **Closed** after shutdown.
- **Artifact DB** — the database write-pool backlog and the database circuit-breaker states. The DB breakers row's count beside the tripped badge is the number of users holding a database-failure record inside the failure-counting window, so a user whose failures have all aged out of it is not counted; such a record is also **released**, on the next breaker write and at most once per window, so the number is bounded by current activity rather than by everything the worker has ever seen — the same release rule the User Circuit Breakers row documents.
- **Workers** — per worker: PID, uptime, active-request count, last-seen age, and a status badge (Active, Stale, or Warmup failed). This card appears in every deployment.

### Storage

The Storage tab summarizes the artifact store:

- **Storage Overview** — total items, total size, and the encryption and compression modes.
- **By Type** — item counts and sizes grouped by artifact type.
- **By Model** — a scrollable table of per-model storage usage.

A field reads `-` when the pipe could not read it, and the tab then shows a *Storage queries degraded*
banner naming the exception's class. A query that failed is isolated: the fields the other queries did
read stay filled in, and only the encrypted-items count drops its percentage, because a share of a
total that was never read is not a measurement — the count itself is still shown. An empty store reads
`0` with no percentage rather than `0 (0%)`.

### Config

The Config tab lists every admin valve for editing in place. See [Editing configuration](#editing-configuration) and [Live configuration updates](#live-configuration-updates) below.

### Update

The Update tab self-updates the pipe from tagged GitHub releases of the repo configured in
`PIPE_DASHBOARD_UPDATE_REPO` (the upstream project by default; point it at a fork to ship your own
builds — forks inherit the release workflow, so assets, digests, and the changelog keep working).

- **Installed / Latest cards** — installed version with a flat/compressed variant badge (locally
  derived; the card shows no dates — the row's timestamps are modification times, not install
  times, and a release date belongs to the release, shown on the Latest card); the latest release with date, size, "Last checked" (the time of the most recent check, whatever its
  outcome; a separate line appears while checks are failing, reporting when the current run
  of failures began), the tracked repo (with a `fork` badge when it is
  not the default), and the auto-update status. Every visit to the tab refreshes its data (release
  data is memoized server-side for a minute, so tab switching never hammers GitHub). **Check now**
  bypasses that memo for admins; for read-only viewers it refreshes the installed/snapshot state
  but reuses the memoized release data — force-refreshing the shared GitHub budget is admin-only.
- **Changelog** — the release page's own generated notes, rendered as escaped text; commit
  references appear as plain code text, not clickable links (the sandboxed panel cannot open new
  tabs). A fork that strips its CI ships no notes; the block then reads "changelog unavailable".
- **Update / Reinstall** — opens a confirm modal showing from→to, size, and a **Compressed bundle**
  checkbox preselected to the installed variant (an unavailable variant is disabled; `no_plugins`
  bundles are never offered — they would remove this dashboard). When you are already on the latest
  version the button reads **Reinstall** and re-applies the same release — that is the tab-native way
  to switch between the flat and compressed variants. Downgrades are refused here; restore a snapshot
  instead. Releases older than 2.7.0 — the first release carrying this tab — are refused outright:
  installing one would remove the updater itself. Applying downloads the asset (8 MiB cap), verifies its
  sha256 against the release digest, checks the frontmatter (id, newer version, required Open WebUI
  version), exec-validates the new bundle through Open WebUI's own loader, checks that the function's
  revision has not moved while it was loading, snapshots the current code, and only then writes the
  function row — and verifies the database accepted the write,
  failing the update loudly instead of reporting a success that did not persist. A load failure
  surfaces the real error in the tab and the pipe keeps serving the old code, and it leaves the pipe's
  on/off switch exactly as you had it — so a pipe you had already switched off in Workspace > Functions
  stays off rather than being switched back on by an update that failed. It is only when the database
  also refuses the write that puts a *live* row back that the tab reports `exec_failed_inactive` and tells
  you to switch the pipe on in Workspace > Functions. And a **refused write of the
  newly loaded code** is the same: the freshly loaded bundle is un-installed from the running process
  (every `sys.modules` key and `sys.meta_path` entry **the load itself** wrote is put back to its
  pre-attempt state, including the keys the compressed bundle deletes), the serving instance is rebuilt from the
  restored module, and the function cache is repointed at it — so "the previous version remains active"
  is literally true and the next chat is served, on both the one-click and the automatic path. This is
  reported as `write_failed`, separately from a validation failure, because the code passed every check
  and the store refused the row. The installing worker pauses briefly (up to ~90s); other
  workers pick the new version up on their next request. After a successful update, reload the
  dashboard to load the matching UI.
- **Previous versions** — the retained snapshots (`PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP`) with
  Restore and Delete. Both use a click-again confirmation: the first click arms the button
  (**Confirm restore** / **Confirm delete**, shown in red) and the second click within five seconds
  executes — native browser dialogs cannot appear inside the sandboxed panel, so the confirmation
  lives in the button itself. The Actor column shows the account name of whoever took the snapshot
  (`auto` for the auto-updater). Snapshots are stored as a fixed set of dedicated records in Open WebUI's own
  Files storage with their metadata (version, checksum, date, actor) on the file record itself —
  nothing about them lives in the function entry, so editing or re-pasting the function in the
  admin panel can never erase the rollback list. A restore re-validates the snapshot against its
  pinned sha256 before loading it, and once the revision check has passed it snapshots the current
  code, so restores are themselves undoable; a restore refused for a concurrent edit takes no
  snapshot at all, so the list is left exactly as it was and no older rollback point is rotated
  out. Delete double-checks the snapshot is still the one shown in the list before removing
  it and refuses with a refresh prompt if it changed. Snapshot records are owned by the primary
  admin account and are ordinary Open WebUI file records under the hood (there is no general file
  browser in the Open WebUI UI, so they stay out of the way, but they are listable through the
  files API) — manage them from this tab only: deleting them elsewhere, including the admin
  "delete all files" maintenance action, destroys rollback points.
- **Auto-update** — opt-in via `PIPE_DASHBOARD_UPDATE_AUTO`. In multi-worker deployments the
  workers elect a single update leader through Open WebUI's own Redis lock: only the leader checks
  GitHub (roughly every six hours, renewing its lease as it goes), while the other workers make no
  GitHub calls at all — they probe the lease hourly and take over if the leader dies or restarts
  (a gracefully stopped leader releases the lease immediately). Worker count therefore never
  multiplies update traffic. Single-worker installs (no Redis) skip the election and check directly.
  The leader applies a release once it is older than `PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS`
  (default 7 days — a bad release yanked within the window never reaches auto-updaters, and a fixed
  follow-up release supersedes it). The task runs headless (it keeps working while the dashboard
  model is disabled, as long as `PIPE_DASHBOARD_UPDATE_ENABLE` and the plugin system master switch
  `ENABLE_PLUGIN_SYSTEM` are both on), re-reads its valves from the
  database each cycle — so turning either of those off, or the auto-update valve itself, stops the
  *next* cycle rather than the current one, and turning the master switch back on resumes on the
  cycle after that with no restart — backs off on GitHub rate limits (honoring the reset header) and network
  failures without ever losing an update, and pauses a release on that worker after a deterministic
  failure until a restart, a newer release, or a successful manual apply. Those three ways out are
  offered only where one of them can work: a release the same check would reject again — one whose
  bundle declares a version that is not newer than the installed one, say — is cleared by neither a
  manual apply nor a restart, so the tab names the refusal instead of an instruction that cannot
  work, and the pause stands until a newer release arrives. An I/O error reaching the
  leader lease itself backs the loop off and retries rather than ending it, so a Redis blip costs
  one interval and not the rest of the worker's life. The tab's Auto-update line
  shows this worker's role (leader/follower), successes, pauses, and an unreachable leader lease; a
  pause also names *why* — the version it stopped on and the refusal's own sentence, versions
  included, so you can tell a rollback you need from a checksum that may clear on a retry.

Requirements for the checks and downloads: unauthenticated GitHub API (shared 60/hour budget per
egress IP — checks are memoized, and only one worker per deployment polls in the background).
Apply, restore, and snapshot-delete additionally require the acting account to hold the `admin`
role, as do the two configuration actions behind the Config tab; package/stub installs show a pin-bump
note instead of an Update button.

Every action in this tab — check, apply, restore, snapshot-delete, and the auto-tick — is also
refused when the server cannot read the *stored* valve set at all. That includes a database
that is unreachable at that moment, and a stored set that will not decrypt under the server's
current `WEBUI_SECRET_KEY` (a rotated key with valve encryption on does this). This differs from
the Config tab's behaviour on a rotated key, described below under [Editing configuration](#editing-configuration):
the two tabs genuinely differ, and this one refuses rather than showing defaults.

A worker the update service is not up on yet is refused the same way, and it is not an admin
disable either: the tab says the service is not ready on that worker and retries by itself, so
there is no valve to go and check. A `Pipe` whose start-up raised keeps that state for good, so
the sentence is worth reading as a fact about the worker rather than as something that will
clear by itself on every one of them.

### About

The About tab lists the registered plugins by name, id, and version.

---

## Editing configuration

The Config tab is the pipe's configuration editor. It lists every admin valve in a searchable, grouped tree, each with its own help text. Edit any value in place.

**How a save is stored.** A save records only the valves whose values differ from their defaults. Those valves show as **Custom** on Open WebUI's native valves screen; every other valve reads as its default. Clearing a **nullable** setting's box returns it to unset, so the saved row carries no name for it. Clearing a **`..._TEMPLATE`** box puts the built-in text back, so the saved row carries no name for that either — the box shows the factory wording again, ready to edit. Clearing an ordinary text setting's box stores an empty value, which is not the same as its default. A stored value that a later release no longer accepts is dropped at the next save, so the tab can never be wedged by a row it cannot write.

**When a value cannot be applied.** A save never fails for a value you did not type. If a setting stored by an earlier release is one this version no longer accepts — renamed, or a number whose allowed range has moved — it is dropped on its own, every other saved setting is kept, and nothing else on the row is affected. A setting this version *renamed* is the one exception: its value is carried across to the new name, the new control is shown pre-filled with it, and the old key is dropped rather than listed — the pipe log names the rename, so the two can be connected. The Config tab counts what it dropped in its note bar, and the save that dropped a value names them beside the count as `reset to default`; the pipe log names each one too. The next load of the configuration clears that note, because by then the row is repaired and there is nothing left to report. What is left when a stored value is dropped is the valve's default, so re-enter the value you want before you rely on it.

**Concurrent edits.** Each configuration carries a revision number, and that number is the stored row's own last-changed time, counted in whole seconds. Two administrators who save inside the same second therefore share one revision, so one revision can cover two saves and the revision by itself will not tell you that somebody else got there first. The revision also moves for writes that changed no setting you can see — Open WebUI's own Functions screen, an activation or global toggle, or an applied update — so the reload can be offered when nothing you changed has moved. This is why a save does not rely on the revision by itself: along with your edits it carries, for each setting you changed, the value that setting held when this tab loaded. Each of those is compared with what the store holds now. A setting that still holds the value you started from is saved as you left it; a setting somebody else changed in the meantime is not written, and the banner names it and your value for it stays in the unsaved bar. So a save that lands alongside another administrator's saves the settings neither of you touched and refuses only the ones that were actually contested, and nothing you typed is overwritten silently. If another administrator saves while you have the tab open, the tab loads the current values and keeps your unsaved edits, so you can re-apply your change on top of them. The reload takes every other setting from the new configuration; only the settings you have edited stay as you left them, and only an edit for a setting the server no longer publishes is dropped. Your pending values stay visible in the unsaved bar and in the setting's own pending line, so you can see both your value and the other administrator's, and revert per setting. A value the refreshed version no longer accepts is flagged for you rather than saved, and a value it now accepts is no longer held as an error; either way your typed value stays in the unsaved bar until you revert or save it. A store that cannot be read holds the save too, and the tab says so rather than writing over what it could not read. The banner carrying a **Reload** control appears when another administrator's save is detected and nothing else: a save refused in transport or by the framework — an expired session, a request the server throttled, a body it would not accept — says what refused it, names nothing was saved, and leaves your unsaved edits where they are instead of offering a Reload that would discard them. A store that refuses the write is the one other fault that raises a banner, and it raises the same strip **without** that control: the write never reached the row, so what you typed is the only copy of the change and a Reload there would discard exactly what the refusal kept. It names nothing was saved and says your edits are still here, and it goes by itself as soon as the next save is accepted or the tab next reloads the configuration — until then, every save will fail the same way. A value the stored schema rejects is neither of these: it names the fields in the toast, banners nothing, and is fixed by correcting the value. And two saves that arrive on *different* workers at the same moment are held apart by a Redis lease, the same primitive the updater uses to elect one worker: the first takes it and the second is refused outright — nothing was saved — rather than writing over the first administrator's edit. The revision cannot see that collision, because both saves carry the same whole-second value. On an install with no Redis there is no second worker to collide with, and the save behaves as it did before the lease existed.

**Upgrading from an older release.** If a stored setting is one this version no longer accepts, the Config tab repairs it on your next save, as described above. That repairs the **tab**. A row that takes the pipe down at start-up — because a value the boot path validates was tightened or renamed out from under it — is a different problem, and the Config tab cannot be opened at all until the row is fixed, so correct the stored value on Open WebUI's own Functions valves screen.

**Secrets.** Secret valves — API keys, passwords — are write-only. Their values stay on the server and never reach the browser. The tab shows each secret as **configured** or **not set**; typing a value sets a new one. A secret the pipe has stored can also be **cleared**, which removes the stored value and returns the setting to its default; the clear is reviewed like any other change before it is saved. A secret that is only coming from the environment has nothing stored to remove, so no Clear control is offered for it. After a clear is saved, the tab shows the secret as **not set** and stops offering Clear, because what it shows is the stored row's own state and not the request it just sent. Typing a value that happens to be the same as the environment's is still a save: the stored value is byte-identical to the default, and it is written like any other edit. Leaving a secret out of a save does not change it; a blank secret box changes nothing either.

If the stored configuration cannot be read at all — the database is unreachable, say — the Config tab says so and refuses to save, rather than showing defaults over your real settings. Your settings are still stored and are not being changed. Restore the database, then reload; the tab will not overwrite what it cannot read. A stored set that cannot be decrypted — a rotated `WEBUI_SECRET_KEY` with valve encryption on — is a different fault, and it decodes to an empty set just as a genuinely unset row does. The pipe reads the raw stored column to tell the two apart, so this case is reported the same way rather than shown as factory defaults: nothing is written over it, and the log names the likely cause. Restore the key, then reload; the settings come back.

**A save that lands over a store that then goes quiet.** The write is committed before the saved values are read back, so a save can go through and still come back with no list of what the store now holds — a database blip in that one read is the only cause. The tab keeps the value you typed and reports the save as made, and the settings you changed are the settings in the database. To confirm, restore the database and reload the tab; there is nothing to re-enter.

**Access.** The Config tab is for administrators end to end: reading the configuration and saving it both require the `admin` role, on top of a model grant. A write grant alone opens the tab's button but not its contents — it answers *forbidden* and the tab reports that it could not load the configuration.

---

## Live configuration updates

Open Config tabs follow the live configuration. When a valve changes — from a save in this tab, or an edit on Open WebUI's own valves screen — every open tab reflects the change on its own.

A tab reacts according to its edit state:

- **No unsaved edits.** The tab loads the current values and keeps your place: scroll position, selected valve, and search text.
- **Edits in progress.** The tab shows a notice with a **reload latest** control and leaves your edits in place until you choose.

---

## Requirements

**The interactive dashboard requires the iframe sandbox setting.** Open WebUI's **Settings →
Interface → iframe sandbox allow same origin** must be enabled. Without it the embedded page runs
with an opaque origin and cannot read the sign-in token, so every server-backed tab — Config,
Usage, and Update, as well as the live feed — fails to load or act (the Update tab says so
explicitly and points at this setting). Native browser dialogs are additionally unavailable in the
sandbox regardless of settings, which is why every confirmation in the dashboard uses inline
click-again-to-confirm buttons instead.

**A panel you scroll back to does not reconnect on its own.** Open WebUI keeps the rendered
dashboard inside the chat message, so reopening an old conversation re-runs it. Rather than let
every past panel quietly resume streaming live statistics, a panel this browser first saw more than a
few minutes ago opens in the DISCONNECTED state and says so; press **Connect** for live data. The
panel you have just opened with the command is unaffected.

**Content Security Policy.** If a restrictive `IFRAME_CSP` is configured, allow `script-src 'unsafe-inline'` and `connect-src 'self'` — the same policy the [OpenRouter Fusion panel](openrouter_fusion.md) uses.

---

## See Also

- [Plugin System](plugin_system.md) — The developer manual for building plugins.
- [Pipe Dashboard Internals](plugins_pipe_dashboard_internals.md) — The dashboard's internals and extension reference.
