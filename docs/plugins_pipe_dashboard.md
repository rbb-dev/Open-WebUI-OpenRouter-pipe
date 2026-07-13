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

1. `ENABLE_PLUGIN_SYSTEM` — the master switch for the plugin system (default: off). Set it on before any plugin loads.
2. `PIPE_DASHBOARD_ENABLE` — adds the Pipe Dashboard model to the selector.

Three admin valves control the feature. They appear in Open WebUI's Settings once the plugin system is enabled.

| Valve | Type | Default | What it does |
|-------|------|---------|--------------|
| `PIPE_DASHBOARD_ENABLE` | bool | `False` | Shows or hides the Pipe Dashboard model in the model selector. |
| `PIPE_DASHBOARD_USAGE_COLLECT` | bool | `False` | Records one usage entry per completed request (user, model, tokens, tools, cost) to power the Usage tab. Read live: turning it on starts recording without a restart. |
| `PIPE_DASHBOARD_USAGE_RETENTION_DAYS` | int | `30` | How long usage records are kept. A background purge removes older records. Read live. |

---

## Access

Access is governed by Open WebUI's model access control on this overlay model — there is no separate admin flag. Assign users and groups in the model's **Access** editor:

- A **read** grant makes a **viewer**: open the dashboard and watch the live feed.
- A **write** grant makes an **operator**: everything a viewer can do, plus edit configuration and use the action buttons.

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

![Live dashboard](images/dashboard_live.jpeg)

The Live tab shows one row per in-flight or recently-completed request, across every worker. Each row carries:

- User and model. The model column shows display names and toggles to model slugs.
- A status badge: `queued`, `streaming`, `tool:<name>`, `completed`, `failed`, or `cancelled`.
- Elapsed time, tool success and failure counts, tokens (in → cached → out), cost, and the worker PID.

Cost updates live as the request runs; the completed row shows the final cost. Task-model calls — titles, tags, follow-ups — fold their cost into their parent chat's row.

Completed rows stay visible, dimmed, for the **Keep completed** window (5 minutes to 3 hours, default 10 minutes), set in the table itself.

The table sorts on any column and has a filter box.

### Usage

![Usage](images/dashboard_usage.jpeg)

The Usage tab needs `PIPE_DASHBOARD_USAGE_COLLECT` on. Without it, the tab shows a hint to enable collection.

With collection on, the tab presents:

- **Metric cards** — Sessions, Tokens, Cost, Tools, Errors, and Cached input, each with a change chip against the previous period of equal length and a per-bucket sparkline.
- **Resource cards** — live CPU, Memory, and Disk.
- **Usage trend** — a chart with two lines per bucket: tokens (left axis) and cost (right axis), with a hover tooltip and a timezone-aware range caption.
- **By model** — cost share per model. Each task-model appears as its own `model (tasks)` row with its own cost.
- **By user** — sorted by cost, showing the top 10 with an "N others" roll-up. Search and column-sort reach every user, including those inside the roll-up. A pinned **Totals** row at the bottom sums the visible rows (sessions, tokens, tools, cost).

Select a range: 1h, 6h, 24h, 7d, or 30d. Ranges longer than the retention window are disabled. A footnote shows the collection-start date, the retention window, and the record count.

**Invoice note.** Task models configured outside this pipe never reach it, so they are absent from these totals. Expect a small gap against the OpenRouter invoice when such task models are in use.

The Usage tables sort on any column and have a filter box.

### Health

The Health tab tracks the pipe's live load:

- **Concurrency** — active requests and tools, in-flight calls, and active video generations when the video pool is configured.
- **Queues** — pending requests and the log and archive queue depths, each with its bound.
- **User Circuit Breakers** — request, tool, and auth breaker counts. "Seen" counts distinct users since the worker started; "Users w/ fail" counts users with recent failures.
- **Models** — the model catalog with a per-type breakdown (text, image, video), the ZDR-capable count, and per-type fetch clocks. The status badge tracks the chat-catalog fetch loop.

### System

The System tab covers readiness and infrastructure:

- **Readiness** — initialization state, HTTP session, the session-logging worker, log-buffer RAM usage, and pipe-level Redis with a liveness ping. The session-logging worker reads **Idle** until the first record is persisted, which is its normal starting state.
- **Artifact DB** — the database write-pool backlog and the database circuit-breaker states.
- **Workers** — per worker: PID, uptime, active-request count, last-seen age, and a status badge (Active, Stale, or Warmup failed). This card appears in every deployment.

### Storage

![Storage](images/dashboard_storage.jpeg)

The Storage tab summarizes the artifact store:

- **Storage Overview** — total items, total size, and the encryption and compression modes.
- **By Type** — item counts and sizes grouped by artifact type.
- **By Model** — a scrollable table of per-model storage usage.

### Config

The Config tab lists every admin valve for editing in place. See [Editing configuration](#editing-configuration) and [Live configuration updates](#live-configuration-updates) below.

### About

The About tab lists the registered plugins by name, id, and version.

---

## Editing configuration

![Config tab](images/dashboard_config.jpeg)

The Config tab is the pipe's configuration editor. It lists every admin valve in a searchable, grouped tree, each with its own help text. Edit any value in place.

**How a save is stored.** A save records only the valves whose values differ from their defaults. Those valves show as **Custom** on Open WebUI's native valves screen; every other valve reads as its default.

**Concurrent edits.** Each configuration carries a revision number. If another administrator saves while you have the tab open, the tab holds your save, loads the current values, and lets you re-apply your change on top of them.

**Secrets.** Secret valves — API keys, passwords — are write-only. Their values stay on the server and never reach the browser. The tab shows each secret as **configured** or **not set**; typing a value sets a new one.

**Access.** A read grant opens the tab. Saving requires a write grant.

---

## Live configuration updates

Open Config tabs follow the live configuration. When a valve changes — from a save in this tab, or an edit on Open WebUI's own valves screen — every open tab reflects the change on its own.

A tab reacts according to its edit state:

- **No unsaved edits.** The tab loads the current values and keeps your place: scroll position, selected valve, and search text.
- **Edits in progress.** The tab shows a notice with a **reload latest** control and leaves your edits in place until you choose.

---

## Requirements

**Live mode requires the iframe sandbox setting.** Open WebUI's **Settings → Interface → iframe sandbox allow same origin** must be enabled for the live feed. With that setting off, the dashboard shows a static notice and opens no live feed.

**Content Security Policy.** If a restrictive `IFRAME_CSP` is configured, allow `script-src 'unsafe-inline'` and `connect-src 'self'` — the same policy the [OpenRouter Fusion panel](openrouter_fusion.md) uses.

---

## See Also

- [Plugin System](plugin_system.md) — The developer manual for building plugins.
- [Pipe Dashboard Internals](plugins_pipe_dashboard_internals.md) — The dashboard's internals and extension reference.
