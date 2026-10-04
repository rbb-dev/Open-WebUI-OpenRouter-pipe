"""``contextlib.suppress(Exception)`` is ``except Exception: pass`` that no linter sees.

The two are semantically identical, and the second is what five commits in this project
went through the package to remove. The first survived all of them, because ``BLE001``
(blind except) and ``S110`` (try-except-pass) both match the ``except`` statement and
neither matches a context manager -- and ``SIM105``, which is on by default, actively
rewrites the second form into the first.

So the sweep closed the spelling ruff flags rather than the class. This file makes the
remainder visible: the count is pinned per module, so adding one is a deliberate edit
here rather than a silent widening, and removing one shows up as work done.

This is an inventory, not a prohibition. Some of these are correct -- suppressing a
failure whose only consequence is the suppression itself. What was wrong was that
nothing distinguished those from the ones with a named cost, such as the temp-directory
cleanup in ``integrations/video.py`` whose own comment said the failure mode was inode
exhaustion, and which is now logged.
"""

from __future__ import annotations

import ast
import os
from collections import Counter
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"
# scripts/ too, because scripts/anyio_1111_workaround.py is emitted into the head of all
# four bundles and is therefore production code. Its two sibling censuses in
# test_swallowed_failure_diagnostics.py already walk both roots; this one did not, and
# the shipped artifact's first executable statement sat in the gap between them.
SCAN_ROOTS = [PACKAGE_DIR, Path(__file__).resolve().parents[1] / "scripts"]
def _broad_suppressions(source: str) -> int:
    """Count every `suppress(...)` that swallows Exception, however it is spelled.

    A regex over `suppress(Exception)` saw one spelling. It missed
    `suppress(asyncio.CancelledError, Exception)` -- strictly BROADER, since it also
    swallows cancellation -- and any `from contextlib import suppress as _quiet`.
    Matching the call in the AST cannot be spelled around.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0

    aliases = {"suppress"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "contextlib":
            for a in node.names:
                if a.name == "suppress":
                    aliases.add(a.asname or a.name)

    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in aliases:
            continue
        if any(
            (getattr(a, "id", None) or getattr(a, "attr", None)) in {"Exception", "BaseException"}
            for a in node.args
        ):
            total += 1
    return total

_EXPECTED: dict[str, int] = {
    # Wraps only the diagnostic itself: the workaround must not fail a request because
    # its own logging failed, and it runs before any pipe module exists.
    "../scripts/anyio_1111_workaround.py": 1,
    "core/logging_system.py": 4,
    # 1st: the drain cap in `_drop_member_log_buffer`, which waits for the pipe's own log
    # worker to turn a member's queued records into buffers before the member's key is
    # dropped. A wedged or absent worker must not hang a member's teardown, and the pop
    # runs on timeout anyway -- a straggler record is reaped by `cleanup()`'s one-hour age
    # window, which is today's behaviour for that record. It also swallows
    # `CancelledError` because the caller's own task may be cancelled while it waits, and
    # a release that must happen on every path cannot raise out of the cancellation that
    # is already unwinding the turn.
    # A second one used to sit here: the member's own `generation_complete` dispatch,
    # which the member awaited (bounded, five seconds) before releasing its own
    # `on_generation_complete` mark. Both that await and the release it guarded are gone
    # -- no dispatch site a member can reach is reachable, because every one of them is
    # `fusion_inner`-guarded, so nothing ever writes that mark and there was nothing for
    # the release to take back out. `tests/test_no_fusion_dispatch_site_adds_a_member_id.py`
    # is the census that pins the guards, so a future unguarded site is caught there.
    # 2nd: `_archive_publish_changed_file`'s `stat()` after the write, whose only
    # consequence is falling back to "unchanged", i.e. keeping the rows.
    # 3rd: unlinking the finished temporary archive when the publish-time ownership guard
    # refuses, so a pass that abandoned its write leaves no full archive on disk under a
    # name the reader would take for the turn's. A refusal must not be able to raise out
    # of the writer, which swallows its own failures by contract, and an unlink that fails
    # is reaped by the same cleanup sweep either way.
    "requests/fusion_engine.py": 1,
    # 7th: the cost snapshot, now one guarded helper reached from all three exits. A job
    # OpenRouter has already billed for is recorded whatever the pipe does with the
    # bytes, and a storage error while recording it must not replace the failure the
    # user is being shown -- nor take down a cancellation that is already unwinding.
    "integrations/video.py": 7,
    # 19th: the diagnostic, reassembled below. Measured from the finished tree, not
    # derived: B7-3's stamp restore dropped its `suppress` for a `try`/`except` that
    # reports a failed restore, and the stranded-turn capture now routes its row
    # delete through `_release_assembly_lock`, which is itself one guarded helper.
    # The success path keeps its own inline suppression, and the capture adds none.
    # T453 added the nineteenth: the writer thread's `task_done()` after it learns
    # the manager died. `task_done` is bookkeeping on a queue the dying manager will
    # never again read, and a ValueError from over-counting it must not take the
    # thread's exit down with it.
    # H685-1 added the twentieth and twenty-first: the two `rescue_path.stat()` calls
    # around the stranded-turn capture's write. They are the stat-diff the sibling
    # `_assemble_and_write_bundle` already uses for the same question -- did this
    # write publish -- and a `stat()` that raises has no consequence the caller's own
    # report does not already cover, because a path that cannot be stat'd cannot have
    # been written either, so the capture reports the failure and leaves the rows.
    # B193 then removed the inline `after_stat` stat-diff in `_assemble_and_write_bundle`
    # (hoisted into the shared `_archive_publish_changed_file` helper), so the count was 20.
    # B312-2 then deleted the whole `enqueue_archive` method, which carried the
    # `contextlib.suppress(Exception)` around its `_dirs.add(base_dir)`, so the count is 19.
    # B421 added `_cleanup_stale_segments`, which reaps the staged segment rows the
    # artifact sweep is now scoped away from. Its suppression wraps the same
    # `_delete_artifacts_sync` call the lock reaper above it already suppresses for the
    # same question -- could this row be removed -- and with the same reasoning: the ids
    # have already been selected, the delete is best-effort, and a delete that raises
    # leaves rows that the next pass, on its own interval, selects again. Suppressing it
    # does not hide a consequence the caller would otherwise report, because this pass
    # reports nothing about a successful reap either.
    # B622 then removed the last suppression on the *post-publish* delete in
    # `_assemble_and_write_bundle`. That one is not like the reaps above it: the archive
    # is already written and `wrote` is already true, so a delete that raises leaves both
    # the segment rows AND the assembly lock behind a "successful" assembly, and the next
    # pass re-acquires the turn, re-writes it and re-fails the same delete. That is the one
    # route on this path to unbounded starvation, so it is now reported on the existing
    # `_unreadable_archive_warnings` latch rather than swallowed.
    # 20th: the join of a thread displaced from one of the three slots, in `stop_workers`'
    # second loop over `_retiring_threads`. Same shape and the same reason as the three slot
    # joins above it: `stop_workers()` is best effort by contract, it runs on the `close()`
    # path, and a thread that cannot be joined must not raise out of the teardown -- it is a
    # thread that is already retiring, which is what the join is waiting for.
    "logging/session_log_manager.py": 20,
    # 3rd: the generic ffmpeg arm, which now stops the child it started before it
    # reports a transport fault. The suppression is load-bearing and is not a test
    # guard: PIL's `UnidentifiedImageError` subclasses `OSError`, so the common
    # corrupt-output decode failure lands in that arm on a child asyncio has ALREADY
    # reaped, and `kill()` on a reaped child raises `ProcessLookupError`. Without the
    # suppression that path surfaces `ProcessLookupError` instead of
    # `FrameExtractionError`, and the caller in `integrations/video.py` catches only
    # the latter -- so an escaping `ProcessLookupError` would fall to the outer
    # `except Exception` and degrade to a generic `materialise_failed` instead of
    # the honest `frame_extract_failed_idx_*` disclosure.
    # 4th: `_stop_child`, the shared kill-and-reap the read-cap abort and the timeout
    # both use. `kill()` on a child asyncio has already reaped raises
    # `ProcessLookupError`, and a `wait()` on a closed transport raises too, so the
    # suppression is what keeps the byte-budget refusal a `FrameExtractionError` with
    # `byte_budget=True` rather than a `ProcessLookupError` the caller does not catch.
    # Suppressing a `CancelledError` there would swallow the caller's cancellation,
    # which is why it is listed explicitly rather than caught by the broad arm.
    # H4434-1 moved the stderr drain out of `_stop_child` and into the new `_reap`, which
    # took the drain's suppression with it (so 4 -> 3 there) and added two of its own.
    # 5th: `_reap`'s two post-kill steps, the wait for a child it has just killed and the
    # collection of the child's stderr. The bound is what the rung owes; once it is spent
    # the kill is sent and the wait is the other half of the reap, because `kill()`
    # signals and `wait()` reaps -- a killed child nobody waited on is one unreaped child
    # per wedged clip for the life of the worker. The drain is diagnostics, and the
    # diagnostics the pipe keeps are the returncode and the pipe's own wording: a `.result()`
    # on a finished drain re-raises a read fault the caller has no better answer for, and
    # the bounded `wait_for` on an unfinished one leaves the bytes unread and the rung's
    # deadline error unreplaced. Neither changes what the caller is told -- `_reap` answers
    # the same `_rung_deadline_error` on the reaped path and on this one -- so a failure to
    # collect them is a missing word, not a wrong one. `CancelledError` is listed because
    # both of these need it: the caller may be cancelled while the reap waits, and the
    # drain is the task the line above cancelled whenever it did not answer, so awaiting
    # it re-raises the cancellation the reap asked for.
    "media/frame_extraction.py": 5,
    "models/catalog_manager.py": 2,
    # 19th: `MultimodalHandler.aclose()` during shutdown, closing the vetted transport's
    # session and its decode pool. It sits among the teardown steps either side of it,
    # all of which suppress for the same reason: a failure here has no consequence
    # beyond itself, and letting it out would skip the cleanup that follows.
    # 22nd-24th: the pooled request session's teardown and retirement. Closing a
    # session whose loop has already gone away can fail on any transport resource, and
    # the caller is either shutting down or building a replacement -- in both cases there
    # is nothing left to serve, and a raised close would skip the rest of the cleanup.
    # 21st: clearing the pointer to the background Web Tools repair task in `_do_close`.
    # The task has just been cancelled, so a failure here is a stale reference at worst;
    # letting it out would abort the shutdown that follows.
    # One more: the `shutdown()` inside `__del__`, which runs at interpreter teardown where an
    # exception escaping has no caller to see it; the close is scheduled outside the suppression.
    # T453 added the second half of that destructor: the per-awaitable disposal of what
    # `dispatch_on_shutdown` returned. The shapes differ -- a coroutine has `close()` and no
    # `cancel()`, a Task the reverse, and a third-party plugin may hand back neither -- so the
    # getattr-and-suppress is what makes the destructor total. It must be: an exception out of
    # `__del__` is printed and ignored, and a bare awaitable left undisposed is a RuntimeWarning
    # from the collector instead.
    # Two more: the `OWUI_CHAT_ID.reset(token)` calls in the `finally` blocks that undo the
    # channel-chat-id token; resetting a ContextVar is bookkeeping for the next request, and letting
    # one out would replace the turn's real outcome with an error raised while unwinding it.
    # 25th: the per-call reset of a timing contextvar token in `pipe()`'s `finally`.
    # The token belongs to this task's Context, and the reset only raises if the request
    # crossed a task boundary -- in which case the token is already spent and the next
    # request's own reset mints its own. Letting it out would replace the answer the
    # caller is still waiting for with a bookkeeping error.
    # 30th: the `pipes()` refresh session's close, in the `finally` that the key
    # resolution moved inside of. A close that fails after the refresh has already been
    # turned into a `refresh_error` would replace a reported catalogue fault with a
    # teardown fault and skip the `list_models()` that decides what the picker is
    # served, so the close is the last thing allowed to fail silently. Same reason as
    # the pooled session closes above: the caller is already on its way out and there
    # is nothing left to serve.
    # 31st: the reset in `_pipe_impl`'s `finally` for a request that was never enqueued
    # (`clear_timing_context`). It was the 31st and 32nd while that `finally` also cleared
    # the request's timing events; the events themselves are gone (B680, T967), so the
    # suppression that guarded that clear went with it. Same reason as the 25th: the
    # request has already been answered with a degraded result, the reset is bookkeeping
    # on this task's own timing state, and a failure there must not replace the answer the
    # caller is receiving.
    # 33rd (B218, H742-1): the resize of the shared connector in `_shared_request_session`
    # when the concurrency valve was raised. It sets aiohttp's plain `_limit` attribute and
    # then calls the private `_release_waiter()`; if that call is missing or raises, the
    # raised limit is already in place and waiting requests take the new slots at the next
    # connection release, so the only consequence is a later wake, not a lost resize.
    # 34th (B223, H784-1): the release of the worker's own permit, now in the module-level
    # `_release_permit` helper in `pipe.py`, when the job died before the
    # `async with _acquire_semaphore(...)` took it over and before the body's own release
    # claimed it; the dispatch loop's done callback reaches the same helper for a job
    # cancelled before its first step. A semaphore that raises on `release()` -- a
    # loop-bound one whose loop is gone -- must not replace the failure that is already
    # unwinding the request; the permit is already accounted for either way.
    # 35th (B248, H900-1): the close-path drain in `_do_close`, which commits whatever is
    # still in the Redis pending queue before the client goes down. It swallows
    # `CancelledError` for the same reason the four in `streaming/streaming_core.py` do: a
    # teardown step must not raise out of its own teardown. A `CancelledError` raised here
    # skips the entire remainder of `_do_close` -- the Redis socket stays open and the
    # artifact store's DB executor is never shut down -- and a drained-queue failure must
    # not stop the socket from being closed either, since the rows are then still in Redis
    # for another worker. The drain reports its own failures through the store's logging.
    # 34 after B288 (H1033-1): the terminal backstop's `suppress(Exception)` around
    # `job.future.exception()`. The backstop now reads the pipeline's recorded terminal
    # first and falls back to the future only where it recorded none, and that read is
    # `_future_failed` -- a named predicate with its own narrow `except`, rather than a
    # context manager that hid the whole status derivation from the census and from ruff.
    # 35th (B382, H1624-4): the cancel of a log worker stranded on a closed loop, in
    # `_maybe_start_log_worker`. Same reason as the one the loop-swap block above it
    # already takes for the same class of task, and the same shape: the worker is being
    # discarded because its loop can never run it again, the slot is nulled on the next
    # line either way, and a task object that refuses to be cancelled must not stop the
    # fresh worker from being started on this loop. Its loop is closed, so there is
    # nothing on it left to log the failure to.
    # 36th (B325/H1428-2): `cancel()` on a task whose loop is closed raises `RuntimeError`
    # out of `call_soon`. `_drop_task` is the shared helper every per-loop task slot uses
    # when it replaces a task bound to a loop that is not the current one, and it is called
    # from `__init__` and `_pipes` where there may be no running loop at all. The task is
    # dead either way -- nothing will ever run it -- so the drop must still happen rather
    # than faulting the caller.
    # 37th, 38th (B325/H1316-3): the two `put_nowait` calls in `_wake_refused_stream`, which
    # wake a refused job's stream generator with the terminal item and the end-of-turn
    # sentinel. A full queue is the only failure mode, and the generator's own `finally`
    # still answers the caller; letting the refusal raise out of the worker's admission
    # path would fault the request queue instead of the one job being shed.
    # B325's own count for this module was 38 on the 00c4bb62b tree, where the backstop still
    # read the future through its own `suppress(Exception)`; B288 replaced that read with
    # `_future_failed`, and B382 added the 35th above, so B325's three land on 35: 35 + 3 = 38.
    # 39th (B300, H1134-1): the pooled session's own timeout refresh in
    # `_shared_request_session`. It writes aiohttp's private `_timeout`, the same idiom the
    # 33rd above already uses for the connector; the write is guarded by `valves is not
    # None` and the new test asserts the value actually changed, so a suppression that hid
    # a no-op would fail there rather than pass silently.
    # 40th (B407 H1685-1): the done-callback's `suppress(Exception)` around
    # `task.exception()` in `_forget_abandoned_tool_task`, which retires a tool call that
    # outlived the batch ceiling's cancel grace. The callback exists only to drop the
    # reference and retrieve the exception so a tool that ignored its cancellation and
    # then raised does not log "Task exception was never retrieved"; it runs after the
    # round has already reported the call to the model, so a retrieval failure has nothing
    # left to affect and must not raise out of a done callback, where asyncio would only
    # log it.
    # The 41st and 42nd, the two release sites for a discarded request's timing profile
    # (one in `_stream`'s own `finally`, one beside the counter release in
    # `_abandon_request_queue`), are gone: a function that cleared nothing but carried a
    # name saying a request's timing state was retained is dead code, and deleting it
    # took both suppressions with it (B680, T967; see also the 31st above). The file is
    # what the timing valve keeps, and nothing in the teardown touches it. Those three
    # removals are what leave 40 of the 43 the base commit's count stood at; the 43rd
    # below was added after B680 built and is unaffected by them.
    # 43rd: the per-edge `setattr(owner, attr, None)` in `_do_close`'s back-reference
    # clear. The loop drops the seven edges the pipe's own collaborators hold back to it,
    # and every owner is one this constructor just built, so an attribute that cannot be
    # set means the collaborator is not the shape the loop names. Failing to clear one
    # leaves the pipe reachable through its own attributes and only a gc pass reclaims it;
    # failing to clear it loudly would abort the rest of the loop and leave the other six
    # edges standing, which is the worse of the two. The clear runs on the teardown path,
    # after every drain has already finished, so a raise here has nothing left to protect.
    # 41st and 42nd (B1216-1): the two `suppress(Exception)` around the contextvar resets at
    # the tail of `_stream()`'s `finally`, one for `OWUI_CHAT_ID` and one for `CONTINUED_REPLY`.
    # They run after the generator has published whatever it published and cleaned the
    # session logger up, so a reset that failed -- a token from another context, which is the
    # only way a reset raises here -- has nothing left to protect: the turn is over and the
    # caller is about to run the next one on this task. Letting it out would replace a
    # published refusal with a `ValueError` out of a generator's cleanup, and a generator
    # that raises from its `finally` hands the caller an error instead of the frames it has
    # already read. The leak the reset prevents is a channel id reaching the next request,
    # and that is held by the test that reads both contextvars after an abandoned stream.
    "pipe.py": 42,
    # 4th (B724): the done-callback's `suppress(asyncio.CancelledError, Exception)` around
    # `task.exception()` in `_schedule_redis_valve_drain_on`'s `_settle`, which releases
    # the Redis valve's ownership latch when a scheduled drain ends however it ends. The
    # same shape and the same reason as the 40th above in `pipe.py`: the callback exists
    # only to retrieve the exception so a drain that raised does not log "Task exception
    # was never retrieved" on the way to being garbage-collected, and the latch release
    # below it must still run. Letting a retrieval failure out would raise inside a done
    # callback, where asyncio can only log it, and would skip the release that re-arms
    # the valve -- so the swallow is what keeps the recovery total, and `CancelledError`
    # is swallowed with it because the callback fires for a cancelled drain too.
    "storage/persistence.py": 4,
    # 1st: the caller-supplied fallback in `_emit_templated_error_event`. It is reached only because the
    # admin's own template already failed to render, and the generic card below it is the answer if the
    # fallback fails too -- letting it out would replace the failure being reported with a template error.
    "streaming/event_emitter.py": 1,
    # `plugins/pipe_dashboard/http_routes.py` used to carry one: the done-callback's
    # `suppress(asyncio.CancelledError, Exception)` around `task.exception()` in
    # `_consume_task_exception` (B916, H2434-2), attached to the detached last-active
    # refresh `bearer_user` fired. The route now resolves identity through Open WebUI's
    # own `get_verified_user`, so the refresh is OWUI's (`auth.py:483`), fired without a
    # callback -- exactly as it is on every route Open WebUI has -- and both the callback
    # and the suppression go with it.
    # 1st: the close of a vetted transport whose connector belongs to a different event
    # loop, in `_retire_vetted_session`. Measured rather than assumed: with one pooled
    # keep-alive connection the cross-loop `session.close()` raises `RuntimeError` ("got
    # Future ... attached to a different loop") only AFTER the teardown has already
    # happened -- `session.closed=True`, `connector.closed=True`, the pool drained -- so
    # the raise is noise about work that succeeded. Letting it out instead put a visible
    # `RuntimeError` on a request path (`_acquire_vetted`), for every address-gated
    # download on the new loop, which is strictly worse than the leak the branch was
    # closing. The loop-mismatch guard above it is what stops the close from running at
    # all against a foreign connector that still has an in-flight request on it.
    "storage/multimodal.py": 1,
    # 1st: the roster task teardown above, whose only failure mode is a second
    # cancellation arriving while the loop is already unwinding.
    # 3rd: the loop-limit note's write of `{"error": {"content": …}}` to the saved chat row, so the
    # banner survives a reload the way Open WebUI's own `emit_message_error` makes it survive. It
    # runs after the notification has already reached the caller, so a storage failure costs the
    # persisted copy alone and must not replace the note the person is being shown.
    # 4th: the turn's own `event_iter.aclose()`, the first act of the streaming loop's `finally`. It
    # is a teardown on a path that is already leaving, and it is the one await there that can block
    # (delegated three generators deep, through the adapter whose `finally` awaits its workers), so it
    # runs shielded. A failure to close costs the release of the producer, its workers and the aiohttp
    # response -- which the garbage collector would eventually do anyway -- and must not replace the
    # turn's result or mask the exception that ended it.
    # 3 after B518 (H2437-1): the roster task's teardown. It carried two -- the task's own
    # cancellation on the way down, and the await of that task after the hand-back had
    # cancelled it. The second one existed to finalise a task nobody ever awaited, because
    # the guard skipped the await exactly when the task was finished: the only state that
    # has an exception. So the one suppression whose whole job was to hide a lost raise was
    # itself hiding the loss, and a card that could not be built reached nobody. The guard
    # now retrieves the exception unconditionally and reports a non-cancellation one by
    # model, and the cancellation arm is an explicit `except asyncio.CancelledError` that
    # ruff and this census both see.
    # 4th (B651): the abandoned turn's best-effort flush of whatever its cancelled writer
    # had popped, in `_run_streaming_loop`'s teardown. It runs inside the `finally` of a
    # frame the cancellation is unwinding, so it is reached with a cancellation already
    # delivered and a second one raises out of the writer's await; letting that out would
    # replace the turn's own cancellation -- which the caller is waiting on -- with an
    # error from a write nobody is waiting for. The rows it fails to write are named by the
    # two warnings beside it, so nothing the suppression costs is silent, and
    # `test_a_cancelled_flush_returns_the_rows_it_had_popped.py` is what says a flush
    # cancelled mid-flight hands its batch back rather than dropping it.
    # 3 after B1224: the aclose's suppression is gone, and the count is what that left.
    # The site still absorbs -- it is a teardown, and a close that fails must not replace
    # the turn's result -- but it no longer swallows *only*: `await asyncio.shield(...)`
    # is now a `try`/`except (asyncio.CancelledError, Exception)` that remembers a
    # `CancelledError` in `_finalise_cancelled` and re-raises it once the tail has run.
    # The rationale above said the opposite, that swallowing is what the site is for,
    # which is why it is rewritten here rather than merely renumbered: the site still
    # absorbs, so the turn keeps publishing, and it also remembers, so the turn still
    # ends cancelled. Both halves are load-bearing -- the shield is what keeps the Stop
    # from leaving the teardown half-done, and the recording is what stops the absorbed
    # Stop from being reported to `pipe()` as a turn that finished.
    # What is left is the thinking-task drain (`:4562`), the abandoned flush (`:4854`),
    # and the loop-limit note's row write (`:4412`), which is in the loop body and not
    # in the tail at all.
    "streaming/streaming_core.py": 3,
    # 5th: the tool card emitted as each call's result is collected, the twin of the one in the loop that
    # follows. The card is what the person sees; a failure emitting it must not lose the tool result the
    # loop is in the middle of collecting, which is the model's answer.
    "tools/tool_executor.py": 5,
}


def scan_broad_suppressions() -> Counter[str]:
    """The census body, callable without going through the test.

    Extracted because `test_every_source_census_walks_the_same_roots` drives the censuses to
    record which roots each one reads. Calling the TEST as a plain function bypassed its
    `skipif`, so in the four bundled modes pytest reported this scan skipped while it in
    fact ran, and any failure surfaced under the roots-recording test's name -- a message
    about directory coverage for a defect that had nothing to do with it.

    Keeps its own `rglob`: the recorder monkeypatches `Path.rglob` to observe the walk,
    so this one must not come from the shared cache in tests/package_sources.py.
    """
    found: Counter[str] = Counter()
    for path in sorted(q for root in SCAN_ROOTS for q in root.rglob("*.py")):
        hits = _broad_suppressions(path.read_text(encoding="utf-8"))
        if hits:
            root = next(r for r in SCAN_ROOTS if r in path.parents)
            key = str(path.relative_to(root)).replace("\\", "/")
            if root != PACKAGE_DIR:
                key = f"../{root.name}/{key}"
            found[key] = hits
    return found


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_broad_suppressions_are_inventoried_per_module():
    """Any change to the count is a reviewed edit, in either direction."""
    found = scan_broad_suppressions()

    assert dict(found) == _EXPECTED, (
        "the inventory of broad suppressions has changed.\n"
        f"  added:   { {k: v for k, v in found.items() if _EXPECTED.get(k, 0) < v} }\n"
        f"  removed: { {k: v for k, v in _EXPECTED.items() if found.get(k, 0) < v} }\n"
        "A new `suppress(Exception)` swallows a failure where no linter will ever ask "
        "why. If the failure genuinely has no consequence, record it here with that "
        "reason; if it has one, log it instead."
    )


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_the_pattern_still_matches_something():
    """Guards the regex: a scan that matches nothing passes every tree."""
    assert sum(_EXPECTED.values()) > 0
    total = sum(
        _broad_suppressions(p.read_text(encoding="utf-8"))
        for root in SCAN_ROOTS
        for p in root.rglob("*.py")
    )
    # Per-root, so widening the scan to a directory that contributes nothing reads as
    # coverage it does not provide.
    for root in SCAN_ROOTS:
        assert any(
            _broad_suppressions(p.read_text(encoding="utf-8")) for p in root.rglob("*.py")
        ), f"{root.name} contributes no scanned suppression; it is not really covered"
    assert total == sum(_EXPECTED.values()), (
        f"the scan found {total} suppressions against an inventory of "
        f"{sum(_EXPECTED.values())}"
    )
