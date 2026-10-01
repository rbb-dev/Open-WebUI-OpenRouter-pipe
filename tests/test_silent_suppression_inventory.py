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
    "core/logging_system.py": 3,
    # 1st: the drain cap in `_drop_member_log_buffer`, which waits for the pipe's own log
    # worker to turn a member's queued records into buffers before the member's key is
    # dropped. A wedged or absent worker must not hang a member's teardown, and the pop
    # runs on timeout anyway -- a straggler record is reaped by `cleanup()`'s one-hour age
    # window, which is today's behaviour for that record. It also swallows
    # `CancelledError` because the caller's own task may be cancelled while it waits, and
    # a release that must happen on every path cannot raise out of the cancellation that
    # is already unwinding the turn.
    # 2nd: the member's own `generation_complete` dispatch, which the member awaits
    # (bounded, five seconds) before it releases its own `on_generation_complete` mark.
    # The mark is only correct if the dispatch that writes it has already run: on a
    # cancelled member the dispatch is shielded inside the streaming loop, so its `add`
    # can otherwise land after the release and leave the id in a process-lifetime set for
    # the life of the worker. It also swallows `CancelledError` because the member's own
    # `finally` runs while the cancellation is unwinding, and the release that follows
    # must happen on that path too -- the close is the better answer to a cross-loop
    # close than a second orphan.
    "requests/fusion_engine.py": 2,
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
    # reports nothing about a successful reap either. The count is 20.
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
    # `byte_budget=True` rather than a `ProcessLookupError` the caller does not catch;
    # the stderr drain is inside the same shape for the same reason. Suppressing a
    # `CancelledError` there would swallow the caller's cancellation, which is why it
    # is listed explicitly rather than caught by the broad arm.
    "media/frame_extraction.py": 4,
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
    # 31st and 32nd: the two resets in `_pipe_impl`'s `finally` for a request that was never
    # enqueued (`clear_timing_events` for its early request id, `clear_timing_context`).
    # Same reason as the 25th: the request has already been answered with a degraded
    # result, the resets are bookkeeping on this task's own timing state, and a failure
    # there must not replace the answer the caller is receiving.
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
    "pipe.py": 40,
    "storage/persistence.py": 3,
    # 1st: the caller-supplied fallback in `_emit_templated_error_event`. It is reached only because the
    # admin's own template already failed to render, and the generic card below it is the answer if the
    # fallback fails too -- letting it out would replace the failure being reported with a template error.
    "streaming/event_emitter.py": 1,
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
    # turn's result or mask the exception that ended it. It also swallows `CancelledError` for the same
    # reason the two beside it do: a turn being torn down must not raise out of its own teardown.
    # 3 after B518 (H2437-1): the roster task's teardown. It carried two -- the task's own
    # cancellation on the way down, and the await of that task after the hand-back had
    # cancelled it. The second one existed to finalise a task nobody ever awaited, because
    # the guard skipped the await exactly when the task was finished: the only state that
    # has an exception. So the one suppression whose whole job was to hide a lost raise was
    # itself hiding the loss, and a card that could not be built reached nobody. The guard
    # now retrieves the exception unconditionally and reports a non-cancellation one by
    # model, and the cancellation arm is an explicit `except asyncio.CancelledError` that
    # ruff and this census both see.
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
