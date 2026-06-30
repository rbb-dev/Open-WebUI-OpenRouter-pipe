# =============================================================================
# BEGIN LOCAL BACKPORT - anyio #1111 workaround
# -----------------------------------------------------------------------------
# Symptom: a completed asyncio.Task left in CancelScope._tasks causes
# anyio._backends._asyncio.CancelScope._deliver_cancellation to spin forever
# via call_soon, locking the event loop at 100% CPU until the worker is
# restarted. In production this is triggered when OWUI's MCPClient.disconnect()
# suppresses the cross-task RuntimeError from anyio - the scope is left half-
# exited with one done task lingering in _tasks, which feeds the spin.
#
# Upstream fix (not yet merged): anyio PRs #1138 and #1142, both proposing
#   if task.done(): continue
# early in CancelScope._deliver_cancellation. Issue and tracking PRs:
#   https://github.com/agronholm/anyio/issues/1111
#   https://github.com/agronholm/anyio/pull/1138
#   https://github.com/agronholm/anyio/pull/1142
#
# How this block works: at bundle exec time, before any of our pipe code runs,
# we install a wrapper around CancelScope._deliver_cancellation. The wrapper
# delegates to the original method, then checks: if all remaining tasks in the
# scope are done AND the original just scheduled another _deliver_cancellation
# iteration via call_soon, cancel that reschedule. This breaks the spin loop
# without mutating self._tasks (mutating it would trip anyio's task_done
# bookkeeping, which expects done tasks to remain in _tasks until task_done
# removes them itself). Python's attribute lookup means any existing
# CancelScope picks up the patched method on its next invocation, so
# already-spinning scopes recover within one tick.
#
# Version gating: only activates on known-buggy anyio versions. Any other
# version stands down with a WARNING reminding you to remove this block.
#
# Removal: delete this entire BEGIN..END block once anyio releases a version
# with PR #1138 (or #1142) merged. The version gate will quietly stand down
# before then, but the dead code should be cleaned up.
# =============================================================================
def _apply_anyio_1111_workaround() -> None:
    import logging as _logging
    import sys as _sys
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    _log = _logging.getLogger("open_webui_openrouter_pipe.anyio_1111_workaround")

    _KNOWN_BUGGY = {"4.12.1", "4.13.0", "4.14.1"}
    _MARKER = "_anyio_1111_workaround_applied"

    if "pytest" in _sys.modules or "_pytest" in _sys.modules:
        _log.debug("anyio #1111 workaround skipped: running under pytest")
        return

    try:
        ver = _pkg_version("anyio")
    except PackageNotFoundError:
        _log.warning("anyio #1111 workaround not applied: anyio not installed")
        return

    try:
        from anyio._backends._asyncio import CancelScope
    except Exception as exc:
        _log.warning(
            "anyio #1111 workaround not applied: cannot import CancelScope (%r)",
            exc,
        )
        return

    original = getattr(CancelScope, "_deliver_cancellation", None)
    if original is None:
        _log.warning("anyio #1111 workaround not applied: target method missing")
        return
    if getattr(original, _MARKER, False):
        _log.debug("anyio #1111 workaround already applied in this process")
        return

    if ver not in _KNOWN_BUGGY:
        _log.warning(
            "anyio #1111 workaround not applied: installed anyio %s not in "
            "known-buggy set %s. If anyio has shipped a fix for issue #1111 "
            "(PR #1138 / #1142), delete the anyio #1111 workaround block in "
            "scripts/anyio_1111_workaround.py.",
            ver,
            sorted(_KNOWN_BUGGY),
        )
        return

    def _patched_deliver_cancellation(self, origin):
        result = original(self, origin)
        try:
            tasks = self._tasks
            if tasks and all(t.done() for t in tasks):
                handle = getattr(self, "_cancel_handle", None)
                if handle is not None:
                    handle.cancel()
                    self._cancel_handle = None
        except Exception:
            pass
        return result

    _patched_deliver_cancellation._anyio_1111_workaround_applied = True  # type: ignore[attr-defined]
    CancelScope._deliver_cancellation = _patched_deliver_cancellation  # type: ignore[method-assign]

    _log.warning(
        "anyio #1111 workaround APPLIED for anyio %s (backport of upstream "
        "PR #1138 / #1142). Remove the workaround block in "
        "scripts/anyio_1111_workaround.py once anyio releases a fixed version.",
        ver,
    )


_apply_anyio_1111_workaround()
del _apply_anyio_1111_workaround
# =============================================================================
# END LOCAL BACKPORT - anyio #1111 workaround
# =============================================================================
