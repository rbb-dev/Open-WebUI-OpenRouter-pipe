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
# Upstream fix: anyio PR #1217 ("Fixed 100% CPU spin on cancel scope misuse",
# Fixes #1111), released in anyio 4.14.2 on 2026-07-12.
#   https://github.com/agronholm/anyio/issues/1111
#   https://github.com/agronholm/anyio/pull/1217
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
# Version gating: applies on any anyio older than the fixed release (< 4.14.2)
# and stands down on 4.14.2 and later, which already contain the fix.
#
# Removal: delete this entire BEGIN..END block once the deployment can require
# anyio >= 4.14.2.
# =============================================================================
def _apply_anyio_1111_workaround() -> None:
    import logging as _logging
    import sys as _sys
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    _log = _logging.getLogger("open_webui_openrouter_pipe.anyio_1111_workaround")

    _FIXED_IN = "4.14.2"
    _MARKER = "_anyio_1111_workaround_applied"

    if "pytest" in _sys.modules or "_pytest" in _sys.modules:
        _log.debug("anyio #1111 workaround skipped: running under pytest")
        return

    try:
        ver = _pkg_version("anyio")
    except PackageNotFoundError:
        _log.warning("anyio #1111 workaround not applied: anyio not installed")
        return

    def _before_fix(installed: str, fixed: str) -> bool:
        def _nums(v: str) -> list[int]:
            out: list[int] = []
            for part in v.split("."):
                digits = ""
                for ch in part:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                out.append(int(digits) if digits else 0)
            return out

        a, b = _nums(installed), _nums(fixed)
        width = max(len(a), len(b))
        a += [0] * (width - len(a))
        b += [0] * (width - len(b))
        return a < b

    if not _before_fix(ver, _FIXED_IN):
        _log.debug(
            "anyio #1111 workaround not applied: anyio %s already includes the "
            "fix (>= %s, PR #1217)",
            ver,
            _FIXED_IN,
        )
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
        "anyio #1111 workaround APPLIED for anyio %s. Fixed upstream in anyio "
        "%s (PR #1217) - upgrade anyio to >= %s and delete the workaround block "
        "in scripts/anyio_1111_workaround.py.",
        ver,
        _FIXED_IN,
        _FIXED_IN,
    )


_apply_anyio_1111_workaround()
del _apply_anyio_1111_workaround
# =============================================================================
# END LOCAL BACKPORT - anyio #1111 workaround
# =============================================================================
