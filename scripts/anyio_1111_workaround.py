def _apply_anyio_1111_workaround() -> None:
    import contextlib as _contextlib
    import logging as _logging
    import sys as _sys
    from importlib.metadata import version as _pkg_version

    def _pipe_logger():
        """Resolve the logger under whatever root the pipe wires its handlers onto.

        This file is emitted into the bundle HEAD in both bundle shapes, but the two
        disagree about where the package lives: the flat bundle collapses everything
        into the host module, while the compressed bundle keeps the real dotted names.
        A name fixed at import time is therefore wrong in one shape or the other --
        the literal orphaned every record in flat mode, and deriving it from __name__
        orphaned them in compressed mode.

        Resolving per call follows the package once it exists. It cannot help the
        apply-time messages below: the head calls this function before importing any
        pipe module, so the package is not yet in sys.modules and the fallback wins.
        Those records also predate Pipe wiring its handlers, so they reach the host's
        root handlers either way -- the name matters only for the cleanup warning at
        the bottom, which fires long afterwards and re-resolves.
        """
        module = _sys.modules.get("open_webui_openrouter_pipe")
        root = getattr(module, "__name__", None) or __name__
        return _logging.getLogger(f"{root.split('.')[0]}.anyio_1111_workaround")

    _logger = _pipe_logger()

    _FIXED_IN = "4.14.2"
    _MARKER = "_anyio_1111_workaround_applied"

    if "pytest" in _sys.modules or "_pytest" in _sys.modules:
        _logger.debug("anyio #1111 workaround skipped: running under pytest")
        return

    try:
        ver = _pkg_version("anyio")
    except Exception:
        _logger.warning(
            "anyio #1111 workaround not applied: could not read anyio's version",
            exc_info=True,
        )
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
        _logger.debug(
            "anyio #1111 workaround not applied: anyio %s already includes the "
            "fix (>= %s, PR #1217)",
            ver,
            _FIXED_IN,
        )
        return

    try:
        from anyio._backends._asyncio import CancelScope
    except Exception as exc:
        _logger.warning(
            "anyio #1111 workaround not applied: cannot import CancelScope (%r)",
            exc,
            exc_info=True,
        )
        return

    original = getattr(CancelScope, "_deliver_cancellation", None)
    if original is None:
        _logger.warning("anyio #1111 workaround not applied: target method missing")
        return
    if getattr(original, _MARKER, False):
        _logger.debug("anyio #1111 workaround already applied in this process")
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
            # Inlined rather than calling core.warn_latch. Not because of ordering --
            # this runs when anyio delivers a cancellation, long after load -- but
            # because this file is the BUNDLE HEAD, and bundle_v2's validator rejects
            # any absolute internal import surviving there ("leftover absolute internal
            # import"). The census in test_swallowed_failure_diagnostics.py knows about
            # this one site by name and fails if a second appears.
            _seen = getattr(_patched_deliver_cancellation, "_cleanup_failure_logged", False)
            _patched_deliver_cancellation._cleanup_failure_logged = True  # type: ignore[attr-defined]
            with _contextlib.suppress(Exception):
                _cleanup_logger = _pipe_logger()
                _cleanup_logger.log(
                    _logging.DEBUG if _seen else _logging.WARNING,
                    "anyio #1111 workaround: cancel-handle cleanup failed; the "
                    "workaround may no longer be effective",
                    exc_info=True,
                )
        return result

    setattr(_patched_deliver_cancellation, _MARKER, True)
    CancelScope._deliver_cancellation = _patched_deliver_cancellation  # type: ignore[method-assign]

    _logger.warning(
        "anyio #1111 workaround APPLIED for anyio %s. Fixed upstream in anyio "
        "%s (PR #1217) - upgrade anyio to >= %s and delete the workaround block "
        "in scripts/anyio_1111_workaround.py.",
        ver,
        _FIXED_IN,
        _FIXED_IN,
    )


_apply_anyio_1111_workaround()
del _apply_anyio_1111_workaround
