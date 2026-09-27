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
    "core/logging_system.py": 2,
    # 7th: the cost snapshot, now one guarded helper reached from all three exits. A job
    # OpenRouter has already billed for is recorded whatever the pipe does with the
    # bytes, and a storage error while recording it must not replace the failure the
    # user is being shown -- nor take down a cancellation that is already unwinding.
    "integrations/video.py": 7,
    # 18th: the diagnostic, reassembled below. Measured from the finished tree, not
    # derived: B7-3's stamp restore dropped its `suppress` for a `try`/`except` that
    # reports a failed restore, and the stranded-turn capture now routes its row
    # delete through `_release_assembly_lock`, which is itself one guarded helper.
    # The success path keeps its own inline suppression, and the capture adds none.
    "logging/session_log_manager.py": 18,
    "media/frame_extraction.py": 2,
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
    # Two more: the `OWUI_CHAT_ID.reset(token)` calls in the `finally` blocks that undo the
    # channel-chat-id token; resetting a ContextVar is bookkeeping for the next request, and letting
    # one out would replace the turn's real outcome with an error raised while unwinding it.
    "pipe.py": 27,
    "storage/persistence.py": 3,
    # 1st: the caller-supplied fallback in `_emit_templated_error_event`. It is reached only because the
    # admin's own template already failed to render, and the generic card below it is the answer if the
    # fallback fails too -- letting it out would replace the failure being reported with a template error.
    "streaming/event_emitter.py": 1,
    # 1st: the roster task teardown above, whose only failure mode is a second
    # cancellation arriving while the loop is already unwinding. 2nd: awaiting that task
    # after the hand-back has cancelled it, where the await exists to finalise it and a
    # failure to do so changes nothing the caller can act on -- the terminal writes it
    # feeds are all gated off this path anyway.
    # 3rd: the loop-limit note's write of `{"error": {"content": …}}` to the saved chat row, so the
    # banner survives a reload the way Open WebUI's own `emit_message_error` makes it survive. It
    # runs after the notification has already reached the caller, so a storage failure costs the
    # persisted copy alone and must not replace the note the person is being shown.
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
