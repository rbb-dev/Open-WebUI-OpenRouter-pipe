"""Every failure this project chose to swallow must still announce itself once.

A commit series went through the package surfacing failures that were previously
discarded in silence. The diagnostics it added were almost entirely unasserted: the
whole warning block could be deleted from each site and the suite stayed green, so the
guarantee was "someone wrote a logger.warning here", not "the operator finds out".

The two properties that matter are the ones each site was written to satisfy:
announced at least once, and no more than once for the same cause. A latch that never
lets the warning fire and a warning with no latch are both defects, and a source scan
for `logger.warning` near an `except` cannot tell either of them from correct code.

Asserted on records rather than message text throughout -- pinning the prose makes a
test that goes red on a reword and still cannot see a broken latch.
"""

from __future__ import annotations

import logging

import pytest

from open_webui_openrouter_pipe.core.warn_latch import warn_level


def _records(caplog, logger_name: str) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == logger_name
    ]


def _latched_import_sites() -> tuple[set[str], list[str]]:
    """Every cause key on `_warned_import_sites`, read out of the module's own source.

    Discovered rather than listed, because the list is what failed: this class was
    written for four sites against five, named two of them in its parametrize, and the
    two it never drove could have their latch inverted with the whole suite green.

    Shared with the sibling censuses so the accepted cause shapes cannot drift apart
    again: this one matched only string constants while the house idiom is an f-string,
    so a seam keyed `f"room_members:{type(exc).__name__}"` was invisible to it.
    """
    from open_webui_openrouter_pipe.plugins.pipe_dashboard import dashboard_socket

    from tests.warn_latch_census import warn_level_causes

    return warn_level_causes(dashboard_socket, "_warned_import_sites")


_SEAM_DRIVERS = {
    "events": lambda m: m.register_valve_event_sink(),
    "register": lambda m: m.register_socket_handler(),
    "viewer_sids": lambda m: m.local_viewer_sids(),
    "emit": lambda m: m.emit_dashboard({"tick": 0}),
    "reauth": lambda m: m.reauthorize_local_viewers(),
}


class TestDashboardSocketImportGuards:
    """Every optional-import site, all latched on one shared set.

    Scoped to this class rather than the module: the --no-plugins artifacts omit
    pipe_dashboard by design, but the imaging diagnostics below are package code and
    must still run there. A module-level importorskip would silently drop both.

    The latch condition is `if "<key>" not in _warned_import_sites`. Inverting it to
    `in` leaves the code reading exactly as before while the warning can never fire --
    the failure becomes silent again and nothing in the suite noticed.
    """

    @pytest.fixture(autouse=True)
    def _requires_the_dashboard_plugin(self):
        pytest.importorskip(
            "open_webui_openrouter_pipe.plugins.pipe_dashboard",
            reason="the --no-plugins artifacts omit pipe_dashboard by design",
        )

    def test_every_latched_import_site_has_a_driver(self):
        """Forgetting to drive a new seam is a red build, not silence.

        The driver table still has to be extended by hand -- only a human knows how to
        reach each seam -- but this turns "nobody added one" into a failure. Asserting a
        count instead would tell the author to bump a number, which is the same
        non-coverage with an extra step.
        """
        from tests.warn_latch_census import UNRESOLVABLE_MESSAGE

        discovered, unresolvable = _latched_import_sites()
        assert not unresolvable, UNRESOLVABLE_MESSAGE + "\n  ".join(unresolvable)
        missing = sorted(discovered - set(_SEAM_DRIVERS))
        assert not missing, (
            f"these latched import sites have no driver, so inverting their latch is "
            f"invisible: {missing}. Add a driver rather than narrowing the sweep."
        )
        stale = sorted(set(_SEAM_DRIVERS) - discovered)
        assert not stale, f"the driver table names sites that no longer exist: {stale}"

    @pytest.mark.parametrize("key", sorted(_SEAM_DRIVERS))
    @pytest.mark.asyncio
    async def test_an_unavailable_open_webui_seam_is_reported_once(
        self, key, caplog, monkeypatch
    ):
        call = _SEAM_DRIVERS[key]
        import builtins

        from open_webui_openrouter_pipe.plugins.pipe_dashboard import dashboard_socket

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name.startswith("open_webui.events") or name.startswith("open_webui.socket"):
                raise ImportError(f"{name} unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        dashboard_socket._warned_import_sites.clear()

        try:
            with caplog.at_level(logging.WARNING, logger=dashboard_socket.logger.name):
                for _ in range(4):
                    result = call(dashboard_socket)
                    if hasattr(result, "__await__"):
                        await result
                emitted = _records(caplog, dashboard_socket.logger.name)
                armed = sorted(dashboard_socket._warned_import_sites)
        finally:
            dashboard_socket._warned_import_sites.clear()

        assert key in armed, (
            f"driving the {key!r} seam four times with its import blocked never armed "
            "its latch, so the diagnostic it guards was not reached at all"
        )
        assert len(emitted) == len(armed), (
            f"the {key!r} seam was unavailable four times; {len(armed)} distinct causes "
            f"were reported ({armed}) but {len(emitted)} warnings were emitted. Fewer "
            "means a latch is inverted and the failure is silent; more means a latch is "
            "not holding and the operator gets one line per occurrence. Counted against "
            "the armed causes rather than against 1 because some of these seams call "
            "each other -- register_socket_handler drives the events seam first."
        )
        assert all(r.exc_info is not None for r in emitted), (
            f"a {key!r} warning carries no traceback, so it reports that a seam is "
            "missing without saying why"
        )


class TestMissingImagingLibraries:
    """cairosvg and Pillow are optional; their absence changes what the pipe can do."""

    @pytest.mark.parametrize(
        ("library", "blocked", "mime", "payload"),
        [
            (
                "cairosvg",
                "cairosvg",
                "image/svg+xml",
                b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>',
            ),
            ("pillow", "PIL", "image/webp", b"RIFF\x00\x00\x00\x00WEBPVP8 "),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_missing_imaging_library_is_reported_once(
        self, library, blocked, mime, payload, caplog, monkeypatch
    ):
        """Driven through the real fetch path, with the library made unimportable."""
        import builtins

        import aiohttp
        from aioresponses import aioresponses

        from open_webui_openrouter_pipe import Pipe

        pipe = Pipe()
        handler = pipe._multimodal_handler
        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == blocked or name.startswith(blocked + "."):
                raise ImportError(f"{name} unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        handler._warned_missing_imaging.clear()

        url = f"https://example.test/icon-{library}"
        try:
            with caplog.at_level(logging.WARNING, logger=handler.logger.name):
                async with aiohttp.ClientSession() as session:
                    with aioresponses() as http:
                        for _ in range(4):
                            http.get(url, body=payload, headers={"Content-Type": mime})
                        for _ in range(4):
                            assert (
                                await handler._fetch_image_as_data_url(session, url)
                                is None
                            ), (
                                f"the conversion succeeded with {blocked} unimportable, "
                                "so this test never reached the diagnostic"
                            )
                emitted = _records(caplog, handler.logger.name)
        finally:
            handler._warned_missing_imaging.clear()
            await pipe.close()

        assert len(emitted) == 1, (
            f"{library} was missing across four conversions and produced "
            f"{len(emitted)} warnings; the operator either never learns the library is "
            "absent, or learns it once per image"
        )


def _silent_latch_offenders(source: str, label: str) -> list[str]:
    """Every `if <latch>: warning(...)` in *source* with no `else: debug(...)`.

    Extracted so the matcher itself can be driven against known-offending and
    known-compliant text. The correct population on the real tree is zero and will stay
    zero, so the tree scan alone can never show that this matcher still matches
    anything -- narrowing the regex to nonsense left the suite green with a genuine
    offender sitting in the package.
    """
    import ast
    import re

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if not re.search(r"_warned|_warn_once|_last_warn|_warn_ts", test):
            continue
        body = ast.unparse(ast.Module(node.body, []))
        if not re.search(r"\.(warning|error|exception)\(", body):
            continue
        repeats = bool(node.orelse) and ".debug(" in ast.unparse(ast.Module(node.orelse, []))
        if not repeats:
            offenders.append(f"{label}:{node.lineno}  if {test[:60]}")
    return offenders


_SILENT_LATCH_OFFENDER = """
if key not in _warned_sites:
    _warned_sites.add(key)
    logger.warning("something failed")
"""

_SILENT_LATCH_COMPLIANT = """
if key not in _warned_sites:
    _warned_sites.add(key)
    logger.warning("something failed")
else:
    logger.debug("something failed again")
"""


def test_the_silent_latch_matcher_can_still_match():
    """The tree scan below is green because the tree is clean, not because it looks.

    Its regex was narrowed to a name that appears nowhere and the whole suite stayed
    green with a real offender injected into the package. A count floor cannot fix that
    -- zero is the correct answer on this tree -- so the matcher is driven against
    literal text instead, one offending and one compliant.
    """
    assert len(_silent_latch_offenders(_SILENT_LATCH_OFFENDER, "probe")) == 1, (
        "the matcher no longer recognises a latched warning with no DEBUG repeat, so "
        "the tree scan below reports green by matching nothing"
    )
    assert _silent_latch_offenders(_SILENT_LATCH_COMPLIANT, "probe") == [], (
        "the matcher flags a diagnostic that DOES repeat at DEBUG; the tree scan would "
        "fail on correct code"
    )


def test_no_diagnostic_goes_silent_at_every_level():
    """Every latched warning still repeats at DEBUG. Discovered, not enumerated.

    Sixteen sites armed a latch, emitted one WARNING and then said nothing at ANY level
    for the life of the worker -- so an operator who raises the log level to diagnose a
    recurring fault sees strictly less than before. The rule was written down in one
    module and violated everywhere the helper could not be imported from.

    This walks the package for the shape rather than listing the sites, because a list
    is exactly what failed: the sites that had the repeat path and the sites that did
    not were written by the same hand, weeks apart, and nothing compared them.

    An `if <latch>: warning(...)` with no `else: debug(...)` is the defect. Going through
    `warn_level` is the fix, and it satisfies this by construction because the level it
    returns is DEBUG on a repeat.
    """
    import pathlib

    roots = [
        pathlib.Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe",
        pathlib.Path(__file__).resolve().parents[1] / "scripts",
    ]
    offenders = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
            except OSError:
                continue
            offenders += _silent_latch_offenders(source, path.name)

    assert not offenders, (
        "these diagnostics arm a latch, warn once, and then emit nothing at any level "
        "for the life of the worker -- raising the log level to DEBUG shows the "
        f"operator strictly less than before:\n  " + "\n  ".join(offenders)
    )


class TestWarnLevel:
    """The shared decision itself, by returned level rather than by shape.

    The sweep above is a census: it proves no site hand-rolls a latch without a repeat
    path. It cannot prove the helper those sites delegate to is right -- making
    `warn_level` return NOTSET on a repeat passes every structural check while emitting
    nothing at any level, which is the original defect wearing the fix's clothes.
    """

    def test_the_first_occurrence_of_a_cause_warns(self):
        assert warn_level(set(), "boom") == logging.WARNING

    def test_a_repeat_of_the_same_cause_drops_to_debug(self):
        latch: set[str] = set()
        warn_level(latch, "boom")
        assert warn_level(latch, "boom") == logging.DEBUG, (
            "a recurring fault emitted at a level other than DEBUG; NOTSET in "
            "particular is silent at every level, which is the defect this replaced"
        )

    def test_a_different_cause_warns_again(self):
        latch: set[str] = set()
        warn_level(latch, "boom")
        assert warn_level(latch, "other") == logging.WARNING, (
            "the latch is global rather than per-cause, so the second distinct failure "
            "an operator needs to know about is silenced by the first"
        )

    def test_a_cooldown_latch_warns_then_debugs_within_the_window(self):
        latch: dict[str, float] = {}
        assert warn_level(latch, "boom", cooldown_s=300) == logging.WARNING
        assert warn_level(latch, "boom", cooldown_s=300) == logging.DEBUG

    def test_a_cooldown_latch_warns_again_once_the_window_passes(self):
        latch: dict[str, float] = {}
        assert warn_level(latch, "boom", cooldown_s=0) == logging.WARNING
        assert warn_level(latch, "boom", cooldown_s=0) == logging.WARNING, (
            "an elapsed cooldown did not re-warn, so the dict keying degenerates into "
            "the permanent latch it exists to avoid"
        )

    def test_a_dict_latch_without_a_cooldown_is_permanent(self):
        latch: dict[str, float] = {}
        assert warn_level(latch, "boom") == logging.WARNING
        assert warn_level(latch, "boom") == logging.DEBUG


# Every conditional inside an `except` handler that gates a warning-or-above call, with
# the reason it is not a warn-once latch. Keyed by the condition's SOURCE, not its line,
# so moving code needs no edit here but changing the condition does.
_LOG_LEVEL_NAMES = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"}

_NOT_A_WARN_LATCH = {
    ("responses_adapter.py", "is_auth_failure"): (
        "chooses WHAT to say -- the message for an auth error vs a generic one -- not "
        "whether it has said it before"
    ),
    ("pipe.py", "not available_models"): (
        "nothing to report when the catalog came back empty; not a repeat of a report "
        "already made"
    ),
    ("usage_store.py", "self._dropped % _US_DROP_WARN_EVERY == 1"): "every-Nth sampling",
    ("persistence.py", "self._maybe_heal_index_conflict(engine, table, exc)"): (
        "retry branch: drops orphaned indexes and reports whether it healed"
    ),
    ("persistence.py", "reason != failure_reason"): "re-announces when the failure MODE changes",
    ("persistence.py", "flush_reason != last_flush_failure"): "re-announces when the failure MODE changes",
}

# The same rule for the EXPRESSION form -- `level if seen else other` inside a .log()
# call -- keyed by the file and the unparsed conditional.
_NOT_A_WARN_LATCH_EXPR = {
    (
        "anyio_1111_workaround.py",
        "_logging.DEBUG if _seen else _logging.WARNING",
    ): (
        "the bundle head: bundle_v2's validator rejects an absolute internal import "
        "surviving there, so warn_level cannot be called from this file"
    ),
}


def test_no_module_reimplements_the_warn_once_decision():
    """One implementation, found by ELIMINATION rather than by shape or by name.

    Three censuses have now been written for this. The first matched identifier
    substrings and four modules kept their own copy through it. The second matched three
    syntactic shapes -- a membership test, a clock comparison, a truth-read -- and a
    conditional spelled as a plain call escaped all three: `if self._should_warn("x")`
    is what someone reaches for the moment they are told not to inline the latch
    expression, and it is invisible to every one of them.

    So this flags EVERY conditional inside an `except` handler that gates a
    warning-or-above call, and carries a closed list of the ones that are a different
    rule. Going through `warn_level` produces no such conditional at all: the level is a
    value and the log call is unconditional. A new inline latch cannot be spelled around
    this, because the check no longer asks what the condition looks like.

    The exemptions are checked in both directions -- one that stops matching fails as
    loudly as a new offender, so the list cannot rot into blanket permission.
    """
    import ast
    import pathlib
    import re

    # The same roots as the name census above. `scripts/anyio_1111_workaround.py` ships
    # inside the head of all four bundles, so it is production code -- and it is where a
    # latch hid, because the name census walked here but matched only four identifier
    # spellings, and the shape census had the right shapes but the narrower scope.
    roots = [
        pathlib.Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe",
        pathlib.Path(__file__).resolve().parents[1] / "scripts",
    ]
    offenders = []
    seen_exemptions = set()
    seen_expr_exemptions = set()
    unparsed = []
    for path in sorted(p for root in roots for p in root.rglob("*.py")):
        if path.name == "warn_latch.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            unparsed.append(path.name)
            continue
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            for node in ast.walk(ast.Module(handler.body, [])):
                if not isinstance(node, ast.If):
                    continue
                if not re.search(
                    r"\.(warning|error|exception)\(",
                    ast.unparse(ast.Module(node.body, [])),
                ):
                    continue
                key = (path.name, ast.unparse(node.test))
                if key in _NOT_A_WARN_LATCH:
                    seen_exemptions.add(key)
                    continue
                offenders.append(f"{path.name}:{node.lineno}  if {key[1][:70]}")
            # The expression form. `logging.DEBUG if seen else logging.WARNING` passed
            # to `.log()` is the same decision with no `ast.If` to find, and it is what
            # someone writes the moment a statement-level guard is disallowed.
            for node in ast.walk(ast.Module(handler.body, [])):
                if not isinstance(node, ast.IfExp):
                    continue
                branches = {ast.unparse(node.body), ast.unparse(node.orelse)}
                if not all(
                    b.rsplit(".", 1)[-1] in _LOG_LEVEL_NAMES for b in branches
                ) or len(branches) < 2:
                    continue
                key = (path.name, ast.unparse(node))
                if key in _NOT_A_WARN_LATCH_EXPR:
                    seen_expr_exemptions.add(key)
                    continue
                offenders.append(f"{path.name}:{node.lineno}  {key[1][:70]}")

    assert not unparsed, (
        "these files did not parse, so this census skipped them entirely and its silence "
        f"means nothing: {unparsed}"
    )
    stale = sorted(k for k in _NOT_A_WARN_LATCH if k not in seen_exemptions)
    stale += sorted(k for k in _NOT_A_WARN_LATCH_EXPR if k not in seen_expr_exemptions)
    assert not stale, (
        "these exemptions no longer match anything in the tree, so the list is granting "
        f"permission nobody asked for and hiding whatever replaced them: {stale}"
    )
    assert not offenders, (
        "these except-handlers decide for themselves whether to warn, instead of asking "
        "warn_level for a level and logging unconditionally. Each is a copy of the rule "
        "that goes stale on its own:\n  " + "\n  ".join(offenders)
    )


# Named, not counted: a literal 3 tells the author to write 4 when a census stops being
# driven, which is the non-coverage this file rejects everywhere else.
_SOURCE_CENSUSES = {
    "test_no_diagnostic_goes_silent_at_every_level",
    "test_no_module_reimplements_the_warn_once_decision",
    "scan_broad_suppressions",
}


def test_every_source_census_walks_the_same_roots(monkeypatch):
    """Observed by RECORDING the walk, not by reading the functions' source.

    The name census walks the package AND `scripts/`; the shape census that superseded
    it walked the package only. A warn-once latch in `scripts/anyio_1111_workaround.py`
    -- which ships inside the head of all four bundles -- landed in the gap.

    The previous version of this guard collected directory-name STRING LITERALS out of
    each function's source. That is what the text says, not what the code reads: adding
    `if root.name == "scripts": continue` inside the loop, or slicing `roots[:1]`, leaves
    every literal in place and passes. It also went red on a pure refactor -- hoisting
    the roots to a module constant emptied the extraction with no behaviour change, which
    is why the third census needed a hand-written special case.

    Wrapping `Path.read_text` records the FILES each census actually consumes, so a
    narrowing fails however it is spelled and a hoist is invisible. Recording `rglob`
    instead pinned only the call: a census could sweep both roots and then drop every
    `scripts/` file inside its own loop, and the recorder saw both roots and passed.
    """
    import pathlib

    import tests.test_silent_suppression_inventory as inventory

    walked: dict[str, set[str]] = {}
    real_read_text = pathlib.Path.read_text
    current: list[str] = []

    def _recording_read_text(self, *a, **kw):
        if current:
            walked.setdefault(current[0], set()).add(str(self.parent))
        return real_read_text(self, *a, **kw)

    monkeypatch.setattr(pathlib.Path, "read_text", _recording_read_text)

    for fn in (
        test_no_diagnostic_goes_silent_at_every_level,
        test_no_module_reimplements_the_warn_once_decision,
        inventory.scan_broad_suppressions,
    ):
        current[:] = [fn.__name__]
        fn()
    current.clear()

    assert set(walked) == _SOURCE_CENSUSES, (
        "the set of censuses this drives changed. A count told the author to bump a "
        f"number; this names them.\n  only driven: {sorted(set(walked) - _SOURCE_CENSUSES)}"
        f"\n  only pinned: {sorted(_SOURCE_CENSUSES - set(walked))}"
    )
    distinct = {frozenset(v) for v in walked.values()}
    assert len(distinct) == 1, (
        "the censuses read different directories: "
        + f"{ {k: sorted(v) for k, v in walked.items()} }. A latch in a directory only "
        "one of them visits is guarded by neither, which is exactly how the bundle head "
        "came to hold one."
    )


