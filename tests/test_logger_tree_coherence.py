"""Every package logger must sit under the root that Pipe wires its handlers onto.

Pipe attaches the session-log buffer, request-id enrichment and the debug archive to
``SessionLogger.get_logger(__name__.split(".")[0])``. A logger created with a literal
name instead sits in whatever tree that literal names -- which is the same tree in
package mode, and a *different* one in the flat bundle, where every module collapses
into the host module (``function_<id>`` under Open WebUI).

The consequence is silent: those records simply stop reaching the sinks, only in the
bundle that ships by default, and no assertion anywhere notices.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from open_webui_openrouter_pipe import Pipe

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"
@pytest.mark.asyncio
async def test_package_loggers_reach_the_wired_root():
    """Every Logger the package actually holds must sit under the wired root.

    Inspects the loaded module objects, not logging's global registry: the registry
    also contains loggers other tests created by name, which in a bundle look like
    orphans because the modules they are named after do not exist there. An earlier
    version compared `cfg.LOGGER.name` against `pipe.logger.name`, both built by the
    same expression, so in package mode -- the only mode it ran in -- it could not fail.
    """
    import logging
    import sys

    pipe = Pipe()
    try:
        wired = pipe.logger.name
        root_module = sys.modules.get(wired) or sys.modules.get(PACKAGE_DIR.name)
        assert root_module is not None, f"neither {wired!r} nor the package is loaded"

        held: list[tuple[str, str]] = []
        for mod_name, module in list(sys.modules.items()):
            if module is None:
                continue
            if not (mod_name == wired or mod_name.startswith(wired + ".") or module is root_module):
                continue
            for attr, value in list(vars(module).items()):
                if isinstance(value, logging.Logger):
                    held.append((f"{mod_name}.{attr}", value.name))

        assert len(held) >= 15, (
            f"only {len(held)} Logger objects are held by loaded package modules "
            f"({held}); this test is asserting almost nothing."
        )
        outside = sorted(
            f"{where} -> {name}"
            for where, name in held
            if not (name == wired or name.startswith(wired + "."))
        )
        assert not outside, (
            f"these Loggers sit outside the wired root {wired!r}: {outside}. Their "
            "records never reach the session log, the debug archive, or request-id "
            "enrichment."
        )
    finally:
        await pipe.close()


def test_the_bundle_head_resolves_its_logger_root_at_call_time(monkeypatch):
    """``scripts/anyio_1111_workaround.py`` ships INSIDE the bundle head.

    It is the artifact's first executable statement, and it runs before any pipe
    module is imported, so it cannot derive its root from an import. It resolves the
    package's own ``__name__`` out of ``sys.modules`` instead -- which matters because
    the flat bundle ALIASES the package key onto the host module, making presence alone
    meaningless.

    Asserted by running it and reading the logger it built. The previous version of
    this test read the source and accepted the argument if its text contained
    ``__name__``, ``sys.modules`` or the substring ``root``; naming a local variable
    ``root`` and assigning it a hardcoded string satisfied all three while doing
    exactly what the test existed to forbid.

    Safe to invoke: the workaround returns as soon as it sees pytest in
    ``sys.modules``, after building the logger and before patching anything.
    """
    import importlib.util
    import logging
    import sys
    import types

    embedded = PACKAGE_DIR.parent / "scripts" / "anyio_1111_workaround.py"
    assert embedded.exists(), "the embedded workaround moved; update this guard"

    requested: list[str] = []
    real_get_logger = logging.getLogger

    def _recording(name=None):
        if name is not None:
            requested.append(name)
        return real_get_logger(name) if name is not None else real_get_logger()

    monkeypatch.setattr(logging, "getLogger", _recording)

    host = types.ModuleType("function_deadbeef")
    monkeypatch.setitem(sys.modules, "open_webui_openrouter_pipe", host)

    spec = importlib.util.spec_from_file_location("_anyio_wa_under_test", embedded)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert requested, "the bundle head built no logger at all"
    assert requested[0] == "function_deadbeef.anyio_1111_workaround", (
        f"the bundle head logged to {requested[0]!r} while the package resolved to "
        "'function_deadbeef'. In the flat bundle that is a different tree from the one "
        "Pipe wires its handlers onto, so these records reach no sink."
    )


def test_each_module_logs_under_its_own_name():
    """A record's `name` must identify the module that emitted it.

    `core/config.py` used to export a shared logger as `LOGGER`, built not from its own
    `__name__` but from the PACKAGE ROOT: `logging.getLogger("open_webui_openrouter_pipe")`.
    Five modules imported it -- api/transforms, requests/transformer, core/context_budget,
    tools/tool_schema and tools/tool_registry. Every record from those, including every
    artifact-replay and missing-artifact warning from `transform_messages_to_input`,
    carried that bare package name. In package mode it is the same Logger object
    `Pipe.logger` uses, so those records were not misfiled under the config module -- they
    were indistinguishable from the pipe's own top-level records, and an operator reading
    a support archive for a request-shaping fault had nothing to narrow it down with. In
    the flat bundle it is worse: `Pipe` wires its handlers onto the HOST module's name, so
    a hardcoded `open_webui_openrouter_pipe` is a different tree with no handlers and
    those records never reached the archive at all.

    Inspects the Logger objects modules actually hold, so importing another module's
    logger under any local name is caught. Its own name or a child of it is allowed:
    `plugins/registry.py` builds one child logger per plugin id, which identifies the
    emitter more precisely rather than less.
    """
    import logging
    import sys

    from open_webui_openrouter_pipe import Pipe

    pipe = Pipe()
    try:
        root = pipe.logger.name
        checked: list[str] = []
        wrong: list[str] = []
        for mod_name, module in list(sys.modules.items()):
            if not (mod_name == root or mod_name.startswith(root + ".")):
                continue
            for attr, value in list(vars(module).items()):
                if not isinstance(value, logging.Logger):
                    continue
                checked.append(f"{mod_name}.{attr}")
                if not (value.name == mod_name or value.name.startswith(mod_name + ".")):
                    wrong.append(f"{mod_name}.{attr} -> {value.name}")

        # The flat bundle collapses every module into one namespace, so the three
        # distinct variable names are all there is to find there. The attribution
        # assertion below still runs and still means something; only the census floor
        # is package-shaped.
        floor = 3 if os.environ.get("OWUI_PIPE_BUNDLE_PATH") else 15
        assert len(checked) >= floor, (
            f"only {len(checked)} Logger objects were found on loaded package modules "
            f"({checked}); this test is asserting almost nothing"
        )
        assert not wrong, (
            "these modules log under a name that is not their own, so their records "
            f"are attributed to a different module in the session archive: {wrong}"
        )
    finally:
        import asyncio

        asyncio.run(pipe.close())


def test_every_module_logger_is_derived_from_its_own_name():
    """One convention, so a reader cannot pick the wrong one.

    Two modules used to build their logger from `__name__.split(".")[0]` plus a
    hardcoded path. In package and compressed mode that produced the same name as
    `__name__`; in the flat bundle, where every module collapses into the host module,
    it produced a different one -- so whether a record identified its module depended
    on which of two forms the module happened to use, and on which artifact was
    installed.

    The rule now is that a logger's name is `__name__` or a child of it. A child still
    survives the flat bundle as a distinguishable name, so a genuine sub-component
    keeps its identity without a second convention.
    """
    import ast

    from tests.package_sources import parsed_sources

    offenders: list[str] = []
    for path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "getLogger"
                and node.args
            ):
                continue
            expr = ast.unparse(node.args[0])
            if expr == "__name__" or expr.startswith(("f'{__name__}", 'f"{__name__}')):
                continue
            if expr == "name":
                # SessionLogger.get_logger's own body; the name comes from its caller
                continue
            offenders.append(f"{path.relative_to(PACKAGE_DIR)}: getLogger({expr})")

    assert not offenders, (
        "these loggers are not named after the module that holds them, so whether "
        "their records identify their origin depends on the build shape:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.timeout(600)
@pytest.mark.skipif(
    not (PACKAGE_DIR.parent / "open_webui_openrouter_pipe_bundled.py").exists(),
    reason="the flat bundle is not built; this asserts a property of that artifact",
)
def test_the_flat_bundle_proxy_does_not_shadow_a_stdlib_attribute():
    """`logging.debug` must be the stdlib function, not a module.

    The flat bundle rebinds the global name `logging` to a `_PackageProxy` so that both
    `logging.getLogger(...)` and `import ..._pipe.logging.session_log_manager` resolve.
    Its `__getattr__` consulted `_SUBMODULE_ATTRS` first, and `requests/debug.py` puts
    `debug` in that set -- so `logging.debug` returned the flat module. Any
    `logging.debug(...)` convenience call added to the package would be
    `TypeError: 'module' object is not callable` in the shipped artifact and nowhere else.

    Run in a SUBPROCESS. Loading the artifact executes its module body, which installs
    package aliases and rewrites sys.modules for the whole interpreter -- doing that
    in-process poisons every test that runs afterwards, and popping the probe's own
    module name afterwards undoes none of it.

    Carries its own timeout because it can be the first caller of the shared probe and
    then pays the whole load, which is ~4s against the suite-wide 60s the CI run passes
    on the command line. The same marker is on the other consumer, because which of the
    two runs first decides which one pays, and the suite's verdict must not depend on
    collection order.
    """
    from bundle_probe import probe

    artifact = PACKAGE_DIR.parent / "open_webui_openrouter_pipe_bundled.py"
    result = probe(artifact)
    assert result["ok"], f"the flat bundle did not load -- {result['error']}"
    data = result["proxy"]

    assert data["has_proxy"], "the flat bundle has no global named 'logging'"
    assert not data["is_stdlib"], (
        "the global named 'logging' IS the stdlib module, so the bundler stopped "
        "wrapping the shadowed package. Every assertion below then inspects the stdlib "
        "and proves nothing about the artifact -- `is not None` cannot tell the two "
        "apart, because the unwrapped case is a module too."
    )
    assert set(data["collisions"]) == {"config", "debug"}, (
        "the package component names that shadow a stdlib logging attribute are "
        f"{sorted(data['collisions'])}, not {{'config', 'debug'}}. core/config.py and "
        "requests/debug.py are the two; if that changed, re-pin it deliberately. A "
        "non-empty check passed on one name for as long as the compared set was "
        "decided by whatever the probe process happened to import."
    )
    assert not data["wrong"], (
        "these stdlib logging attributes resolve to something else in the flat bundle: "
        f"{data['wrong']}. Calling one raises TypeError in the shipped artifact and in "
        "no other build shape."
    )
