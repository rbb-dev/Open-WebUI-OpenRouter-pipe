"""The version of every gating tool must be the same everywhere it is stated.

ruff has `required-version`, which refuses to run under a mismatched version. pyright
has no equivalent, so the pin lives in three files that can drift apart silently: a
developer following the documented setup gets whatever `pip install pyright` yields
today, and can be red locally while CI is green, or the reverse.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = tuple(
    sorted(
        f".github/workflows/{p.name}"
        for p in (PROJECT_ROOT / ".github" / "workflows").glob("*.yml")
        if re.search(r"\b(?:ruff|pyright)==", p.read_text(encoding="utf-8"))
    )
)


def _pinned(text: str, tool: str) -> set[str]:
    return set(re.findall(rf"\b{tool}==([0-9][0-9.]*)", text))


def test_every_workflow_pins_the_same_ruff_and_pyright():
    found: dict[str, dict[str, set[str]]] = {}
    for rel in WORKFLOWS:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        found[rel] = {"ruff": _pinned(text, "ruff"), "pyright": _pinned(text, "pyright")}

    assert WORKFLOWS, "no workflow pins any tool version; the sweep found nothing"
    for tool in ("ruff", "pyright"):
        versions = {v for f in found.values() for v in f[tool]}
        assert len(versions) == 1, (
            f"{tool} is pinned to {sorted(versions) or 'nothing'} across the workflows. "
            f"Per file: { {k: sorted(v[tool]) for k, v in found.items()} }. Two gates "
            "running different versions of the same checker report different findings."
        )


def test_the_ruff_pin_matches_the_config_that_enforces_it():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    required = re.search(r'required-version\s*=\s*"([^"]+)"', pyproject)
    assert required, "pyproject no longer sets [tool.ruff] required-version"

    workflow = (PROJECT_ROOT / WORKFLOWS[0]).read_text(encoding="utf-8")
    pinned = _pinned(workflow, "ruff")
    assert pinned == {required.group(1).lstrip("=")}, (
        f"required-version is {required.group(1)!r} but the workflow installs "
        f"{sorted(pinned)}. ruff would refuse to run, so the lint gate never executes."
    )


def test_the_repro_scripts_pin_what_ci_pins():
    """A repro env that lints with a different version reports different findings."""
    workflow = (PROJECT_ROOT / WORKFLOWS[0]).read_text(encoding="utf-8")
    expected = {tool: _pinned(workflow, tool) for tool in ("ruff", "pyright")}

    for rel in ("scripts/repro_venv.sh", "scripts/repro_venv_uv.sh"):
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for tool, versions in expected.items():
            assert _pinned(text, tool) == versions, (
                f"{rel} pins {tool}=={sorted(_pinned(text, tool)) or 'nothing'} but CI "
                f"pins {sorted(versions)}. The documented reproduction environment is "
                "not the environment the gate runs in."
            )


def test_pyright_enforces_the_language_version_ci_enforces():
    """Asserts pyright's own verdict, because stating the version is not setting it.

    The pin previously sat in `[tool.pyright]` in pyproject.toml, where pyright never
    read it: that table is consulted ONLY when no pyrightconfig.json exists, and this
    repo has one. `pyright --verbose` reported "Assuming Python version 3.12.2" -- the
    local interpreter -- while CI ran 3.11, so post-3.11 syntax type-checked clean
    locally and failed the build. Every text-level check agreed the whole time.

    So this runs pyright on a construct the pinned version must reject. Reading the
    setting out of the config would restate the bug: the question is not what the file
    says, it is what the checker does.
    """
    import json
    import shutil
    import subprocess
    import tempfile

    if shutil.which("pyright") is None:
        import pytest

        pytest.skip("pyright is not installed in this environment")

    config = json.loads(
        re.sub(
            r",(\s*[}\]])",
            r"\1",
            re.sub(r"^\s*//.*$", "", (PROJECT_ROOT / "pyrightconfig.json").read_text(encoding="utf-8"), flags=re.M),
        )
    )
    pinned = config.get("pythonVersion")
    assert pinned == "3.11", (
        f"pyrightconfig.json sets pythonVersion={pinned!r}. It must match the "
        "python-version the workflows run and the requires-python floor."
    )

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.py"
        probe.write_text("type _Probe = int\n", encoding="utf-8")
        result = subprocess.run(
            ["pyright", "--outputjson", str(probe)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    try:
        diagnostics = json.loads(result.stdout).get("generalDiagnostics", [])
    except json.JSONDecodeError:  # pragma: no cover - pyright failed to run at all
        raise AssertionError(f"pyright produced no JSON: {result.stdout[-400:]}") from None

    errors = [d for d in diagnostics if d.get("severity") == "error"]
    assert errors, (
        "pyright accepted a PEP 695 `type` statement, which requires Python 3.12. It is "
        "therefore checking at 3.12 or later while the workflows check at "
        f"{pinned}. Anything newer than {pinned} passes here and fails the build."
    )
    assert any("3.12" in d.get("message", "") for d in errors), (
        f"pyright errored on the probe for an unexpected reason: "
        f"{[d.get('message') for d in errors]}"
    )


def _addopts_plugins() -> set[str]:
    """Every module pytest.ini loads with -p."""
    import re

    ini = (PROJECT_ROOT / "pytest.ini").read_text(encoding="utf-8")
    line = next(l for l in ini.splitlines() if l.startswith("addopts"))
    return {m.group(1) for m in re.finditer(r"-p\s+([A-Za-z0-9_.]+)", line)}


def test_every_plugin_pytest_ini_loads_is_importable():
    """`-p <missing module>` aborts pytest at startup, before collection.

    `pytest_bootstrap` sets PYTEST_DISABLE_PLUGIN_AUTOLOAD, so `-p` is the only way a
    plugin loads here -- each entry is load-bearing, not belt-and-braces.
    """
    import importlib.util

    missing = sorted(
        name
        for name in _addopts_plugins()
        if not name.startswith("open_webui_openrouter_pipe")
        and importlib.util.find_spec(name) is None
    )
    assert not missing, (
        f"pytest.ini loads {missing} with -p, and they are not importable. pytest "
        "aborts at startup, so zero tests run."
    )


def test_every_third_party_plugin_is_declared_in_the_test_extra():
    """Provisioning is derived from one declaration, not maintained in four places.

    `-p pytest_timeout` was added to support a per-test timeout marker without touching
    provisioning, and nothing installed it: `repro_venv.sh`, `repro_venv_uv.sh` and
    `docs/testing_bootstrap_and_operations.md` all predated it. A contributor following
    the repo's own documented reproduction got a venv in which pytest could not start at
    all, while the pin guard above passed because ruff and pyright were still correct.

    Checked as a mapping from `-p` entries to the extra, so the NEXT entry is covered
    too -- listing pytest-timeout by name would close this case and reopen on the next.
    """
    import re

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^test = \[(.*?)^\]", pyproject, re.M | re.S)
    assert block, "pyproject.toml has no [project.optional-dependencies] test extra"
    declared = {
        m.group(1).lower() for m in re.finditer(r'"([A-Za-z0-9_.-]+)"', block.group(1))
    }

    undeclared = sorted(
        name
        for name in _addopts_plugins()
        if not name.startswith("open_webui_openrouter_pipe")
        and name.split(".")[0].replace("_", "-") not in declared
    )
    assert not undeclared, (
        f"pytest.ini loads {undeclared} with -p, but the test extra does not require "
        "them. Anyone installing with the documented command gets a venv where pytest "
        "cannot start."
    )


def test_the_documented_install_commands_use_the_test_extra():
    """Every provisioning path derives from the extra rather than listing packages.

    Checked as DISJOINTNESS, not as a name allow-list. The previous condition was
    `'[test]' not in text and "pytest-timeout" not in text` -- `repro_venv_uv.sh`
    contains that literal, so it satisfied the guard forever while still maintaining its
    own hand-written list, and the next `-p` plugin would have drifted exactly as
    pytest-timeout did. A package named in both places is the defect, whatever it is
    called.
    """
    import re

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^test = \[(.*?)^\]", pyproject, re.M | re.S)
    assert block, "pyproject.toml has no [project.optional-dependencies] test extra"
    declared = {
        m.group(1).lower() for m in re.finditer(r'"([A-Za-z0-9_.-]+)"', block.group(1))
    }

    offenders = []
    for rel in (
        "scripts/repro_venv.sh",
        "scripts/repro_venv_uv.sh",
        "docs/testing_bootstrap_and_operations.md",
    ):
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # Only INSTALL commands. The file may legitimately mention `pytest` in prose or
        # in a command that RUNS the suite; restating it as something to install is the
        # defect.
        # Collapse shell line-continuations first: a package on a `\`-continued line is
        # part of the install command, and filtering line-by-line would miss it -- which
        # it did, leaving the restatement this guard exists to catch invisible.
        joined = re.sub(r"\\\n\s*", " ", text)
        install_lines = "\n".join(
            ln for ln in joined.splitlines() if re.search(r"\bpip install\b", ln)
        )
        if "[test]" not in install_lines:
            offenders.append(f"{rel}: does not install the `test` extra")
        restated = sorted(
            name
            for name in declared
            if re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", install_lines)
        )
        if restated:
            offenders.append(f"{rel}: also installs {restated} by hand")
    assert not offenders, (
        "provisioning must be derived from the `test` extra, not restated:\n  "
        + "\n  ".join(offenders)
    )


def _workflow(name: str) -> dict:
    import yaml

    return yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_the_verification_gate_is_defined_exactly_once():
    """Two copies of a gate drift, and the release-gating copy is the one nobody edits.

    `bundle.yml` carried a byte-for-byte second copy of `ci.yml`'s job because `publish`
    cannot gate on a job in another workflow. They had already diverged in the revision
    that created them -- one ran `compileall -q`, the other did not -- and every push
    paid for two full five-mode runs.

    Counted across the whole workflows directory rather than diffing the two files:
    diffing freezes today's duplication in place and goes red on legitimate divergence
    (different triggers, different job names), so it gets deleted the first time it is
    inconvenient. The property is one definition, not two kept equal.
    """
    workflows = sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml"))
    owners = [p.name for p in workflows if "OWUI_PIPE_BUNDLE_PATH" in p.read_text(encoding="utf-8")]
    assert owners == ["verify.yml"], (
        f"the five-mode test gate appears in {owners}. It belongs in exactly one "
        "reusable workflow that the others call with `uses:`."
    )


def test_both_workflows_call_the_shared_gate():
    """A workflow that stopped calling it would go green without running the gate."""
    for name, job in (("ci.yml", "tests-and-typecheck"), ("bundle.yml", "verify")):
        spec = _workflow(name)["jobs"][job]
        assert spec.get("uses") == "./.github/workflows/verify.yml", (
            f"{name}'s {job!r} job does not call the shared gate; it is either a second "
            "copy or it verifies nothing."
        )


def test_publish_ships_the_artifacts_the_gate_tested():
    """Rebuilding for release means shipping something no test ever ran against."""
    jobs = _workflow("bundle.yml")["jobs"]
    build_steps = jobs["bundle"]["steps"]
    assert jobs["bundle"].get("needs") == "verify", (
        "the build job does not depend on the gate, so it can publish while the gate "
        "is still running or has failed"
    )
    assert not any("bundle_v2.py" in str(s.get("run", "")) for s in build_steps), (
        "the build job re-runs the bundler, so the published artifacts are not the ones "
        "the five-mode suite ran against and nothing compares them"
    )
    assert any(
        str(s.get("uses", "")).startswith("actions/download-artifact") for s in build_steps
    ), "the build job neither builds nor downloads the bundles it uploads"


def test_every_job_has_a_timeout():
    """A hung job with no timeout burns a runner until the 6-hour default."""
    missing = []
    for name in ("ci.yml", "bundle.yml", "verify.yml"):
        for job_name, job in _workflow(name)["jobs"].items():
            if "uses" in job:
                continue
            if "timeout-minutes" not in job:
                missing.append(f"{name}:{job_name}")
    assert not missing, f"these jobs have no timeout-minutes: {missing}"


def test_the_gate_executes_once_per_push():
    """Defined once is not the same as run once.

    Extracting `verify.yml` made the gate have one DEFINITION while both callers still
    triggered on `push: branches: ["**"]`, so every push created two workflow runs and
    paid for two full five-mode suites. The guard written at the time counted the files
    containing `OWUI_PIPE_BUNDLE_PATH` -- which was correctly 1 the whole time.

    `bundle.yml` must keep the push trigger: it calls the gate directly because
    `publish` cannot depend on a job in another workflow, and it ships the artifact
    `verify` uploaded. So `ci.yml` is the one that gives up `push`, and covers pulls.
    """
    callers = {}
    for path in sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml")):
        spec = _workflow(path.name)
        if not any(
            str(job.get("uses", "")).endswith("verify.yml")
            for job in spec["jobs"].values()
        ):
            continue
        triggers = spec.get(True) or spec.get("on") or {}
        callers[path.name] = set(triggers)

    on_push = sorted(name for name, events in callers.items() if "push" in events)
    assert len(on_push) == 1, (
        f"{on_push} all call the shared gate AND trigger on push, so a single push runs "
        "the five-mode suite once per workflow. Exactly one caller may own the push "
        "event."
    )
    assert callers, "no workflow calls verify.yml at all; the gate never runs"
    covered = set().union(*callers.values())
    assert {"push", "pull_request"} <= covered, (
        f"the gate is only reachable from {sorted(covered)}; both pushes and pull "
        "requests must be verified."
    )


def test_no_prose_names_a_workflow_that_does_not_hold_the_pin():
    """A stale pointer sends the next reader to a file with nothing in it.

    `pyproject.toml` and `README-dev.md` both told the reader the ruff pin lived in
    `ci.yml` and `bundle.yml`. After the extraction it lives in `verify.yml`, and
    `WORKFLOWS` is glob-discovered, so nothing compared the prose to the fact.
    """
    import re

    holders = {rel.rsplit("/", 1)[-1] for rel in WORKFLOWS}
    offenders = []
    for rel in ("pyproject.toml", "README-dev.md"):
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"([A-Za-z0-9_.-]+\.yml)", text):
            named = match.group(1)
            if named.endswith(".yml") and named not in holders:
                context = text[max(0, match.start() - 120) : match.end() + 40]
                if re.search(r"ruff|pyright|pin", context, re.I):
                    offenders.append(f"{rel}: names {named}, but the pin lives in {sorted(holders)}")
    assert not offenders, "\n  ".join(offenders)
