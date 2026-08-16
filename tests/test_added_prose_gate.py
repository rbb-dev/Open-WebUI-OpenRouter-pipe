"""The Phase 0 prose gate, exercised against real diffs and real modules.

A grep over added ``#`` lines is satisfied by moving the sentence into an adjacent string
literal, and the changeset that motivated this gate added 259 docstring lines and 44 bare
string-literal lines that such a grep cannot see. So the gate reports three categories and
this file drives each of them separately: a check that only ever ran the tokenize half
would pass with the ``ast`` half deleted, and the two string categories would go dark.

Every test builds a real git repository and runs the real ``git diff``, because the one
thing the gate must get right is *which lines the diff added* -- the previous shape used
``base..HEAD`` and could not see the working tree it was written to inspect.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_GATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_added_prose.py"


def _load_gate() -> Any:
    spec = importlib.util.spec_from_file_location("_check_added_prose_under_test", _GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


gate = _load_gate()


def _repo(tmp_path: Path, baseline: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for rel, body in baseline.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base"],
        cwd=root,
        check=True,
    )
    return root


def _write(root: Path, rel: str, body: str) -> None:
    (root / rel).write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    ("appended", "expected"),
    [("x = 2\n", {3}), ("x = 2\ny = 3\n", {3, 4})],
)
def test_added_lines_are_the_working_tree_lines_the_diff_introduced(
    tmp_path, appended, expected
):
    """Two different appends, so a constant line set cannot satisfy both.

    The diff runs to the WORKING TREE. ``base..HEAD`` sees nothing here, which is the
    state this gate is written to inspect.
    """
    root = _repo(tmp_path, {"pkg/m.py": "a = 0\nb = 1\n"})
    _write(root, "pkg/m.py", "a = 0\nb = 1\n" + appended)

    assert gate.added_lines("HEAD", "pkg", root=root) == {"pkg/m.py": expected}


def test_a_file_that_did_not_exist_is_reported_under_its_new_path(tmp_path):
    """A brand-new module is entirely added lines, and `/dev/null` is on the minus side."""
    root = _repo(tmp_path, {"pkg/m.py": "a = 0\n"})
    _write(root, "pkg/n.py", "# fresh prose\nb = 1\n")

    added = gate.added_lines("HEAD", "pkg", root=root)

    assert added["pkg/n.py"] == {1, 2}


@pytest.mark.parametrize(
    ("comment", "counted"),
    [("# explain the thing", True), ("# describe the other thing", True)],
)
def test_a_plain_comment_is_reported_as_a_comment(comment, counted):
    found = gate.prose_lines(f"x = 1  {comment}\n", gate.directive_comment_re())

    assert (1 in found["comment"]) is counted
    assert not found["docstring"] and not found["bare-string"]


@pytest.mark.parametrize("directive", ["# noqa: B010", "# type: ignore[arg-type]"])
def test_a_directive_comment_is_not_prose(directive):
    """Sourced from the bundler's own rule: the bundler decides which comments are
    load-bearing, and a second copy here would drift the moment it gains a form."""
    found = gate.prose_lines(f"x = 1  {directive}\n", gate.directive_comment_re())

    assert not found["comment"], (
        f"{directive!r} was scored as prose, so every suppression in the package would "
        "have to be deleted to pass a gate about explanatory sentences"
    )


@pytest.mark.parametrize("owner", ["def f():", "class C:"])
def test_a_docstring_in_a_docstring_position_is_a_docstring(owner):
    source = f'{owner}\n    """Explain."""\n    x = 1\n'

    found = gate.prose_lines(source, gate.directive_comment_re())

    assert found["docstring"] == {2}
    assert not found["bare-string"]


@pytest.mark.parametrize("filler", ['"""Loose prose."""', '"Loose prose."'])
def test_a_string_statement_that_is_not_a_docstring_is_a_bare_string(filler):
    """The third way to add prose, and the one a grep and a docstring rule both miss."""
    source = f"def f():\n    x = 1\n    {filler}\n"

    found = gate.prose_lines(source, gate.directive_comment_re())

    assert found["bare-string"] == {3}
    assert not found["docstring"]


def test_the_module_docstring_is_a_docstring_not_a_bare_string():
    found = gate.prose_lines('"""Module prose."""\nx = 1\n', gate.directive_comment_re())

    assert found["docstring"] == {1}
    assert not found["bare-string"]


def test_the_directive_rule_is_the_bundlers_own(tmp_path):
    """Not a restatement: it is loaded out of scripts/bundle_v2.py at call time."""
    import re

    stand_in = tmp_path / "fake_bundler.py"
    stand_in.write_text(
        "import re\n_DIRECTIVE_COMMENT_RE = re.compile(r'ZZZ-ONLY')\n", encoding="utf-8"
    )

    pattern = gate.directive_comment_re(stand_in)

    assert isinstance(pattern, re.Pattern)
    assert pattern.pattern == "ZZZ-ONLY", (
        "the gate used its own copy of the directive rule instead of the bundler's, so it "
        "will keep passing after the bundler adds a fifth directive form"
    )


def test_a_bundler_that_stops_publishing_the_rule_is_a_hard_stop(tmp_path):
    """Silently falling back to 'no directives' would report every noqa as new prose."""
    stand_in = tmp_path / "fake_bundler.py"
    stand_in.write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(TypeError, match="_DIRECTIVE_COMMENT_RE"):
        gate.directive_comment_re(stand_in)


@pytest.mark.parametrize(
    ("added_line", "category"),
    [("# a new sentence", "comment"), ('"""a new sentence"""', "bare-string")],
)
def test_only_prose_on_an_added_line_is_reported(tmp_path, added_line, category):
    """An untouched comment above an edited line is not this changeset's prose.

    Two categories on the same shape, so a sweep that intersects with the wrong line set
    cannot satisfy both.
    """
    root = _repo(tmp_path, {"pkg/m.py": "# pre-existing sentence\nx = 1\n"})
    _write(root, "pkg/m.py", f"# pre-existing sentence\nx = 1\ny = 2\n{added_line}\n")

    report = gate.sweep("HEAD", "pkg", root=root, bundler=Path("scripts/bundle_v2.py").resolve())

    assert report["pkg/m.py"][category] == [4]
    assert 1 not in report["pkg/m.py"]["comment"], (
        "a comment that was already there was scored against this changeset"
    )


@pytest.mark.parametrize(
    ("allowed", "failing"),
    [(set(), {"comment", "docstring"}), ({"docstring"}, {"comment"})],
)
def test_allowing_a_category_removes_it_from_the_exit_code_and_nothing_else(allowed, failing):
    report = {"pkg/m.py": {"comment": [1], "bare-string": [], "docstring": [2]}}

    assert gate.failing_categories(report, allowed) == failing
    assert "comment" in gate.render(report) and "docstring" in gate.render(report)


@pytest.mark.parametrize(
    ("added", "allow", "expected_code"),
    [("# a sentence\n", "", 1), ('def g():\n    """A sentence."""\n', "docstring", 0)],
)
def test_the_exit_code_says_whether_the_gate_passed(
    tmp_path, monkeypatch, capsys, added, allow, expected_code
):
    """Opposite outcomes from one entry point, so a constant return cannot pass."""
    root = _repo(tmp_path, {"pkg/m.py": "x = 1\n"})
    _write(root, "pkg/m.py", "x = 1\n" + added)
    monkeypatch.setattr(gate, "PROJECT_ROOT", root)

    code = gate.main(["HEAD", "--path", "pkg"] + (["--allow", allow] if allow else []))
    capsys.readouterr()

    assert code == expected_code


def test_an_unknown_allow_category_is_refused_rather_than_ignored(tmp_path, monkeypatch):
    """A typo'd --allow that silently allowed nothing would read as a clean gate."""
    root = _repo(tmp_path, {"pkg/m.py": "x = 1\n"})
    monkeypatch.setattr(gate, "PROJECT_ROOT", root)

    with pytest.raises(SystemExit):
        gate.main(["HEAD", "--path", "pkg", "--allow", "docstrings"])
