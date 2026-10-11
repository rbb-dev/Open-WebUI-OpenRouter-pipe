"""The mid-work guard must refuse a whole bundled suite and still let the gate collect one.

Running every test against a flattened artifact costs minutes; collecting them costs seconds and answers a
different question -- whether the flattened file still imports. The batch tier does the second deliberately, so
the guard has to tell them apart.
"""

from __future__ import annotations

import pytest

from conftest import pytest_collection_modifyitems

_BUNDLE_ENV = "OWUI_PIPE_BUNDLE_PATH"
_EXEMPTIONS = ("GATE_BUNDLE_RUN_APPROVED", "CI", "GITHUB_ACTIONS")


class _Config:
    def __init__(self, *, collectonly: bool) -> None:
        self.option = type("_Option", (), {"collectonly": collectonly})()


def _items(count: int) -> list[object]:
    return [object() for _ in range(count)]


@pytest.fixture(autouse=True)
def _no_inherited_exemptions(monkeypatch):
    for name in (_BUNDLE_ENV, *_EXEMPTIONS):
        monkeypatch.delenv(name, raising=False)


def test_pytest_still_spells_the_collect_only_flag_the_way_the_guard_reads_it(pytestconfig):
    # The stand-in config below is only honest while the real one carries this attribute.
    assert isinstance(pytestconfig.option.collectonly, bool)


def test_running_the_whole_suite_against_a_bundle_is_refused(monkeypatch):
    monkeypatch.setenv(_BUNDLE_ENV, "/artifacts/bundle.py")

    with pytest.raises(pytest.UsageError, match="Refusing to run 600 tests"):
        pytest_collection_modifyitems(None, _Config(collectonly=False), _items(600))


def test_collecting_the_whole_suite_against_a_bundle_is_allowed(monkeypatch):
    # What `scripts/gate.sh batch` does under each artifact, to catch defects that only show once the package is
    # one file. It never runs a test, so the minutes the guard exists to save are not at stake.
    monkeypatch.setenv(_BUNDLE_ENV, "/artifacts/bundle.py")

    assert pytest_collection_modifyitems(None, _Config(collectonly=True), _items(600)) is None


@pytest.mark.parametrize("exemption", [_EXEMPTIONS[i] for i in (0, -1)])
def test_a_run_that_was_asked_for_and_a_run_in_ci_are_let_through(monkeypatch, exemption):
    monkeypatch.setenv(_BUNDLE_ENV, "/artifacts/bundle.py")
    monkeypatch.setenv(exemption, "1")

    assert pytest_collection_modifyitems(None, _Config(collectonly=False), _items(600)) is None


@pytest.mark.parametrize("count", [1, 499])
def test_a_targeted_bundled_run_is_untouched(monkeypatch, count):
    monkeypatch.setenv(_BUNDLE_ENV, "/artifacts/bundle.py")

    assert pytest_collection_modifyitems(None, _Config(collectonly=False), _items(count)) is None


@pytest.mark.parametrize("count", [600, 20_000])
def test_the_package_suite_is_untouched(count):
    assert pytest_collection_modifyitems(None, _Config(collectonly=False), _items(count)) is None
