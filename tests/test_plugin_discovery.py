"""Plugin-system discovery is mode-agnostic and names no specific plugin.

Regression cover for the removed hard-coded ``from . import pipe_dashboard`` in
``plugins/__init__.py``: the framework must *discover* plugins (filesystem in
package mode, the bundle manifest in compressed mode), never import one by name.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

from open_webui_openrouter_pipe import plugins as _plugins_pkg
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry

_PLUGINS_DIR = (
    Path(__file__).resolve().parents[1]
    / "open_webui_openrouter_pipe"
    / "plugins"
)
_FRAMEWORK_FILES = ("__init__.py", "base.py", "registry.py", "_utils.py")
_PREFIX = "open_webui_openrouter_pipe.plugins."

_INTROSPECTION_ATTRS = ("_discover_plugins", "_import_plugin_module", "__path__")
_MISSING_ATTRS = [_a for _a in _INTROSPECTION_ATTRS if not hasattr(_plugins_pkg, _a)]
_DISCOVERY_INTROSPECTABLE = not _MISSING_ATTRS

if _MISSING_ATTRS and not os.environ.get("OWUI_PIPE_BUNDLE_PATH"):
    raise RuntimeError(
        f"plugins package is missing {_MISSING_ATTRS} in package mode. Renaming a "
        "discovery internal must fail here, not silently skip every test below -- a "
        "guard that switches itself off when the thing it guards changes is no guard."
    )

_requires_introspectable_plugins = pytest.mark.skipif(
    not _DISCOVERY_INTROSPECTABLE,
    reason=(
        "flat bundle inlines the plugins package into one monolith exposing only "
        "public names, so the discovery-filtering algorithm is testable only in "
        "package/compressed modes (flat-mode registration is covered by "
        "test_plugin_registers_via_discovery)"
    ),
)


def _plugin_package_names() -> set[str]:
    """Plugin packages = subdirs of plugins/ that carry a plugin.py."""
    return {
        p.name
        for p in _PLUGINS_DIR.iterdir()
        if p.is_dir() and (p / "plugin.py").exists()
    }


def test_framework_never_names_a_plugin():
    """No framework file may reference a specific plugin package by name."""
    plugin_names = _plugin_package_names()
    assert plugin_names, "expected at least one plugin package (e.g. pipe_dashboard)"
    offenders: list[str] = []
    for fname in _FRAMEWORK_FILES:
        text = (_PLUGINS_DIR / fname).read_text(encoding="utf-8")
        for pname in plugin_names:
            if re.search(rf"\b{re.escape(pname)}\b", text):
                offenders.append(f"{fname} references plugin '{pname}'")
    assert not offenders, "Framework must discover, not name, a plugin: " + "; ".join(offenders)


def test_plugin_registers_via_discovery():
    """Importing the plugins package auto-registers plugins (package mode)."""
    pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")
    registered = {c.plugin_id for c in PluginRegistry._plugin_classes}
    assert "pipe-dashboard" in registered


@_requires_introspectable_plugins
def test_manifest_discovery_imports_only_top_level_plugin_packages(monkeypatch):
    """The bundle-mode branch imports exactly the top-level plugin packages a
    finder's generic ``bundled_module_names()`` lists — skipping framework
    modules, deep submodules, and foreign packages — with no plugin name in the
    framework code path."""
    imported: list[str] = []
    monkeypatch.setattr(_plugins_pkg, "_import_plugin_module", lambda name: imported.append(name))
    monkeypatch.setattr(_plugins_pkg, "__path__", [])

    class _FakeBundleFinder:
        def bundled_module_names(self):
            return [
                _PREFIX + "base",
                _PREFIX + "registry",
                _PREFIX + "pipe_dashboard",
                _PREFIX + "pipe_dashboard.plugin",
                "some.other.package",
            ]

    monkeypatch.setattr(sys, "meta_path", [_FakeBundleFinder(), *sys.meta_path])
    _plugins_pkg._discover_plugins()

    assert imported == [_PREFIX + "pipe_dashboard"]


@_requires_introspectable_plugins
def test_finder_without_manifest_contract_is_ignored(monkeypatch):
    """A meta_path finder that does not expose bundled_module_names() is skipped
    (no crash), so ordinary finders never interfere with discovery."""
    monkeypatch.setattr(_plugins_pkg, "__path__", [])

    class _PlainFinder:
        pass

    monkeypatch.setattr(sys, "meta_path", [_PlainFinder(), *sys.meta_path])
    _plugins_pkg._discover_plugins()
