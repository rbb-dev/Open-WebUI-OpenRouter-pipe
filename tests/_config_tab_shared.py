"""The one definition that crosses a section boundary of the config-tab test suite.

`test_config_tab.py` and `test_config_tab_help_text.py` both read the fixture
directory, so it is defined once here and imported by both.
"""

from __future__ import annotations

from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
