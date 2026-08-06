"""The shared extractor's own coverage, over the four shapes that appear in the tree.

Driven through `causes_in_source` on literal text rather than through a refactor of a
real module: moving a module to `warn_latch.warn_level(...)` attribute-style is the
obvious way to exercise the Attribute branch and it is ILLEGAL here -- scripts/bundle_v2
deletes every relative `from . import X`, so that spelling is a NameError in all three
bundled modes and F821 in CI.
"""

from __future__ import annotations

import textwrap

from tests.warn_latch_census import causes_in_source

SOURCE = textwrap.dedent(
    '''
    warn_level(_warned_x, "plain")
    warn_level(_warned_x, f"fstring:{type(exc).__name__}")
    warn_level(self._warned_x, "on_self")
    warn_level(mod._warned_x, f"dotted:{exc}")
    warn_level(_warned_x, type(exc).__name__)
    warn_level(_warned_other, "not_mine")
    warn_level(_warned_x)
    '''
)


def test_every_shape_in_the_tree_is_read_or_reported():
    causes, unresolvable = causes_in_source(SOURCE, "_warned_x")
    assert causes == {"plain", "fstring", "on_self", "dotted"}
    assert unresolvable == ["line 6: type(exc).__name__"]
