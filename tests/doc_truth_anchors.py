"""One anchor idiom for the doc-truth tests, so a claim is found the same way everywhere.

Every test in this group asks the same question of a shipped text: does the sentence
an operator reads there say what the code does? They differ only in which file and
which boundary they point at. Repeating the "split on the anchor, assert the anchor
exists exactly once, return the span" idiom in each file is how one of them ends up
silently passing on an empty span -- `str.split` on a missing anchor still returns a
list, just the wrong one -- so the idiom lives here once and is imported by each.

The boundary is named, never a line number. A doc-truth test that pins line 80 fails
the next time a paragraph above it grows a line, and the fix is then to bump the
number, which trains the reader to ignore red without reading it. A heading or a
sentence is a thing the document either still has or genuinely no longer has, and
either answer is worth failing on.

The same holds for the help texts a valve ships. Those are Python strings, and a regex
over the source reads the *literal*, not the value: `"..." + _CONST` -- an ordinary
refactor once one sentence is shared between two valves -- reads short, so the guard
quietly checks less than it claims. The dashboard reader fails in the other direction,
running on into the next entry, where a claim can be satisfied by a neighbour's wording.
So the two readers below parse the module and fold the string expression, and what comes
back is the words the Config tab renders. They read the files, never import them, so
every caller still collects under the release gate's no-plugins artifact.
"""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
def doc(name: str) -> str:
    """A docs/ file's text, or an AssertionError naming the file that went missing."""
    path = _ROOT / "docs" / name
    assert path.is_file(), f"{name} is gone from docs/; the test pointing at it is stale"
    return path.read_text(encoding="utf-8")


def section(text: str, start_heading: str, end_heading: str, *, what: str) -> str:
    """The span from `start_heading`'s own line up to `end_heading`'s own line.

    Both boundaries are full ATX headings and must occur exactly once, so a document
    that duplicates a section cannot make the extraction quietly stop at the first
    copy and hand the assertions a span that no longer says what it used to. A
    subheading (`###`) cannot close a `###` section, so the two are normally written
    as siblings and the level is not part of the contract.
    """
    lines = text.splitlines()
    starts = [n for n, line in enumerate(lines) if line == start_heading]
    assert len(starts) == 1, (
        f"{what}: the heading {start_heading!r} occurs {len(starts)} times as a whole "
        f"line, so the section's start is not the single place this test thinks it is. "
        f"Either the section was duplicated, or it was renamed and the test is stale."
    )
    tail = lines[starts[0] + 1 :]
    ends = [n for n, line in enumerate(tail) if line == end_heading]
    assert len(ends) == 1, (
        f"{what}: the section opened by {start_heading!r} has {len(ends)} whole lines "
        f"equal to {end_heading!r} after it, so its end is not where this test thinks it "
        f"is. The section was probably duplicated, or it was renamed and the test is stale."
    )
    return "\n".join(lines[starts[0] : starts[0] + 1 + ends[0]])


def between(text: str, anchor: str, *, what: str) -> str:
    """The anchor plus the paragraph it opens, up to the next blank line.

    Doc-truth assertions are about a claim, and a claim is written as a paragraph
    starting with a sentence this test can name. Reading to the next blank line keeps
    the assertion on the claim without pinning how many sentences the paragraph has.
    """
    assert anchor in text, (
        f"{what}: the claim {anchor!r} is not in the document any more, so this test can "
        f"no longer tell which text it was reading"
    )
    head = text.index(anchor)
    return text[head : head + len(text[head:].split("\n\n", 1)[0])].strip()
