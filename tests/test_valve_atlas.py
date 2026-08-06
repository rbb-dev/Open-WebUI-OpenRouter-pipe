"""The atlas is the authoritative valve reference, and nothing read it.

Four surfaces describe each valve -- the `Field(description=...)`, the dashboard
`detail`, a topic doc, and the atlas row -- and the atlas was the one no test touched. It
kept saying `USE_MODEL_MAX_OUTPUT_TOKENS` "forwards provider-advertised max_output_tokens
automatically" through the entire correction of the other three, which is false in exactly
the case those three were rewritten for. Two valves had no row at all.

What is checked here is what CAN be checked without pinning prose. A content guard was
tried and measured first: requiring every number in a Field description to appear in that
valve's row produced 14 findings on this tree, of which 13 were HTTP status codes, key
lengths, queue sizes and version digits inside example model ids. A guard with a 93%
false-positive rate is an exemption list waiting to happen, and a number appearing in a
row does not mean the row agrees with the Field anyway.

One direction only for presence. The atlas also documents filter valves, plugin valves and
provider-routing sub-keys -- 68 rows measured -- that are not fields of these models, so
pinning the reverse direction would need an exemption list longer than the guard.
"""

from __future__ import annotations

import re
from pathlib import Path

from open_webui_openrouter_pipe.core.config import UserValves, Valves

ATLAS = Path(__file__).resolve().parents[1] / "docs" / "valves_and_configuration_atlas.md"
_ROW = re.compile(r"^\|\s*`([A-Z0-9_]+)`\s*\|")


def _rows() -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = {}
    for line in ATLAS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        match = _ROW.match(stripped)
        if match:
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            rows.setdefault(match.group(1), []).append(cells)
    return rows


def test_every_valve_has_an_atlas_row():
    """A valve an operator can set and cannot look up is undocumented, not documented."""
    documented = set(_rows())
    assert len(documented) > 150, (
        f"only {len(documented)} valve rows parsed out of the atlas; the row format "
        "changed and this guard is asserting almost nothing"
    )
    declared = set(Valves.model_fields) | set(UserValves.model_fields)
    missing = sorted(declared - documented)
    assert not missing, (
        f"these valves have no row in {ATLAS.name}: {missing}. The atlas is the only "
        "place an operator can look a valve up by name."
    )


def test_the_atlas_default_column_matches_the_model():
    """The atlas heads that column "Default (verified)". Verify it.

    Booleans only: they are the subset where the cell is mechanically comparable, and all
    of them are written as exactly `True`/`False`. A string or Literal default is written
    with surrounding prose about where it comes from, and pinning that would be pinning
    prose. Every row for a name is checked, not just the first -- several valves are
    documented in two sections and a second row can disagree with the first.
    """
    rows = _rows()
    declared = dict(Valves.model_fields)
    declared.update(UserValves.model_fields)

    checked = 0
    wrong: list[str] = []
    for name, field in declared.items():
        if not isinstance(field.default, bool):
            continue
        for cells in rows.get(name, []):
            cell = cells[2] if len(cells) > 2 else ""
            checked += 1
            if cell.strip().strip("`").strip() != str(field.default):
                wrong.append(
                    f"{name}: atlas says {cell!r}, the model says {field.default!r}"
                )

    assert checked > 50, (
        f"only {checked} boolean default cells were compared; the discovery has gone "
        "blind and this guard is asserting nothing"
    )
    assert not wrong, (
        "the atlas states its defaults are verified against the source, and these are "
        "not:\n  " + "\n  ".join(sorted(wrong))
    )
