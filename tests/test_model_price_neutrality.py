"""What a model charges is OpenRouter's to publish, so no surface here may rank models by it.

The repo owner's ruling: "remove anything about pricing or what cheaper or not, its not up
to us". Three separate passes have now had to strip the same claim back out of a different
surface -- the help cards, the subagent's "cheaper worker model", and the Fusion presets --
because nothing held any of them in place once the words were gone.

`tests/test_help_command_routing.py` owns the help-card corpus and
`tests/test_web_tools_filter.py` owns the subagent surfaces. This owns the third set: the
Fusion preset surfaces (a generated filter and its two mirrors), the housekeeping task-model
guidance, and the model-variant guidance. Between them the three sweeps cover every text in
which this project has ever told a reader that one model costs less than another.

The exemption is one sentence, `OPENROUTER_PRICING`, stripped whole before either rule reads
anything. It names no figure, no direction and no comparison -- it hands the question to the
company that owns the answer. Stripping it rather than exempting a word keeps this a ban: a
surface carrying the pointer AND a ranking of its own still fails on the ranking.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from open_webui_openrouter_pipe.core.config import _OPENROUTER_FUSION_FILTER_MARKER
from open_webui_openrouter_pipe.filters.fusion_filter_renderer import (
    render_openrouter_fusion_filter_source,
)
from open_webui_openrouter_pipe.integrations.image_help import OPENROUTER_PRICING

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_FUSION_DOC = _DOCS / "openrouter_fusion.md"
_TASK_DOC = _DOCS / "task_models_and_housekeeping.md"
_VARIANTS_DOC = _DOCS / "model_variants_and_presets.md"

_PRESET_IDS = ("general-high", "general-budget", "general-fast")

_RANKING_WORDS = re.compile(
    r"\b(?:cheap|cheaper|cheapest|cheaply|low-cost|lower-cost|high-cost|higher-cost"
    r"|costly|costlier|costliest|expensive|inexpensive|cost-efficient|cost-effective"
    r"|cost-optimised|cost-optimized|pricey|pricier|priciest|affordable|economical"
    r"|budget-friendly)\b",
    re.IGNORECASE,
)
"""Adjectives that place a model on a money scale without needing a second model named.

Every one of them was in this project's text at some point; `low-cost`, `cheaper`,
`expensive`, `cost-efficient` and `cost-effective` were on these very surfaces. `budget` is
absent on purpose: `general-budget` is OpenRouter's own enum value, sent on the wire in
`plugins[].preset`, and `max_tool_calls` is documented as a tool budget. The ban is on the
claim, never on the identifier that carries it.
"""

_MONEY_WORDS = frozenset(
    """
    price prices pricing priced pricier priciest
    cost costs costed costing costly costlier costliest
    cheap cheaper cheapest cheaply expensive inexpensive
    bill bills billed billing charge charges charged fee fees rate rates
    pay pays paid spend spends spending spent
    surcharge surcharges discount discounts affordable premium economics
    dollar dollars
    """.split()
)
"""The vocabulary of `tests/test_help_command_routing.py::_MONEY_WORDS`, applied here.

Shared by copy rather than by import, for the reason the subagent sweep gives: each sweep
owns a different corpus with a different exemption, and importing one test module into
another to share a frozenset couples their collection order for nothing.
"""

_COMPARATIVES = re.compile(
    r"\b(?:more|less|than|fewer|greater|higher|lower|times|fraction|multiple"
    r"|most|least|double|triple|half)\b",
    re.IGNORECASE,
)
"""What a ranking reaches for once the adjectives above are gone.

This is the rule the last pass needed and did not have. "costs far more than a normal
completion" carries no banned adjective, reads as a plain statement of fact, and is exactly
the claim the ruling removes -- a word ban alone walks straight past it.
"""

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _clauses(text: str):
    """Every clause a reader meets, with the shared pricing pointer removed first.

    Markdown cells are split on `|` because a table row is not one sentence: the symptom,
    the cause and the remedy are three independent claims, and running them together would
    read a comparative in one cell against a money word in another.
    """
    stripped = text.replace(OPENROUTER_PRICING, " ")
    for line in stripped.replace("\\n", " ").splitlines():
        for cell in line.split("|"):
            for sentence in _SENTENCE.split(cell):
                if sentence.strip():
                    yield sentence


def _rankings(text: str) -> list[str]:
    """Every place `text` puts a model, a preset or a variant on a money scale."""
    found: list[str] = []
    for clause in _clauses(text):
        for match in _RANKING_WORDS.finditer(clause):
            found.append(f"{match.group(0)!r} in {clause.strip()[:150]!r}")
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", clause)}
        money = sorted(words & _MONEY_WORDS)
        if money and _COMPARATIVES.search(clause):
            found.append(f"{money} compared in {clause.strip()[:150]!r}")
    return found


def _money_clauses(text: str) -> int:
    """How many clauses carry money vocabulary at all -- the sweep's own liveness signal."""
    total = 0
    for clause in _clauses(text):
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", clause)}
        if words & _MONEY_WORDS:
            total += 1
    return total


_CONFIG_TAB = "the Config tab: ENABLE_OPENROUTER_FUSION"


def _surfaces() -> dict[str, str]:
    """Every text through which a reader meets a Fusion preset, a task model or a variant.

    The filter is read as the source the pipe *generates and installs*, not as the template
    it is spliced from: users get the rendered copy, and a claim reintroduced anywhere in
    that pipeline reaches them.

    Four of the five need nothing but this repo, so the ban runs in every build. Only the
    Config tab depends on `pipe_dashboard`, which the `--no-plugins` artifacts omit by
    design; skipping the whole sweep for it would take the docs and the filter down with
    it, and those are where two of the three previous passes found the claim.
    """
    surfaces = {
        "the generated Fusion filter": render_openrouter_fusion_filter_source(
            marker=_OPENROUTER_FUSION_FILTER_MARKER
        ),
        "docs/openrouter_fusion.md": _FUSION_DOC.read_text(encoding="utf-8"),
        "docs/task_models_and_housekeeping.md": _TASK_DOC.read_text(encoding="utf-8"),
        "docs/model_variants_and_presets.md": _VARIANTS_DOC.read_text(encoding="utf-8"),
    }
    try:
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_meta import CONFIG_META
    except ImportError:
        return surfaces
    meta = CONFIG_META["ENABLE_OPENROUTER_FUSION"]
    surfaces[_CONFIG_TAB] = f"{meta['title']}. {meta['detail']}"
    return surfaces


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        pytest.param(
            "'general-budget' = a fast low-cost trio with the same frontier judge.",
            True,
            id="the-preset-ranking-that-was-there",
        ),
        pytest.param(
            "the model decides whether the prompt needs the multi-model panel (cheaper).",
            True,
            id="the-forcing-ranking-that-was-there",
        ),
        pytest.param(
            "Running a whole panel plus a judge for one answer costs far more than a "
            "normal completion.",
            True,
            id="a-comparative-carrying-no-banned-adjective",
        ),
        pytest.param(
            "Housekeeping is targeting a high-cost model or generating long outputs.",
            True,
            id="the-housekeeping-ranking-that-was-there",
        ),
        pytest.param(
            "Low-latency and cost-efficient for short outputs.",
            True,
            id="the-housekeeping-guidance-that-was-there",
        ),
        pytest.param(
            "Provides cost-effective options for testing.",
            True,
            id="the-variant-ranking-that-was-there",
        ),
        pytest.param("May have higher costs per token.", True, id="a-variant-comparative"),
        pytest.param(
            "'general-budget' = a faster trio with the same frontier judge.",
            False,
            id="a-roster-described-by-latency",
        ),
        pytest.param(
            "Running a whole panel plus a judge for one answer is roughly four to five "
            "model calls instead of one.",
            False,
            id="our-own-fan-out-stated-as-calls",
        ),
        pytest.param(
            "Confirm the configured task model and review usage/cost snapshots.",
            False,
            id="cost-accounting-is-not-a-ranking",
        ),
        pytest.param(
            OPENROUTER_PRICING,
            False,
            id="the-shared-pointer-is-the-one-exemption",
        ),
        pytest.param(
            "Tool budget for each panel and judge model (1-16; 0 = default 8).",
            False,
            id="a-tool-budget-is-not-a-price",
        ),
        pytest.param(
            "FUSION_PRESET: Literal['', 'general-high', 'general-budget', 'general-fast']",
            False,
            id="the-wire-enum-openrouter-publishes",
        ),
    ],
)
def test_the_sweep_reads_a_ranking_as_a_ranking_and_leaves_everything_else(text, flagged):
    """The detector, exercised on both answers, so no constant can satisfy it."""
    assert bool(_rankings(text)) is flagged, _rankings(text)


def test_no_surface_that_names_a_preset_a_task_model_or_a_variant_ranks_it_by_money():
    """The ruling, applied to every surface at once."""
    surfaces = _surfaces()
    assert len(surfaces) >= 4, f"only {len(surfaces)} surfaces swept; the sweep went hollow"

    offenders = [
        f"{where}: {finding}"
        for where, text in surfaces.items()
        for finding in _rankings(text)
    ]
    assert not offenders, (
        "What a model charges is OpenRouter's to publish, and these surfaces rank it "
        "anyway:\n" + "\n".join(offenders)
    )


def test_the_sweep_actually_reads_the_subjects_it_claims_to_cover():
    """A rename that empties a surface must fail here, not pass the ban by reading nothing."""
    surfaces = _surfaces()

    filter_source = surfaces["the generated Fusion filter"]
    fusion_doc = surfaces["docs/openrouter_fusion.md"]
    for preset in _PRESET_IDS:
        assert preset in filter_source, f"{preset} is no longer in the generated filter"
        assert preset in fusion_doc, f"{preset} is no longer in docs/openrouter_fusion.md"

    assert "task model" in surfaces["docs/task_models_and_housekeeping.md"]
    variants = surfaces["docs/model_variants_and_presets.md"]
    assert ":free" in variants and ":extended" in variants

    money = sum(_money_clauses(text) for text in surfaces.values())
    assert money >= 20, (
        f"only {money} clauses across these surfaces mention money at all; the sweep is "
        "reading too little to be evidence that it found nothing"
    )


def test_the_config_tab_is_swept_in_every_build_that_ships_the_dashboard():
    """The one surface that can drop out must drop out only where it does not exist."""
    pytest.importorskip(
        "open_webui_openrouter_pipe.plugins.pipe_dashboard",
        reason="the --no-plugins artifacts omit pipe_dashboard by design",
    )
    surfaces = _surfaces()
    assert _CONFIG_TAB in surfaces, (
        "pipe_dashboard imports here, so its Fusion entry is a surface a reader meets and "
        "the sweep dropped it silently"
    )
    assert "Fusion" in surfaces[_CONFIG_TAB]
