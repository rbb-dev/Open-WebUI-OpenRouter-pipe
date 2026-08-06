"""Every documented status must be classified, or the build fails.

`_server_tool_status` decides whether a server-tool card reads as in-flight, succeeded
or failed. It consults the classification sets, never the enums those values come from,
so an enum can be widened without the decision changing -- and the default for an
unrecognised value is "incomplete", which the streaming path treats as an error.

The concrete cost of that gap is an image: OpenRouter adds a value to
ImageGenerationStatus, a maintainer adds it to the enum and nothing else, and an
`image_generation_call` carrying it is carded incomplete. streaming_core then takes the
error branch, pushes "Image generation failed" to the user, and skips rendering -- an
image that was generated and billed, thrown away.

Requiring the classification to be TOTAL over both enums closes it: the unclassified
value fails here instead of being silently filed as a failure.
"""

from __future__ import annotations

import pytest

from open_webui_openrouter_pipe.core.utils import (
    SERVER_TOOL_EXTRA_SUCCESS,
    SERVER_TOOL_FAILURE_STATUSES,
    SERVER_TOOL_IN_FLIGHT_STATUSES,
    SERVER_TOOL_SUCCESS_STATUSES,
    TOOL_CALL_STATUSES,
)

_CLASSIFICATIONS = {
    "in-flight": SERVER_TOOL_IN_FLIGHT_STATUSES,
    "success": SERVER_TOOL_SUCCESS_STATUSES,
    "failure": SERVER_TOOL_FAILURE_STATUSES,
}


def test_server_tool_status_classification_is_total():
    """Each documented status is classified exactly once."""
    documented = TOOL_CALL_STATUSES | IMAGE_GENERATION_STATUSES | WEB_SEARCH_STATUSES
    assert documented, "the status enums are empty; this test is checking nothing"

    unclassified, ambiguous = [], []
    for status in sorted(documented):
        holders = [name for name, members in _CLASSIFICATIONS.items() if status in members]
        if not holders:
            unclassified.append(status)
        elif len(holders) > 1:
            ambiguous.append(f"{status} -> {', '.join(sorted(holders))}")

    assert not unclassified, (
        "these statuses are documented by OpenRouter but classified nowhere, so "
        "_server_tool_status falls through to 'incomplete' and cards them as failures. "
        "For an image that means the generated, billed result is discarded behind an "
        f"'Image generation failed' notice: {unclassified}"
    )
    assert not ambiguous, (
        f"these statuses are in more than one classification, so which one wins depends "
        f"on the order the branches happen to be written in: {ambiguous}"
    )


def test_no_classification_invents_a_status_the_enums_do_not_document():
    """The reverse direction: a set may only extend the enums deliberately.

    `ok` is admitted on purpose and recorded in SERVER_TOOL_EXTRA_SUCCESS. Anything else
    means someone matched a value from a provider's prose without checking whether the
    schema actually has it, and the schema is what the wire carries.
    """
    documented = TOOL_CALL_STATUSES | IMAGE_GENERATION_STATUSES | WEB_SEARCH_STATUSES | SERVER_TOOL_EXTRA_SUCCESS
    invented = {
        name: sorted(members - documented)
        for name, members in _CLASSIFICATIONS.items()
        if members - documented
    }
    assert not invented, (
        "these classified values appear in no OpenRouter enum and are not recorded as a "
        f"deliberate extension in SERVER_TOOL_EXTRA_SUCCESS: {invented}"
    )


# The reading each documented status MUST produce, written from what the status means
# rather than derived from the classification sets. Deriving it from the sets makes the
# test tautological: moving "searching" from in-flight to failure moves both sides
# together and the assertion still holds, which is precisely the regression this guards.
# OpenRouter's enums, transcribed here rather than imported from the package.
# Nothing in production read them -- only this file did -- so they were two frozensets
# compiled into all four shipped bundles to serve a test. Keeping them here also keeps
# the totality check honest: derived from SERVER_TOOL_* they would be tautological, which
# is the trap the comment below the table already warns about.
IMAGE_GENERATION_STATUSES = frozenset({"in_progress", "completed", "generating", "failed"})
WEB_SEARCH_STATUSES = frozenset({"in_progress", "searching", "completed", "failed"})


_EXPECTED_READING = {
    "in_progress": "in_progress",
    "generating": "in_progress",
    "searching": "in_progress",
    "completed": "completed",
    "ok": "completed",
    "incomplete": "incomplete",
    "failed": "incomplete",
}

_DOCUMENTED = sorted(
    set(TOOL_CALL_STATUSES)
    | set(IMAGE_GENERATION_STATUSES)
    | set(WEB_SEARCH_STATUSES)
    | set(SERVER_TOOL_EXTRA_SUCCESS)
)


def test_every_documented_status_has_an_expected_reading():
    """The table above must cover the enums, or a new status arrives with no arm.

    A hand-written table had six rows against six documented statuses and still missed
    one: `searching` had none while `ok` had two. Comparing the key sets in both
    directions is what makes the omission a failure instead of a quiet gap.
    """
    missing = sorted(set(_DOCUMENTED) - set(_EXPECTED_READING))
    extra = sorted(set(_EXPECTED_READING) - set(_DOCUMENTED))
    assert not missing, (
        f"these documented statuses have no expected reading, so nothing pins how they "
        f"are carded: {missing}"
    )
    assert not extra, (
        f"these expectations name statuses no enum documents any more: {extra}"
    )


@pytest.mark.parametrize("status", _DOCUMENTED, ids=_DOCUMENTED)
def test_the_card_reads_each_documented_status_the_way_it_is_classified(status):
    """Ties the classification to the function that consumes it.

    DERIVED from the status sets rather than listed by hand. A hand-written table had
    six rows and six documented statuses, and still missed one: `searching` had no arm
    while `ok` -- the undocumented deliberate extension -- had two. Moving `searching`
    from in-flight to failure changed nothing any test could see, so a live web search
    would have been carded as an error with the suite green.

    Scope, stated honestly: this covers the DOCUMENTED statuses only. Swapping the
    whitelist for a blacklist -- the inversion the production comment warns against --
    keeps every case below correct and is observable only for an UNDOCUMENTED status,
    which `test_tools.py::test_server_tool_cards_report_the_tools_own_outcome` covers.
    What this pins is that each documented status reaches the reading its
    classification assigns, so the sets above cannot drift away from the function.
    """
    from open_webui_openrouter_pipe.streaming.streaming_core import _server_tool_status

    expected = _EXPECTED_READING[status]
    assert _server_tool_status({"status": status}) == expected, (
        f"a server tool reporting {status!r} is carded as "
        f"{_server_tool_status({'status': status})!r}, not {expected!r}"
    )


def test_a_status_in_both_sets_is_classified_as_a_failure(monkeypatch):
    """Failure wins over success, so an overlap can never report a broken call as fine.

    The explicit failure branch is unreachable while the two sets are disjoint -- which
    they are today -- so a mutation sweep reports it as dead code and the tempting fix is
    to delete it. Deleting it removes the guarantee: with the branch gone, a status in
    both sets is not `not in SUCCESS`, so it falls through to the error/httpStatus checks
    and a tool that failed can be labelled completed. Open WebUI appends that card
    verbatim onto the persisted assistant message, so the user is told a call worked when
    it did not.

    Driving it through an injected overlap makes the branch live and pins the ordering.
    """
    from open_webui_openrouter_pipe.streaming import streaming_core as sc

    monkeypatch.setattr(sc, "SERVER_TOOL_SUCCESS_STATUSES", frozenset({"completed", "ok", "wobbly"}))
    monkeypatch.setattr(sc, "SERVER_TOOL_FAILURE_STATUSES", frozenset({"incomplete", "failed", "wobbly"}))

    assert sc._server_tool_status({"status": "wobbly"}) == "incomplete", (
        "a status listed as BOTH a failure and a success was classified as a success. "
        "Failure has to win: the alternative labels a broken tool call as one that "
        "worked, on a card the user reads and Open WebUI persists."
    )
