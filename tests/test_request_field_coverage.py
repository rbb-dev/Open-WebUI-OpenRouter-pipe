"""No field of either request format may be missing from the reachability partition.

`callback_url` was unreachable for a sound reason that nobody had written down, and the
video reference list accepted audio and video assets that nobody had noticed. Both look
the same from inside the code: a name in OpenRouter's request format that appears nowhere
here. This compares the partition against the field lists recorded from those formats, so
the two cases become distinguishable and a field added later fails a check instead of
joining the second case in silence.

The fixture is the recording. Re-record it when OpenRouter's request formats change, the
same way the endpoint contracts in this directory are re-recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_webui_openrouter_pipe.integrations.request_fields import (
    IMAGE_FIELD_GAPS,
    IMAGE_FIELD_ROUTES,
    IMAGE_REQUEST_FIELDS,
    VIDEO_FIELD_GAPS,
    VIDEO_FIELD_ROUTES,
    VIDEO_REQUEST_FIELDS,
)

RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "openrouter_request_schema_fields.json").read_text()
)


@pytest.mark.parametrize(
    ("kind", "routes", "gaps", "listed"),
    [
        ("image", IMAGE_FIELD_ROUTES, IMAGE_FIELD_GAPS, IMAGE_REQUEST_FIELDS),
        ("video", VIDEO_FIELD_ROUTES, VIDEO_FIELD_GAPS, VIDEO_REQUEST_FIELDS),
    ],
)
def test_the_partition_covers_the_recorded_request_format_exactly(kind, routes, gaps, listed):
    recorded = set(RECORDED[kind]["properties"])
    assert recorded, f"the recording for {kind} is empty, so this gate asserts nothing"

    covered = set(routes) | set(gaps)
    assert covered == recorded, (
        f"{kind}: unlisted={sorted(recorded - covered)} invented={sorted(covered - recorded)}. "
        "A field OpenRouter defines and nothing here mentions is indistinguishable from "
        "one deliberately left alone, which is how two of them went unnoticed for months."
    )
    assert listed == covered, (
        f"{kind}: the exported field set is the two halves put together, not a third list "
        "beside them; a copy could pass this comparison while the halves it names do not"
    )


@pytest.mark.parametrize(
    ("kind", "routes", "gaps"),
    [
        ("image", IMAGE_FIELD_ROUTES, IMAGE_FIELD_GAPS),
        ("video", VIDEO_FIELD_ROUTES, VIDEO_FIELD_GAPS),
    ],
)
def test_no_field_is_both_reached_and_unreachable(kind, routes, gaps):
    assert not (set(routes) & set(gaps)), kind
    assert all(text.strip() for text in (*routes.values(), *gaps.values())), kind


@pytest.mark.parametrize(("kind", "gaps"), [("video", VIDEO_FIELD_GAPS)])
def test_every_gap_carries_a_reason_long_enough_to_be_one(kind, gaps):
    """A one-word reason is a label, not an explanation, and the point of the list is
    that a reader can tell a decision from an oversight without reading the code."""
    assert gaps, f"{kind} claims no gaps at all; say so by removing this row, not by emptying it"
    for field, reason in gaps.items():
        assert len(reason.split()) >= 12, f"{kind}.{field}: {reason!r}"


@pytest.mark.parametrize("kind", ["image", "video"])
def test_a_required_field_is_never_listed_as_a_gap(kind):
    """A field the format demands cannot be one this deployment declines to send.

    Every recorded required field is checked, not one named here: naming one made the
    test a constant that a constant in the recording satisfied, and the video recording
    was missing `prompt` for exactly as long as nobody compared the two.
    """
    gaps = IMAGE_FIELD_GAPS if kind == "image" else VIDEO_FIELD_GAPS
    required = RECORDED[kind]["required"]
    assert required, f"{kind} records no required fields, so this gate asserts nothing"
    assert not (set(required) & set(gaps)), sorted(set(required) & set(gaps))


def test_the_video_adapter_reads_the_partition_rather_than_a_copy_of_it():
    """The adapter carried its own list of the same twelve names with nothing comparing
    them, so one could gain a field the other never heard about."""
    from open_webui_openrouter_pipe.integrations.video import (
        _DOCUMENTED_TOP_LEVEL_VIDEO_FIELDS,
    )

    assert _DOCUMENTED_TOP_LEVEL_VIDEO_FIELDS == VIDEO_REQUEST_FIELDS


@pytest.mark.parametrize(("kind", "field"), [("video", "callback_url")])
def test_a_named_gap_really_is_absent_from_what_the_pipe_sends(kind, field):
    """The reason has to describe the code. A field listed as unreachable that the
    adapter does send would be a false record, which is worse than none."""
    module = (
        "open_webui_openrouter_pipe/integrations/image.py"
        if kind == "image"
        else "open_webui_openrouter_pipe/integrations/video.py"
    )
    source = (Path(__file__).resolve().parents[1] / module).read_text()
    assert f'payload["{field}"]' not in source
    assert f'"{field}":' not in source
