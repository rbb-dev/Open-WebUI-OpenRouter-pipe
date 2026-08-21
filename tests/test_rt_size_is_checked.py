"""RED TEAM regression: the Output size box checks what its help says it checks.

IMAGE_SIZE's description reads "A tier sets the same thing as Resolution, **is checked
against the tiers this model publishes**". `_split_image_config` only performs that check
when the typed string is already one of OpenRouter's four recognised tiers; every other
tier-shaped string falls through to the wire verbatim with no note, so the user is neither
checked nor told.

Parametrised over an accepted tier and two rejected strings, so a production function that
always accepts, or always refuses, fails at least one row.
"""

from __future__ import annotations

import pytest

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

RECORD = {
    "provider_slug": "google-vertex/global",
    "provider_tag": "google-vertex/global",
    "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}},
}


@pytest.mark.parametrize(
    ("typed", "publishable"),
    [("2K", True), ("8K", False), ("banana", False)],
)
def test_a_tier_this_model_does_not_publish_is_refused_or_reported(typed, publishable):
    top, _passthrough, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {"size": typed}},
        allowed_passthrough=frozenset(),
        record=RECORD,
        records=[RECORD],
    )
    sent = "size" in top
    if publishable:
        assert sent and not notes, f"{typed!r} is published; it must travel silently: {notes}"
        return
    assert not sent or notes, (
        f"size={typed!r} is not one of the tiers {RECORD['supported_parameters']['resolution']['values']} "
        f"this model publishes, yet it went to the wire as {top} with nothing said "
        f"(notes={notes}) -- the control's own help says a tier is checked"
    )
