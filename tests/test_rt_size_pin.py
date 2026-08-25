"""RED TEAM regression: a tier reaches routing the same way whichever control set it.

`size` accepting a tier is the pipe's own documented equivalence -- IMAGE_SIZE's help text
says a tier "is checked against" whatever limit the model publishes for output size.
`_split_image_config` honours it (it validates a tier `size` against the `resolution`
descriptor). `_records_accepting` does not, so the provider pin that keeps a
narrowed tier away from a provider that lacks it is applied for one spelling and not the
other.

Parametrised over both spellings so a production function returning a constant pin, or no
pin, satisfies at most one row. The seam stubbed is the endpoint contract (two provider
records), one level below the subject.
"""

from __future__ import annotations

from typing import Any

import pytest

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter

VERTEX: dict[str, Any] = {
    "provider_slug": "google-vertex/global",
    "provider_tag": "google-vertex/global",
    "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}},
}
STUDIO: dict[str, Any] = {
    "provider_slug": "google-ai-studio/global",
    "provider_tag": "google-ai-studio/global",
    "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}},
}
RECORDS = [VERTEX, STUDIO]


@pytest.mark.parametrize("spelling", ["resolution", "size"])
@pytest.mark.parametrize("tier", ["4K", "2K"])
def test_a_tier_is_routed_by_who_publishes_it_whichever_control_set_it(spelling, tier):
    fitted, _passthrough, notes = ImageGenerationAdapter._split_image_config(
        {"image_config": {spelling: tier}},
        allowed_passthrough=frozenset(),
        record=VERTEX,
        records=RECORDS,
    )
    assert fitted.get(spelling) == tier, (
        f"{spelling}={tier!r} was refused before routing could be considered: {notes}"
    )

    provider: dict[str, Any] = {}
    ImageGenerationAdapter._pin_accepting_providers(provider, RECORDS, fitted)

    publishers = [
        record["provider_tag"]
        for record in RECORDS
        if tier in record["supported_parameters"]["resolution"]["values"]
    ]
    reachable = provider.get("only") or [r["provider_tag"] for r in RECORDS]
    non_publishers = [tag for tag in reachable if tag not in publishers]
    assert not non_publishers, (
        f"{spelling}={tier!r} is published by {publishers}, but the request may still be "
        f"served by {non_publishers}; provider block was {provider}"
    )
