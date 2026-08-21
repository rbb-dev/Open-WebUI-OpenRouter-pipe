"""A choice only some providers accept is either pinned away from the rest, or reported.

`_routing_pins` is right to skip a record OpenRouter gives no routing tag: OpenRouter
documents a null tag as routing being unavailable for that endpoint, and inventing a name
would pin the request at a provider the API cannot address. But the request then goes out
with no `provider.only` at all, so the endpoint that refuses the chosen value is still
reachable -- and the user was told nothing, because the pin is written silently.

Parametrised over a tagged pair and an untagged pair, both narrowing on the same value,
so the two arms cannot be satisfied by one constant: the tagged pair must produce a pin
and no note, the untagged pair must produce a note and no pin.
"""

from __future__ import annotations

from typing import Any

import pytest

from open_webui_openrouter_pipe.integrations.image import (
    _NOTE_TEXT_LIMIT,
    ImageGenerationAdapter,
)


def _records(tagged: bool) -> list[dict[str, Any]]:
    narrow = {
        "provider_slug": "google-vertex/global",
        "supported_parameters": {"resolution": {"type": "enum", "values": ["1K", "2K"]}},
    }
    wide = {
        "provider_slug": "google-ai-studio/global",
        "supported_parameters": {
            "resolution": {"type": "enum", "values": ["1K", "2K", "4K"]}
        },
    }
    if tagged:
        narrow["provider_tag"] = "google-vertex/global"
        wide["provider_tag"] = "google-ai-studio/global"
    return [narrow, wide]


@pytest.mark.parametrize("tagged", [True, False])
@pytest.mark.parametrize("spelling", ["resolution", "size"])
def test_a_narrowing_choice_is_pinned_when_it_can_be_and_reported_when_it_cannot(
    tagged, spelling
):
    provider: dict[str, Any] = {}
    records = _records(tagged)
    notes = ImageGenerationAdapter._pin_accepting_providers(
        provider, records, {spelling: "4K"}
    )

    if tagged:
        assert provider.get("only") == ["google-ai-studio/global"], (
            f"the one endpoint publishing 4K is addressable; pin was {provider!r}"
        )
        assert not notes, f"a pin was written, so there is nothing to report: {notes!r}"
        return

    assert "only" not in provider, (
        "OpenRouter names no endpoint here, so nothing can be pinned; got "
        f"{provider!r}"
    )
    assert notes, (
        f"{spelling}=4K is refused by one of the two endpoints and the request may still "
        "reach it, yet nothing was reported"
    )
    reported = " ".join(note.text for note in notes)
    assert spelling in reported, (
        f"the report has to name the setting that narrowed the routing: {reported!r}"
    )
    assert all(len(note.text) <= _NOTE_TEXT_LIMIT for note in notes), (
        f"a note longer than {_NOTE_TEXT_LIMIT} is truncated before the user reads it: "
        f"{[len(n.text) for n in notes]!r}"
    )


@pytest.mark.parametrize("tier", ["1K", "2K"])
def test_a_choice_every_endpoint_accepts_is_neither_pinned_nor_reported(tier):
    provider: dict[str, Any] = {}
    notes = ImageGenerationAdapter._pin_accepting_providers(
        provider, _records(False), {"resolution": tier}
    )
    assert provider == {} and not notes, (
        f"{tier} is published by both endpoints; there is nothing to say. "
        f"provider={provider!r} notes={notes!r}"
    )
