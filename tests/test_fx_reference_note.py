"""The note that reports dropped references must name the ones that were actually kept.

Two lists feed `input_references` and they are trimmed from opposite ends. Links the user
typed are placed first and kept in the order given, so the FIRST N survive; attachments
fill whatever room is left and are taken from the tail, so the LAST N survive. The note
said "keeping the most recent" in every case, which is true only of attachments -- and
when the published limit is zero it said it while keeping nothing at all.

Parametrised over the four shapes the trim can take, each asserting the phrase that is
true of it and the phrase that is not, so a note that always says one thing fails.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.integrations.image import ImageGenerationAdapter
from tests.test_image_api_path import _adapter, _KeyPipe, _user_turn_with_images

PIPE_KEY = "openrouter_pipe"


def _record(limit: int | None) -> dict[str, Any]:
    supported: dict[str, Any] = {}
    if limit is not None:
        supported["input_references"] = {"type": "range", "min": 0, "max": limit}
    return {"provider_slug": "openai", "provider_tag": "openai", "supported_parameters": supported}


def _metadata(links: list[str]) -> dict[str, Any]:
    return {PIPE_KEY: {"image_generation": {"reference_urls": links}}}


class _AllowAll:
    async def _is_safe_url(self, _url: str, *, seconds: float = 5.0) -> bool:
        assert seconds > 0, "the address check was handed no time at all"
        return True


def _adapter_allowing_links() -> ImageGenerationAdapter:
    pipe = _KeyPipe("sk-x")
    pipe._multimodal_handler = _AllowAll()  # type: ignore[attr-defined]
    return _adapter(pipe)


@pytest.mark.parametrize(
    ("limit", "links", "attachments", "expected", "forbidden"),
    [
        (2, 0, 5, "keeping the most recent", "first"),
        (2, 5, 0, "keeping the first 2 link(s) you listed", "most recent"),
        (3, 2, 4, "keeping the first 2 link(s) and the 1 most recent attachment(s)", "x"),
        (0, 2, 2, "sending none", "keeping"),
    ],
)
@pytest.mark.asyncio
async def test_the_drop_note_names_the_references_that_survived(
    limit, links, attachments, expected, forbidden
):
    adapter = _adapter_allowing_links()
    notes: list[Any] = []
    urls = [f"https://example.com/{index}.png" for index in range(links)]

    refs = await adapter._reference_payload(
        _user_turn_with_images(attachments),
        _metadata(urls),
        record=_record(limit),
        notes=cast(Any, notes),
    )

    assert len(refs) == limit or len(refs) == links + attachments, (
        f"limit={limit} links={links} attached={attachments} produced {len(refs)} refs"
    )
    reported = " ".join(note.text for note in notes)
    assert expected in reported, (
        f"limit={limit}, {links} link(s) and {attachments} attachment(s) kept "
        f"{[ref['image_url']['url'] for ref in refs]!r}; the note said {reported!r}"
    )
    if forbidden != "x":
        assert forbidden not in reported, (
            f"the note claims {forbidden!r} of references it did not keep: {reported!r}"
        )


@pytest.mark.parametrize(("limit", "links"), [(2, 5), (3, 6)])
@pytest.mark.asyncio
async def test_the_links_that_survive_are_the_ones_the_note_names(limit, links):
    adapter = _adapter_allowing_links()
    notes: list[Any] = []
    urls = [f"https://example.com/{index}.png" for index in range(links)]

    refs = await adapter._reference_payload(
        _user_turn_with_images(0), _metadata(urls), record=_record(limit), notes=cast(Any, notes)
    )

    kept = [ref["image_url"]["url"] for ref in refs]
    assert kept == urls[:limit], (
        f"the first {limit} links are the ones sent, so that is what the note must say; "
        f"sent {kept!r}"
    )
    assert f"keeping the first {limit} link(s)" in " ".join(note.text for note in notes)
