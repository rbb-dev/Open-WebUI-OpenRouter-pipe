"""Four documentation sections that have now been deleted twice.

Each was removed by a rewrite that kept the table-of-contents entry pointing at
it, so the loss showed up only as a dead anchor. These tests fail if a section
disappears, and -- because a heading with nothing under it would satisfy a
presence check -- they also fail if its body is gutted or if a claim the section
exists to make is reverted.

The required strings are drawn from facts that were WRONG in the previous copy of
each section and had to be corrected against the code: which function decides the
image transport, what the video client is actually called, where the video upload
helpers actually live. A regression to the old wording reddens these.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

IMAGE_DOC = "openrouter_image_generation.md"
VIDEO_DOC = "openrouter_video_generation.md"

# (doc, heading level, heading text, minimum non-blank body lines, required strings)
SECTIONS: list[tuple[str, int, str, int, tuple[str, ...]]] = [
    (
        IMAGE_DOC,
        2,
        "## Architecture overview",
        60,
        (
            "uses_dedicated_image_api",
            "ensure_openrouter_image_filter_function_ids",
            "provider.options.<slug>",
            "![alt](file_url)",
        ),
    ),
    (
        VIDEO_DOC,
        2,
        "## Architecture overview",
        40,
        (
            "OpenRouterVideoClient",
            "_run_lifecycle_after_submit",
            "media_relay",
            "_emit_completion",
        ),
    ),
    (
        IMAGE_DOC,
        3,
        "### `_inject_image_modalities()` (orchestrator)",
        15,
        (
            '["image", "text"]',
            '["image"]',
            "output_modalities",
        ),
    ),
    (
        IMAGE_DOC,
        3,
        "### Body validation error mentioning `image_config`",
        6,
        ("image_config",),
    ),
]


def _identify(doc: str, heading: str) -> str:
    where = doc.removeprefix("openrouter_").removesuffix("_generation.md")
    what = heading.lstrip("# ").replace("`", "").split("(")[0].strip()
    return f"{where}-{what.replace(' ', '-')}"


_IDS = [_identify(doc, heading) for doc, _level, heading, _min, _required in SECTIONS]


def _read(doc: str) -> str:
    return (DOCS / doc).read_text(encoding="utf-8")


def _body(text: str, level: int, heading: str) -> list[str]:
    """Lines under `heading`, up to the next heading of the same or higher level."""
    lines = text.splitlines()
    assert heading in lines, f"{heading!r} is not a heading in this document any more"
    start = lines.index(heading)
    stop = len(lines)
    boundary = re.compile(r"^#{1,%d} " % level)
    for offset, line in enumerate(lines[start + 1 :], start=start + 1):
        if boundary.match(line):
            stop = offset
            break
    return lines[start + 1 : stop]


@pytest.mark.parametrize(("doc", "level", "heading", "minimum", "required"), SECTIONS, ids=_IDS)
def test_section_is_present_exactly_once(doc, level, heading, minimum, required):
    lines = _read(doc).splitlines()
    found = [n for n, line in enumerate(lines, 1) if line == heading]
    assert found, (
        f"{doc} no longer contains {heading!r}. It has been deleted twice before; "
        "if it is genuinely obsolete, delete this parametrisation in the same commit "
        "and say why, rather than leaving the table of contents pointing at nothing."
    )
    assert len(found) == 1, (
        f"{doc} contains {heading!r} {len(found)} times (lines {found}). Two copies of "
        "one explanation drift apart; merge them."
    )


@pytest.mark.parametrize(("doc", "level", "heading", "minimum", "required"), SECTIONS, ids=_IDS)
def test_section_body_is_not_a_stub(doc, level, heading, minimum, required):
    body = [line for line in _body(_read(doc), level, heading) if line.strip()]
    assert len(body) >= minimum, (
        f"{heading!r} in {doc} has {len(body)} non-blank lines, under the {minimum} it "
        "needs to say anything. A heading with an empty body passes a presence check "
        "and tells a reader nothing."
    )


@pytest.mark.parametrize(("doc", "level", "heading", "minimum", "required"), SECTIONS, ids=_IDS)
def test_section_still_makes_the_claims_it_exists_for(doc, level, heading, minimum, required):
    body = "\n".join(_body(_read(doc), level, heading))
    missing = [needle for needle in required if needle not in body]
    assert not missing, (
        f"{heading!r} in {doc} no longer mentions {missing}. These are the facts the "
        "section was rewritten to get right; losing one means it is describing code "
        "that no longer behaves that way."
    )


def test_the_tables_of_contents_do_not_point_at_a_deleted_section():
    """Both rewrites left `#architecture-overview` in the contents with no target."""
    dangling = []
    for doc in (IMAGE_DOC, VIDEO_DOC):
        text = _read(doc)
        if "(#architecture-overview)" in text and "\n## Architecture overview\n" not in text:
            dangling.append(doc)
    assert not dangling, (
        f"{dangling} link to #architecture-overview from the table of contents but have "
        "no such heading, so the link resolves to nothing."
    )


def test_the_video_client_is_not_called_by_a_name_that_does_not_exist():
    """`VideoGenClient` was in this document for as long as it existed and is not a
    class anywhere in the pipe -- the client is `OpenRouterVideoClient`. A reader who
    greps for the documented name finds nothing."""
    assert "VideoGenClient" not in _read(VIDEO_DOC), (
        "the video document names a class that does not exist; the client submitting "
        "jobs is OpenRouterVideoClient in integrations/video_client.py"
    )


def test_documented_module_paths_exist():
    """Every ../open_webui_openrouter_pipe/... link in the two documents resolves.

    The upload helpers were documented in `storage/multimodal.py` after they had
    moved to `storage/owui_files.py`, which a path check alone would not have caught
    -- so the names are checked in the file the link points at, below.
    """
    root = DOCS.parent
    broken = []
    for doc in (IMAGE_DOC, VIDEO_DOC):
        for target in re.findall(r"\]\(\.\./(open_webui_openrouter_pipe/[^)#]+)\)", _read(doc)):
            path = target.split("::", 1)[0]
            if not (root / path).exists():
                broken.append(f"{doc} -> {target}")
    assert not broken, "\n".join(broken)


@pytest.mark.parametrize(
    ("symbol", "module"),
    [
        ("def uses_dedicated_image_api", "open_webui_openrouter_pipe/models/registry.py"),
        ("def _inject_image_modalities", "open_webui_openrouter_pipe/requests/orchestrator.py"),
        ("class OpenRouterVideoClient", "open_webui_openrouter_pipe/integrations/video_client.py"),
        ("async def relay_to_public_url", "open_webui_openrouter_pipe/integrations/media_relay.py"),
        (
            "async def upload_to_owui_storage_from_path",
            "open_webui_openrouter_pipe/storage/owui_files.py",
        ),
        (
            "async def _download_remote_url_streaming",
            "open_webui_openrouter_pipe/storage/multimodal.py",
        ),
    ],
)
def test_named_code_is_where_the_documents_say_it_is(symbol, module):
    """Two distinct modules per claim, so no single file satisfies the whole set."""
    source = (DOCS.parent / module).read_text(encoding="utf-8")
    assert symbol in source, (
        f"{module} no longer defines {symbol!r}; the architecture sections name it there"
    )
