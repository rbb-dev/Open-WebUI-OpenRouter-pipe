"""An attachment already in Open WebUI storage must cost what its bytes cost.

`_budget_shape` measures the string sitting in an `input_file` block's payload key. For
a stored attachment that key holds a ~36-character identifier, so a 1 MB PDF was charged
27 bytes -- 0 document tokens -- while the identical bytes inline were charged 8,462.
The bytes really do travel: `inline_internal_responses_input_files_inplace` swaps the
identifier for the full base64 on the way to the wire, but that runs on the `model_dump`
copy, AFTER the budget, and `body.input` keeps the reference for every later tool round.

The fix resolves the reference ONCE per request, in async code, and hands the resulting
`{reference string -> (size, media_type, filename)}` map to the synchronous budget. A
reference the map cannot resolve stays at zero -- deliberately, because a guess is worse
than nothing in both directions: an over-charge silently drops a tool call and its result
as a pair, an under-charge produces a recoverable provider error.
"""

from __future__ import annotations

import json
import logging
import tracemalloc
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.api.transforms import ResponsesBody
from open_webui_openrouter_pipe.core.context_budget import (
    _CHARS_PER_TOKEN_HEURISTIC,
    BudgetOutcome,
    _DOCUMENT_BYTES_PER_TOKEN,
    apply_live_tool_output_budget,
    estimate_serialized_chars,
    is_tool_omission_stub,
)
from open_webui_openrouter_pipe.models.registry import ModelFamily
from open_webui_openrouter_pipe.storage import owui_files
from open_webui_openrouter_pipe.storage.owui_files import (
    index_referenced_file_payloads,
)

_FILE_ID = "9f2c1a7b-3e4d-4f5a-8b6c-000000000001"
_INTERNAL_URL = f"/api/v1/files/{_FILE_ID}/content"
_SECOND_ID = "9f2c1a7b-3e4d-4f5a-8b6c-000000000002"
_SECOND_URL = f"/api/v1/files/{_SECOND_ID}/content"
_THIRD_ID = "9f2c1a7b-3e4d-4f5a-8b6c-000000000003"
_FOUR_MB_DATA_URL = "data:text/plain;base64," + "QQQQ" * 1_000_000
_UNPARSEABLE_INTERNAL_URL = "/api/v1/files/_leading/content"


def _blocks(*blocks: dict) -> list[dict]:
    return [{"type": "message", "role": "user", "content": list(blocks)}]


class _Requester:
    def __init__(self, user_id: str = "user-1", role: str = "user"):
        self.id = user_id
        self.role = role


_OWNER = _Requester()


class _Record:
    def __init__(
        self, file_id: str, size: Any, content_type: str, name: str = "", user_id: str = "user-1"
    ):
        self.id = file_id
        self.user_id = user_id
        self.meta: dict[str, Any] = {"content_type": content_type}
        if size is not None:
            self.meta["size"] = size
        if name:
            self.meta["name"] = name


def _install_files(monkeypatch, records: list[_Record], *, bulk: bool = True):
    """Replace the OWUI Files accessor one seam below the function under test."""
    seen: dict[str, list] = {"bulk": [], "single": []}
    by_id = {record.id: record for record in records}

    class _Files:
        @staticmethod
        async def get_file_by_id(file_id):
            seen["single"].append(file_id)
            return by_id.get(file_id)

    if bulk:
        async def _metadatas(ids, db=None):
            seen["bulk"].append(list(ids))
            return [by_id[i] for i in ids if i in by_id]

        _Files.get_file_metadatas_by_ids = staticmethod(_metadatas)  # type: ignore[attr-defined]

    monkeypatch.setattr(owui_files, "Files", _Files)
    return seen


# ---------------------------------------------------------------- the estimate


@pytest.mark.parametrize("declared_bytes", [1_050_009, 8_400_072])
@pytest.mark.parametrize(
    "reference_key", ["file_id", "file_url", "file_data"], ids=["id", "url", "data"]
)
def test_a_stored_attachment_is_charged_for_the_bytes_it_will_ship(
    declared_bytes: int, reference_key: str
) -> None:
    """The measured defect, asserted from both directions.

    Parametrised over two sizes an order of magnitude apart, so `return 8462` in the rate
    cannot satisfy it, and over all three reference spellings the gateway inlines --
    `file_id`, an internal `file_url`, and an internal URL parked in `file_data` --
    because the swap at dispatch treats all three the same and so must the estimate.
    """
    reference = _FILE_ID if reference_key == "file_id" else _INTERNAL_URL
    items = _blocks({"type": "input_file", reference_key: reference})
    index = {reference: (declared_bytes, "application/pdf", "statement.pdf")}

    unresolved = estimate_serialized_chars(items)
    resolved = estimate_serialized_chars(items, referenced_sizes=index)

    assert unresolved < 200, (
        "an unresolved reference must stay at today's behaviour, charged nothing"
    )
    expected = (declared_bytes // _DOCUMENT_BYTES_PER_TOKEN) * _CHARS_PER_TOKEN_HEURISTIC
    assert abs(resolved - expected) < 200, (
        f"{declared_bytes} stored bytes were charged {resolved} chars, not the ~{expected} "
        "the same bytes cost inline"
    )


@pytest.mark.parametrize(
    ("content_type", "text_rate"),
    [
        ("text/plain", True),
        ("application/json", True),
        ("application/pdf", False),
        ("image/png", False),
    ],
)
def test_the_stored_record_s_own_media_type_picks_the_rate(
    content_type: str, text_rate: bool
) -> None:
    """A reference carries no bytes to sniff, so the record's declared type is the evidence.

    Rows either side of the boundary: text costs 4 chars per token and a document 500
    bytes per token, a 125-fold difference, so a single rate cannot satisfy both.
    """
    declared_bytes = 200_000
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})
    index = {_FILE_ID: (declared_bytes, content_type, "a.dat")}

    charged = estimate_serialized_chars(items, referenced_sizes=index)

    if text_rate:
        assert charged > declared_bytes // 2, (
            f"a {content_type} attachment of {declared_bytes} bytes was charged {charged} "
            "chars -- the document rate under-states text by two orders of magnitude"
        )
    else:
        assert charged < declared_bytes // 50, (
            f"a {content_type} attachment was charged {charged} chars, the text rate"
        )


@pytest.mark.parametrize(
    ("stored_name", "text_rate"), [("notes.txt", True), ("scan.bin", False)]
)
def test_an_untyped_record_falls_back_to_the_name_the_record_stores(
    stored_name: str, text_rate: bool
) -> None:
    """`meta.content_type` is `application/octet-stream` for anything OWUI could not type.

    The block itself usually carries no `filename` at estimation time -- the gateway adds
    it at dispatch, from the same record, long after the budget has run -- so the record's
    own name is the only thing left that separates 200 KB of prose from 200 KB of binary.
    Two rows, opposite verdicts, so dropping the name from the index reddens one of them.
    """
    declared_bytes = 200_000
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})
    index = {_FILE_ID: (declared_bytes, "application/octet-stream", stored_name)}

    charged = estimate_serialized_chars(items, referenced_sizes=index)

    if text_rate:
        assert charged > declared_bytes // 2, (
            f"{stored_name} was charged {charged} chars at the binary rate"
        )
    else:
        assert charged < declared_bytes // 50, (
            f"{stored_name} was charged {charged} chars at the text rate"
        )


def test_a_filename_on_the_block_still_outranks_the_stored_name() -> None:
    """The block is what ships, so what it declares wins where it declares anything."""
    index = {_FILE_ID: (200_000, "application/octet-stream", "scan.bin")}
    with_block_name = estimate_serialized_chars(
        _blocks({"type": "input_file", "file_id": _FILE_ID, "filename": "notes.txt"}),
        referenced_sizes=index,
    )
    assert with_block_name > 100_000, (
        "the block declared a .txt name and the stored .bin name was used instead"
    )


@pytest.mark.parametrize("declared_bytes", [1_000_000, 8_000_000])
def test_one_stored_document_is_charged_once_however_many_keys_name_it(
    declared_bytes: int,
) -> None:
    """`file_id` and an internal `file_url` on one block name one document.

    The gateway inlines exactly one of them and drops the rest, so billing per matching
    key charges a single PDF twice. Two sizes, so a constant cannot satisfy it.
    """
    index = {
        _FILE_ID: (declared_bytes, "application/pdf", "a.pdf"),
        _INTERNAL_URL: (declared_bytes, "application/pdf", "a.pdf"),
    }
    alone = estimate_serialized_chars(
        _blocks({"type": "input_file", "file_id": _FILE_ID}), referenced_sizes=index
    )
    both = estimate_serialized_chars(
        _blocks({"type": "input_file", "file_id": _FILE_ID, "file_url": _INTERNAL_URL}),
        referenced_sizes=index,
    )
    assert both - alone < 100, (
        f"one document on two keys cost {both} against {alone} on one key"
    )


def test_an_external_url_and_an_unindexed_reference_stay_at_zero() -> None:
    """The map is the only authority; nothing else may invent a size.

    An `https://` `file_url` is never inlined by the gateway, so the provider fetches it
    or does not and the pipe ships a short string either way. A reference the index could
    not resolve -- a deleted record, a lookup that failed -- is the same case, and must
    degrade to today's behaviour rather than to a guess or a cap.
    """
    index = {_FILE_ID: (1_050_009, "application/pdf", "a.pdf")}

    external = estimate_serialized_chars(
        _blocks({"type": "input_file", "file_url": "https://example.com/big.pdf"}),
        referenced_sizes=index,
    )
    unknown = estimate_serialized_chars(
        _blocks({"type": "input_file", "file_id": "00000000-dead-beef-0000-000000000000"}),
        referenced_sizes=index,
    )

    assert external < 200, f"an external URL was charged {external} chars"
    assert unknown < 200, f"an unresolved reference was charged {unknown} chars"


def test_estimating_with_an_index_does_not_alter_the_request() -> None:
    """The estimator is handed `body.input` itself, not a copy."""
    import copy

    items = _blocks({"type": "input_file", "file_id": _FILE_ID})
    before = copy.deepcopy(items)
    estimate_serialized_chars(
        items, referenced_sizes={_FILE_ID: (1_000_000, "application/pdf", "a.pdf")}
    )
    assert items == before, "estimating the request rewrote the block it was measuring"


# ------------------------------------------------------------------ the index


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_bytes", [1_050_009, 42])
async def test_the_index_reads_the_size_off_the_stored_record(
    monkeypatch, declared_bytes: int
) -> None:
    """Built from the record, not from the reference string's own length."""
    _install_files(monkeypatch, [_Record(_FILE_ID, declared_bytes, "application/pdf", "s.pdf")])
    items = _blocks(
        {"type": "input_file", "file_id": _FILE_ID},
        {"type": "input_file", "file_url": _INTERNAL_URL},
    )

    index = await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    assert index == {
        _FILE_ID: (declared_bytes, "application/pdf", "s.pdf"),
        _INTERNAL_URL: (declared_bytes, "application/pdf", "s.pdf"),
    }


@pytest.mark.asyncio
async def test_the_index_prefers_the_bulk_metadata_read(monkeypatch) -> None:
    """`get_file_by_id` selects the `data` column, which can hold the file's own bytes.

    One request can reference many attachments, so the per-id read is the fallback and the
    five-column metadata select is the primary. Asserted by which accessor was called.
    """
    other = "9f2c1a7b-3e4d-4f5a-8b6c-000000000002"
    seen = _install_files(
        monkeypatch,
        [
            _Record(_FILE_ID, 1_000, "application/pdf", "a.pdf"),
            _Record(other, 2_000, "application/pdf", "b.pdf"),
        ],
    )
    items = _blocks(
        {"type": "input_file", "file_id": _FILE_ID},
        {"type": "input_file", "file_id": other},
    )

    index = await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    assert index[_FILE_ID][0] == 1_000 and index[other][0] == 2_000
    assert seen["bulk"] == [sorted([_FILE_ID, other])], (
        f"the bulk metadata accessor was called {seen['bulk']}"
    )
    assert seen["single"] == [], (
        f"the per-id blob read ran anyway for {seen['single']}"
    )


@pytest.mark.asyncio
async def test_the_index_falls_back_per_id_when_the_bulk_read_is_absent(
    monkeypatch,
) -> None:
    """`get_file_metadatas_by_ids` does not exist on every supported Open WebUI.

    The manifest requires only 0.9.1, so the fallback is the difference between sizing
    every attachment and sizing none of them on an older deployment.
    """
    seen = _install_files(
        monkeypatch, [_Record(_FILE_ID, 777, "text/plain", "a.txt")], bulk=False
    )
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})

    index = await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    assert index == {_FILE_ID: (777, "text/plain", "a.txt")}
    assert seen["single"] == [_FILE_ID]


@pytest.mark.asyncio
async def test_a_record_with_no_declared_size_degrades_to_zero_and_says_so(
    monkeypatch, caplog
) -> None:
    """An unsizable record must not become a guess, and must not be silent."""
    _install_files(monkeypatch, [_Record(_FILE_ID, None, "application/pdf", "a.pdf")])
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})

    with caplog.at_level(logging.WARNING):
        index = await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    assert index == {}, "a record with no declared size produced a size anyway"
    assert any(_FILE_ID in record.getMessage() for record in caplog.records), (
        "the pipe silently charged a stored attachment nothing"
    )
    assert estimate_serialized_chars(items, referenced_sizes=index) < 200


@pytest.mark.asyncio
async def test_a_bulk_read_that_fails_still_falls_back_per_id(monkeypatch) -> None:
    """The bulk accessor exists on this deployment and the call fails anyway.

    A narrowed `except` around the bulk read turns a transient failure into a request
    where nothing is sized -- silently, because the outer guard returns an empty map for
    both "the read failed" and "there was nothing to read". The fallback is what makes
    those two different, so it is asserted from the failing direction.
    """
    record = _Record(_FILE_ID, 1_050_009, "application/pdf", "s.pdf")

    class _HalfBroken:
        @staticmethod
        async def get_file_metadatas_by_ids(ids, db=None):
            raise RuntimeError("the bulk statement failed")

        @staticmethod
        async def get_file_by_id(file_id):
            return record if file_id == _FILE_ID else None

    monkeypatch.setattr(owui_files, "Files", _HalfBroken)
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})

    index = await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    assert index == {_FILE_ID: (1_050_009, "application/pdf", "s.pdf")}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("block", "expect_read", "expect_charged"),
    [
        ({"file_id": _FILE_ID}, [_FILE_ID], 8),
        ({"file_id": _FILE_ID, "file_url": _SECOND_URL}, [_FILE_ID], 8),
        ({"file_url": _SECOND_URL}, [_SECOND_ID], 32_000),
        ({"file_data": _SECOND_URL, "file_url": _INTERNAL_URL}, [_SECOND_ID], 32_000),
        ({"file_id": "file-provider-123", "file_url": _SECOND_URL}, [], 0),
        ({"file_id": _FILE_ID, "file_data": _FOUR_MB_DATA_URL}, [_FILE_ID], 8),
        ({"file_data": _UNPARSEABLE_INTERNAL_URL, "file_url": _SECOND_URL}, [], 0),
    ],
    ids=[
        "id-alone",
        "id-outranks-url",
        "url-alone",
        "data-outranks-url",
        "a-provider-side-id-skips-the-block",
        "id-outranks-an-inline-payload",
        "an-unparseable-internal-reference-consumes-the-block",
    ],
)
async def test_the_index_resolves_the_one_reference_the_gateway_will_inline(
    monkeypatch, block: dict, expect_read: list, expect_charged: int
) -> None:
    """The budget must charge for the payload the request actually sends, and read no more.

    `inline_internal_responses_input_files_inplace` is an if/elif chain: it resolves at
    most ONE reference per block -- `file_id`, else an internal `file_data`, else an
    internal `file_url` -- and skips the block entirely when `file_id` carries the
    provider-side `file-` prefix. The index collected all three keys and `_budget_shape`
    charged the largest, so a block naming a 1 KB file by id and a 4 MB file by url was
    charged for the 4 MB one the gateway never dispatches.

    The second effect is the reason this is not merely an over-charge: the extra key was
    resolved against Open WebUI storage, so the budget read a file record the request
    never touches and nothing authorises -- `index_referenced_file_payloads` takes no
    user. `expect_read` is therefore asserted directly, not inferred from the charge.

    The rows disagree on both columns, so neither a constant read-set nor a constant
    charge satisfies them.
    """
    seen = _install_files(
        monkeypatch,
        [
            _Record(_FILE_ID, 1_000, "application/pdf", "small.pdf"),
            _Record(_SECOND_ID, 4_000_000, "application/pdf", "huge.pdf"),
        ],
    )
    items = _blocks({"type": "input_file", **block})

    index = await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    read = sorted({i for call in seen["bulk"] for i in call} | set(seen["single"]))
    assert read == sorted(expect_read), (
        f"the budget read {read} from Open WebUI storage; the gateway will resolve "
        f"{sorted(expect_read)} for this block and nothing authorises the rest"
    )
    charged = estimate_serialized_chars(items, referenced_sizes=index)
    assert abs(charged - expect_charged) < 250, (
        f"this block was charged {charged} chars; the reference the gateway will inline "
        f"is worth about {expect_charged}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bulk", "bulk_fails", "stored", "expect_single_reads"),
    [
        (True, False, 3, 0),
        (True, False, 1, 0),
        (False, False, 3, 3),
        (True, True, 3, 3),
    ],
    ids=["bulk-answered-all", "bulk-answered-partially", "no-bulk-accessor", "bulk-raised"],
)
async def test_a_bulk_read_that_returned_is_not_asked_again_id_by_id(
    monkeypatch, bulk: bool, bulk_fails: bool, stored: int, expect_single_reads: int
) -> None:
    """`get_file_metadatas_by_ids` has no limit, so an id it omits does not exist.

    Re-asking for each one individually costs a serialised round trip per missing id,
    in the request path, before the first token -- and the everyday case is a chat whose
    attachment was deleted, which pays it on every turn forever. The fallback still has
    two real jobs: an Open WebUI without the bulk accessor at all (the manifest supports
    0.9.1), and a bulk read that raises, which it does on SQLite past the variable limit.

    The middle row is the one that matters: the bulk read answered and simply did not
    carry two of the three ids. The four rows expect three different counts, so no
    constant satisfies them.
    """
    ids = [_FILE_ID, _SECOND_ID, _THIRD_ID]
    records = [
        _Record(file_id, 1_000, "application/pdf", "a.pdf") for file_id in ids[:stored]
    ]
    if bulk_fails:
        by_id = {r.id: r for r in records}
        calls: dict[str, list] = {"bulk": [], "single": []}

        class _Failing:
            @staticmethod
            async def get_file_metadatas_by_ids(requested, db=None):
                raise RuntimeError("the bulk statement failed")

            @staticmethod
            async def get_file_by_id(file_id):
                calls["single"].append(file_id)
                return by_id.get(file_id)

        monkeypatch.setattr(owui_files, "Files", _Failing)
        seen = calls
    else:
        seen = _install_files(monkeypatch, records, bulk=bulk)

    items = _blocks(*({"type": "input_file", "file_id": file_id} for file_id in ids))

    await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    assert len(seen["single"]) == expect_single_reads, (
        f"{len(seen['single'])} per-id reads followed a bulk read that "
        f"{'raised' if bulk_fails else 'was absent' if not bulk else 'returned'}; "
        f"{expect_single_reads} were warranted"
    )


@pytest.mark.asyncio
async def test_the_unsized_record_warning_latches_on_its_reason(monkeypatch, caplog) -> None:
    """A `warn_level` cause names a reason, not an instance.

    Keyed by file id, every distinct unsized attachment warned at WARNING and the latch
    dict grew an entry per id for the life of the worker -- an unbounded key space fed by
    user data. A record whose `meta['size']` is missing is a documented class with its own
    valve, not an exotic case. Every other latch in this module keys by a reason, and the
    one twenty lines above this call site keys by the constant "bulk-metadata".

    Two records with two ids must therefore produce one WARNING and one DEBUG, not two
    WARNINGs -- so a constant level cannot satisfy both halves.
    """
    _install_files(
        monkeypatch,
        [
            _Record(_FILE_ID, None, "application/pdf", "a.pdf"),
            _Record(_SECOND_ID, None, "application/pdf", "b.pdf"),
        ],
    )
    items = _blocks(
        {"type": "input_file", "file_id": _FILE_ID},
        {"type": "input_file", "file_id": _SECOND_ID},
    )

    with caplog.at_level(logging.DEBUG, logger="test"):
        index = await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        )

    assert index == {}
    unsized = [r for r in caplog.records if "no readable declared size" in r.getMessage()]
    levels = sorted(r.levelno for r in unsized)
    assert levels == [logging.DEBUG, logging.WARNING], (
        f"two unsized records logged {[logging.getLevelName(x) for x in levels]}; the "
        "latch must fire once for the reason and fall to DEBUG for every later instance"
    )


@pytest.mark.asyncio
async def test_a_failing_lookup_never_reaches_the_request_it_was_estimating(
    monkeypatch,
) -> None:
    """This runs inside `process_request`; raising here 500s the whole chat turn.

    An estimate is an optimisation. Losing it costs the accuracy the rest of this file
    asserts; raising costs the user their answer.
    """

    class _Exploding:
        @staticmethod
        async def get_file_metadatas_by_ids(ids, db=None):
            raise RuntimeError("database is on fire")

        @staticmethod
        async def get_file_by_id(file_id):
            raise RuntimeError("database is still on fire")

    monkeypatch.setattr(owui_files, "Files", _Exploding)
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})

    assert await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        ) == {}


@pytest.mark.asyncio
async def test_a_record_that_cannot_be_read_never_reaches_the_request(
    monkeypatch,
) -> None:
    """The row arrives and then raises when it is touched.

    SQLAlchemy raises on attribute access for an instance detached from its session, so a
    record that loaded fine can still explode at `.meta`. That happens BELOW every accessor
    guard, in the loop that reads the size off the row -- the only thing standing between
    it and a 500 is the guard wrapping the whole index build.
    """

    class _Detached:
        id = _FILE_ID

        @property
        def meta(self):
            raise RuntimeError("Instance is not bound to a Session")

    class _Files:
        @staticmethod
        async def get_file_metadatas_by_ids(ids, db=None):
            return [_Detached()]

        @staticmethod
        async def get_file_by_id(file_id):
            return None

    monkeypatch.setattr(owui_files, "Files", _Files)
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})

    assert await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        ) == {}


@pytest.mark.asyncio
async def test_an_openrouter_file_id_is_not_looked_up_in_open_webui(monkeypatch) -> None:
    """`file-`-prefixed ids belong to the provider; the gateway skips them too."""
    seen = _install_files(monkeypatch, [])
    items = _blocks({"type": "input_file", "file_id": "file-abc123"})

    assert await index_referenced_file_payloads(
            items, logging.getLogger("test"), user=_OWNER
        ) == {}
    assert seen["bulk"] == [] and seen["single"] == []


# ------------------------------------------------------------------- the path


def test_the_declared_map_never_reaches_the_wire() -> None:
    """`ResponsesBody` is `extra="allow"`, so an undeclared attribute survives `model_dump`.

    `api_model` is the proof: it is set as an extra and `streaming_core` has to pop it out
    of the payload by hand. A declared field with `exclude=True` cannot be forgotten that
    way, and the contrast below is what shows the exclusion is doing the work rather than
    pydantic quietly dropping an unknown attribute.
    """
    body = ResponsesBody.model_validate(
        {
            "model": "test/model",
            "api_model": "vendor/real-model",
            "input": [],
            "input_file_sizes": {_FILE_ID: (1_000_000, "application/pdf", "a.pdf")},
        }
    )

    internal = {
        name
        for name, field in ResponsesBody.model_fields.items()
        if getattr(field, "exclude", False)
    }
    assert internal == {
        "input_file_sizes",
        "budget_futility_notified",
        "budget_reported_call_ids",
        "budget_chars_per_token",
    }, (
        f"the set of pipe-internal ResponsesBody fields changed to {sorted(internal)}. "
        "Each one is bookkeeping that must never be dispatched; add it here deliberately "
        "so the exclusion below covers it."
    )

    assert body.input_file_sizes is not None
    dumped = body.model_dump()
    dumped_no_none = body.model_dump(exclude_none=True)
    for name in internal:
        assert name not in dumped, f"{name} is dispatched as a top-level request key"
        assert name not in dumped_no_none, f"{name} survives exclude_none=True"
    assert "api_model" in body.model_dump(), (
        "the contrast is gone: an undeclared extra no longer survives model_dump, so this "
        "test would pass even if input_file_sizes were left undeclared"
    )


@pytest.mark.parametrize("declared_bytes", [0, 3_000_000])
def test_the_sanitiser_spends_the_budget_the_index_reports(
    pipe_instance, declared_bytes: int
) -> None:
    """The replay pass reads the map off the body; five call sites share this one seam.

    Two rows from one mechanism: with the attachment sized, the request no longer has room
    for the replayed tool result and it is stubbed; with nothing sized, it is delivered
    whole. A budget that ignored the map cannot produce both.
    """
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 4_000}, "context_length": 4_000}}
    )
    body = ResponsesBody.model_validate(
        {
            "model": "test/model",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": _FILE_ID}],
                },
                {"type": "function_call", "call_id": "c1", "name": "lookup",
                 "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "R" * 8_000},
            ],
        }
    )
    if declared_bytes:
        body.input_file_sizes = {_FILE_ID: (declared_bytes, "application/pdf", "a.pdf")}

    outcome = _sanitize_request_input(pipe_instance, body)

    assert outcome is not None
    shipped = [
        item
        for item in cast(list, body.input)
        if item.get("type") == "function_call_output"
    ]
    assert bool(outcome.omitted_call_ids) is bool(declared_bytes), (
        f"a {declared_bytes}-byte attachment produced {sorted(outcome.omitted_call_ids)}"
    )
    assert is_tool_omission_stub(shipped[0]["output"]) is bool(declared_bytes)


@pytest.mark.parametrize("declared_bytes", [0, 3_000_000])
def test_the_live_pass_spends_the_budget_the_index_reports(declared_bytes: int) -> None:
    """The same map, on the pass that decides what a freshly executed tool returns."""
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 4_000}, "context_length": 4_000}}
    )
    existing = _blocks({"type": "input_file", "file_id": _FILE_ID}) + [
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"}
    ]
    outputs = [{"type": "function_call_output", "call_id": "c1", "output": "R" * 8_000}]
    index = (
        {_FILE_ID: (declared_bytes, "application/pdf", "a.pdf")} if declared_bytes else None
    )

    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=existing,
        model_id="test/model",
        referenced_sizes=index,
    ).omitted_call_ids

    assert bool(omitted) is bool(declared_bytes)


@pytest.mark.asyncio
@pytest.mark.parametrize("declared_bytes", [0, 3_000_000])
async def test_the_streaming_loop_carries_the_index_and_not_the_field(
    monkeypatch, pipe_instance_async, declared_bytes: int
) -> None:
    """End to end: the loop's live pass honours the map, and the map is not dispatched.

    The observable for the live pass is the NOTIFICATION, not the dispatched request. The
    sanitiser's replay pass runs over the whole of `body.input` immediately afterwards and
    would stub the same result from the same map, so asserting on the request cannot tell
    the two passes apart -- the live site could stop reading the map entirely and the wire
    would look identical. What only the live pass produces is `omitted_call_ids`, which is
    what tells the user which tool result the model never saw and what gates citation
    harvesting.

    The wire is asserted in the same test, on the DISPATCHED BODY, because that is the one
    place a stray top-level key is visible.
    """
    from open_webui_openrouter_pipe.pipe import Pipe

    pipe = pipe_instance_async
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 1_200}, "context_length": 1_200}}
    )
    body = ResponsesBody.model_validate(
        {
            "model": "test/model",
            "stream": True,
            "input": _blocks({"type": "input_file", "file_id": _FILE_ID}),
        }
    )
    if declared_bytes:
        body.input_file_sizes = {_FILE_ID: (declared_bytes, "application/pdf", "a.pdf")}
    valves = pipe.valves.model_copy(
        update={"TOOL_EXECUTION_MODE": "Pipeline", "MAX_FUNCTION_CALL_LOOPS": 2}
    )

    events_by_call = [
        [
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "lookup",
                            "arguments": "{}",
                        }
                    ],
                    "usage": {},
                },
            }
        ],
        [
            {"type": "response.output_text.delta", "delta": "Done."},
            {"type": "response.completed", "response": {"output": [], "usage": {}}},
        ],
    ]
    captured: list[dict[str, Any]] = []
    call_index = 0

    async def streaming(self, session, request_body, **_kwargs):
        nonlocal call_index
        idx = call_index
        call_index += 1
        captured.append(json.loads(json.dumps(request_body)))
        for event in events_by_call[idx]:
            yield event

    async def mock_execute(calls, registry):
        return [{"type": "function_call_output", "call_id": "call-1", "output": "y" * 3_000}]

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)
    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", mock_execute)

    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._streaming_handler._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "test"}, "chat_id": "chat-1", "message_id": "msg-1"},
        tools={"lookup": {"callable": lambda **_kwargs: "ok"}},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert len(captured) >= 2, "the tool continuation never dispatched"
    assert all("input_file_sizes" not in request for request in captured), (
        "the size map was dispatched to OpenRouter as a top-level request key"
    )
    announced = [
        event["data"]["content"]
        for event in emitted
        if isinstance(event, dict)
        and event.get("type") == "notification"
        and isinstance(event.get("data"), dict)
        and "did not receive" in str(event["data"].get("content", ""))
    ]
    assert bool(announced) is bool(declared_bytes), (
        f"a {declared_bytes}-byte attachment on the request produced {announced}; the "
        "live pass decides what the user is told the model never read"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("declared_bytes", "found"),
    [(1_050_009, True), (8_400_072, True), (None, False)],
    ids=["1MB", "8MB", "record-gone"],
)
async def test_the_orchestrator_sizes_every_attachment_before_the_budget_runs(
    monkeypatch, declared_bytes, found: bool
) -> None:
    """The single seam: one index per request, built where the code can still await.

    Driven through `process_request` from an Open WebUI `file` message block, so the whole
    path is real -- the transformer turning it into an `input_file` block, the index
    resolving that block's identifier against the stored record, and the map arriving on
    the body the budget is about to read. `_sanitize_request_input` is replaced with a
    capture that stops the turn there: one seam BELOW the wiring under test, so the wiring
    itself still runs.

    Two sizes so a constant cannot satisfy it, and a third row where the record is gone,
    which must produce no entry rather than a guess.
    """
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.pipe import Pipe
    from open_webui_openrouter_pipe.requests import orchestrator as orchestrator_module

    class _Session:
        pass

    class _Stop(Exception):
        pass

    records = (
        [_Record(_FILE_ID, declared_bytes, "application/pdf", "statement.pdf")]
        if declared_bytes is not None
        else []
    )
    _install_files(monkeypatch, records)

    async def _requester(user_id, logger):
        return _Requester(user_id)

    monkeypatch.setattr(orchestrator_module, "get_user_by_id", _requester)

    captured: dict[str, Any] = {}

    def _capture(pipe, body):
        captured["body"] = body
        raise _Stop()

    monkeypatch.setattr(orchestrator_module, "_sanitize_request_input", _capture)

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"

    async def emitter(event):
        return None

    try:
        with pytest.raises(_Stop):
            await pipe._ensure_request_orchestrator().process_request(
                body={
                    "model": "test/model",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "file", "file": {"file_id": _FILE_ID}},
                                {"type": "text", "text": "summarise this"},
                            ],
                        }
                    ],
                    "stream": False,
                },
                __user__={"id": "user-1"},
                __request__=None,
                __event_emitter__=emitter,
                __event_call__=None,
                __metadata__={},
                __tools__=None,
                __task__=None,
                __task_body__=None,
                valves=pipe.valves,
                session=cast(Any, _Session()),
                openwebui_model_id="test/model",
                pipe_identifier="test-pipe",
                allowlist_norm_ids={"test/model"},
                enforced_norm_ids=set(),
                catalog_norm_ids={"test/model"},
                features={},
            )
    finally:
        await pipe.close()

    body = captured["body"]
    blocks = [
        block
        for item in cast(list, body.input)
        for block in (item.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "input_file"
    ]
    assert blocks == [{"type": "input_file", "file_id": _FILE_ID}], (
        f"the transformer did not emit the reference this test is about: {blocks}"
    )
    expected = (
        {_FILE_ID: (declared_bytes, "application/pdf", "statement.pdf")} if found else {}
    )
    assert body.input_file_sizes == expected, (
        f"the body handed to the budget carried {body.input_file_sizes}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("futile", "needed_tokens", "limit_tokens", "expect_notice"),
    [(True, 6_274, 800, True), (True, 9_000, 1_200, True), (False, 6_274, 800, False)],
    ids=["hopeless", "hopeless-other-numbers", "fits"],
)
async def test_the_pre_dispatch_pass_tells_the_user_when_the_request_is_hopeless(
    monkeypatch, futile: bool, needed_tokens: int, limit_tokens: int, expect_notice: bool
) -> None:
    """The verdict every request produces had no way to reach the user.

    `_sanitize_request_input` runs on the orchestrator's pre-dispatch pass -- the one pass
    every request makes, tools or not -- and it was called for its side effects with its
    return value dropped on the floor. The only place the futility verdict was announced
    sat inside the tool loop, so a plain chat whose attachments alone exceed the window had
    every tool result reduced to a placeholder and was told nothing at all.

    Stubbed one seam below the wiring under test: the pass's verdict is supplied, and what
    is asserted is that the orchestrator reads it and that the message carries THAT
    verdict's numbers. The two hopeless rows differ in both numbers, so a hardcoded notice
    satisfies neither, and the third row makes silence the right answer somewhere.
    """
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.pipe import Pipe
    from open_webui_openrouter_pipe.requests import orchestrator as orchestrator_module

    class _Stop(Exception):
        pass

    def _verdict(pipe, body):
        return BudgetOutcome(
            frozenset(),
            futile,
            needed_tokens * _CHARS_PER_TOKEN_HEURISTIC,
            limit_tokens * _CHARS_PER_TOKEN_HEURISTIC,
        )

    def _halt(*_args, **_kwargs):
        raise _Stop()

    monkeypatch.setattr(orchestrator_module, "_sanitize_request_input", _verdict)
    monkeypatch.setattr(orchestrator_module, "apply_context_transforms", _halt)

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    class _Session:
        pass

    try:
        with pytest.raises(_Stop):
            await pipe._ensure_request_orchestrator().process_request(
                body={
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "summarise this"}],
                    "stream": False,
                },
                __user__={"id": "user-1"},
                __request__=None,
                __event_emitter__=emitter,
                __event_call__=None,
                __metadata__={},
                __tools__=None,
                __task__=None,
                __task_body__=None,
                valves=pipe.valves,
                session=cast(Any, _Session()),
                openwebui_model_id="test/model",
                pipe_identifier="test-pipe",
                allowlist_norm_ids={"test/model"},
                enforced_norm_ids=set(),
                catalog_norm_ids={"test/model"},
                features={},
            )
    finally:
        await pipe.close()

    notices = [
        event["data"]["content"]
        for event in emitted
        if isinstance(event, dict)
        and event.get("type") == "notification"
        and isinstance(event.get("data"), dict)
        and "reduced to a placeholder" in str(event["data"].get("content", ""))
    ]
    assert len(notices) == (1 if expect_notice else 0), (
        f"a pre-dispatch verdict of futile={futile} produced {len(notices)} notices"
    )
    if expect_notice:
        assert f"about {needed_tokens} tokens" in notices[0], (
            f"the notice does not carry this verdict's figure: {notices[0]}"
        )
        assert f"{limit_tokens}-token limit" in notices[0], (
            f"the notice does not carry this verdict's limit: {notices[0]}"
        )


@pytest.mark.parametrize("declared_mb", [16, 128])
def test_charging_a_reference_does_not_materialise_the_payload(declared_mb: int) -> None:
    """The estimator counts the bytes; it must never build them.

    `_budget_shape` used to represent a charged payload as a filler string of exactly the
    charged length, which was harmless while the payload was already in `body.input` as
    base64. Once a stored reference supplies the size, a 36-character `file_id` made the
    estimator allocate the whole declared file -- then `json.dumps` copied it. Measured
    before the fix: a 100 MB `text/csv` reference charged 104,857,689 chars in 243 ms with
    a 340.8 MB peak, and the budget runs about five times per turn.

    Both halves are asserted. The charge must still track the declared size, so a fix that
    caps the filler -- which caps the charge with it, undoing the point of sizing stored
    attachments at all -- fails the first assertion; the allocation bound catches the
    regression. Two sizes an order of magnitude apart, so no constant satisfies either.
    """
    size = declared_mb * 1024 * 1024
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})
    index = {_FILE_ID: (size, "text/csv", "data.csv")}

    tracemalloc.start()
    try:
        charged = estimate_serialized_chars(items, referenced_sizes=index)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert abs(charged - size) < 512, (
        f"a {declared_mb} MB reference was charged {charged} chars, not the ~{size} its "
        "bytes cost at the text rate"
    )
    assert peak < 1_000_000, (
        f"charging a {declared_mb} MB reference allocated {peak / 1e6:.1f} MB; the request "
        f"holds {len(json.dumps(items))} characters for it"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task", "expect_notices"),
    [(None, 1), ("title_generation", 0), ("tags_generation", 0)],
    ids=["the-user-s-own-request", "title-generation", "tags-generation"],
)
async def test_the_futility_notice_is_silent_on_background_tasks(
    monkeypatch, task, expect_notices: int
) -> None:
    """Open WebUI's housekeeping calls are not the user asking a question.

    Title, tag and follow-up generation each arrive as their own `process_request` with
    their own `ResponsesBody`, so the per-body latch dedupes nothing across them: one user
    message could produce four identical toasts. Worse, the budget for a task request is
    computed against the *task* model, whose window is far smaller -- so a user on a large
    model whose request fits comfortably was told it exceeded "this model's" limit and
    advised to start a new chat.

    The neighbouring notification in the same function is already written
    `if direct_uploads_warnings and not use_task_model_adapter`; this follows it. Three
    rows, two distinct counts, so no constant satisfies them.
    """
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.pipe import Pipe
    from open_webui_openrouter_pipe.requests import orchestrator as orchestrator_module

    class _Stop(Exception):
        pass

    class _Session:
        pass

    def _verdict(pipe, body):
        return BudgetOutcome(frozenset(), True, 6_274 * _CHARS_PER_TOKEN_HEURISTIC,
                             800 * _CHARS_PER_TOKEN_HEURISTIC)

    def _halt(*_args, **_kwargs):
        raise _Stop()

    monkeypatch.setattr(orchestrator_module, "_sanitize_request_input", _verdict)
    monkeypatch.setattr(orchestrator_module, "apply_context_transforms", _halt)

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    try:
        with pytest.raises(_Stop):
            await pipe._ensure_request_orchestrator().process_request(
                body={
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "summarise this"}],
                    "stream": False,
                },
                __user__={"id": "user-1"},
                __request__=None,
                __event_emitter__=emitter,
                __event_call__=None,
                __metadata__={},
                __tools__=None,
                __task__=task,
                __task_body__=None,
                valves=pipe.valves,
                session=cast(Any, _Session()),
                openwebui_model_id="test/model",
                pipe_identifier="test-pipe",
                allowlist_norm_ids={"test/model"},
                enforced_norm_ids=set(),
                catalog_norm_ids={"test/model"},
                features={},
            )
    finally:
        await pipe.close()

    notices = [
        event["data"]["content"]
        for event in emitted
        if isinstance(event, dict)
        and event.get("type") == "notification"
        and isinstance(event.get("data"), dict)
        and "reduced to a placeholder" in str(event["data"].get("content", ""))
    ]
    assert len(notices) == expect_notices, (
        f"__task__={task!r} produced {len(notices)} futility notices, expected "
        f"{expect_notices}; a background task is not the user asking a question, and its "
        "budget is measured against a different model"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_model_max", "expect_reservation"),
    [(True, 65_536), (False, None)],
    ids=["valve-on", "valve-off"],
)
async def test_the_reply_allowance_is_decided_before_the_budget_reads_it(
    monkeypatch, use_model_max: bool, expect_reservation
) -> None:
    """The budget cannot spend a reservation that has not been filled in yet.

    `USE_MODEL_MAX_OUTPUT_TOKENS` fills `max_output_tokens` from the catalog. That fill
    used to sit ~160 lines BELOW the pre-dispatch budget pass, so the first pass budgeted
    against the whole window while every later pass in the streaming loop budgeted against
    the window minus the reservation -- the limit moved mid-turn. Moving the fill above
    the budget is invisible to every other test: reverting the move left all 830 tests in
    the orchestrator and budget suites green.

    Asserted on the value the budget pass actually receives. The two rows differ, so a fix
    that always fills or never fills satisfies neither, and the filled figure is half the
    window rather than the catalog's 100,352 -- the pipe must not claim the provider's
    largest possible answer as this request's reservation.
    """
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.pipe import Pipe
    from open_webui_openrouter_pipe.requests import orchestrator as orchestrator_module

    class _Stop(Exception):
        pass

    class _Session:
        pass

    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "context_length": 131_072,
                "max_completion_tokens": 100_352,
                "full_model": {"context_length": 131_072, "max_completion_tokens": 100_352},
            }
        }
    )
    seen: dict[str, Any] = {}

    def _capture(pipe, body):
        seen["reservation"] = getattr(body, "max_output_tokens", None)
        raise _Stop()

    monkeypatch.setattr(orchestrator_module, "_sanitize_request_input", _capture)

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
    valves = pipe.valves.model_copy(update={"USE_MODEL_MAX_OUTPUT_TOKENS": use_model_max})

    async def emitter(event):
        return None

    try:
        with pytest.raises(_Stop):
            await pipe._ensure_request_orchestrator().process_request(
                body={
                    "model": "test/model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
                __user__={"id": "user-1"},
                __request__=None,
                __event_emitter__=emitter,
                __event_call__=None,
                __metadata__={},
                __tools__=None,
                __task__=None,
                __task_body__=None,
                valves=valves,
                session=cast(Any, _Session()),
                openwebui_model_id="test/model",
                pipe_identifier="test-pipe",
                allowlist_norm_ids={"test/model"},
                enforced_norm_ids=set(),
                catalog_norm_ids={"test/model"},
                features={},
            )
    finally:
        await pipe.close()

    assert seen["reservation"] == expect_reservation, (
        f"with USE_MODEL_MAX_OUTPUT_TOKENS={use_model_max} the budget pass saw a "
        f"reservation of {seen['reservation']}, not {expect_reservation}; the fill must "
        "happen before the budget reads it, and must not claim the provider's maximum"
    )


@pytest.mark.asyncio
async def test_one_futility_notice_per_turn_across_both_dispatch_paths(monkeypatch) -> None:
    """Both passes that can announce futility share one body, so the user hears it once.

    The verdict is reached twice on a turn with tools: once by the orchestrator's
    pre-dispatch pass and again by every iteration of the streaming loop. The latch that
    keeps that to one message lives on `ResponsesBody` rather than in the loop's closure
    for exactly this reason -- a closure-local flag silences the loop but not the
    orchestrator, so the user gets two.

    Nothing covered it: the loop test calls `_run_streaming_loop` directly so the
    orchestrator pass never runs, and the orchestrator test halts before the loop starts.
    This drives the real `process_request` with `_sanitize_request_input` left UNSTUBBED,
    so both real emitters see the same body, and counts what reaches the one emitter.

    Asserted on the notices the user receives, not on the latch field: a closure-local
    latch that still sets `budget_futility_notified` would satisfy a field assertion while
    emitting twice.
    """
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.pipe import Pipe

    class _Session:
        pass

    ModelFamily.set_dynamic_specs(
        {"test.model": {"context_length": 200, "full_model": {"context_length": 200}}}
    )

    rounds = iter(range(1, 200))

    async def streaming(self, session, request_body, **_kwargs):
        n = next(rounds)
        if n == 1:
            yield {
                "type": "response.completed",
                "response": {
                    "output": [
                        {"type": "function_call", "call_id": "c1", "name": "lookup",
                         "arguments": "{}"}
                    ],
                    "usage": {},
                },
            }
        else:
            yield {"type": "response.output_text.delta", "delta": "Done."}
            yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    async def mock_execute(calls, registry):
        return [
            {"type": "function_call_output", "call_id": call.get("call_id"),
             "output": "y" * 4_000}
            for call in calls
        ]

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", streaming)

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
    monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", mock_execute)
    valves = pipe.valves.model_copy(
        update={"TOOL_EXECUTION_MODE": "Pipeline", "MAX_FUNCTION_CALL_LOOPS": 3}
    )

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    try:
        await pipe._ensure_request_orchestrator().process_request(
            body={
                "model": "test/model",
                "messages": [{"role": "user", "content": "x" * 6_000}],
                "stream": True,
            },
            __user__={"id": "user-1"},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={"chat_id": "chat-1", "message_id": "msg-1"},
            __tools__={"lookup": {"callable": lambda **_kwargs: "ok"}},
            __task__=None,
            __task_body__=None,
            valves=valves,
            session=cast(Any, _Session()),
            openwebui_model_id="test/model",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"test.model"},
            enforced_norm_ids={"test.model"},
            catalog_norm_ids={"test.model"},
            features={},
        )
    finally:
        await pipe.close()

    assert next(rounds) > 2, (
        "the streaming loop never ran, so only one pass could have spoken and this row "
        "asserts nothing about the latch"
    )
    futile = [
        event["data"]["content"]
        for event in emitted
        if isinstance(event, dict)
        and event.get("type") == "notification"
        and isinstance(event.get("data"), dict)
        and "reduced to a placeholder" in str(event["data"].get("content", ""))
    ]
    assert len(futile) == 1, (
        f"a turn that reached the futility verdict on both the pre-dispatch pass and the "
        f"tool loop emitted {len(futile)} notices; the verdict belongs to the turn"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("owner", "expect_indexed", "expect_charged"),
    [("user-1", True, 32_000), ("someone-else", False, 0)],
    ids=["the-requesters-own-file", "not-the-requesters-file"],
)
async def test_a_file_the_requester_cannot_read_is_charged_nothing(
    monkeypatch, owner: str, expect_indexed: bool, expect_charged: int
) -> None:
    """The budget must not price a file the dispatch path will refuse to send.

    `index_referenced_file_payloads` took no user and authorised nothing, so any file id
    a request named was resolved against storage and its declared size folded into the
    budget. That size then reaches the requester twice: it steers which tool results
    survive, and it is summed into the "needs about N tokens" notice. With attacker-
    controlled padding, any monotone function of the size is a full oracle for a file
    the requester cannot read -- measured 9,700x of the reported figure coming from
    someone else's 40 MB attachment.

    It is a correctness bug in the same motion: the gateway refuses an unauthorised file
    at dispatch, so its bytes never travel and charging them can declare a turn hopeless
    over a payload that was never in the request.

    Asserted on the charge, not on a log line, because the charge is what the user sees.
    """
    _install_files(
        monkeypatch,
        [_Record(_FILE_ID, 4_000_000, "application/pdf", "big.pdf", user_id=owner)],
    )
    items = _blocks({"type": "input_file", "file_id": _FILE_ID})

    index = await index_referenced_file_payloads(
        items, logging.getLogger("test"), user=_OWNER
    )

    assert bool(index) is expect_indexed, (
        f"a file owned by {owner!r} was {'indexed' if index else 'skipped'} for a "
        "requester who is user-1; only files the requester may read may be priced"
    )
    charged = estimate_serialized_chars(items, referenced_sizes=index)
    assert abs(charged - expect_charged) < 250, (
        f"the reference was charged {charged} chars, not ~{expect_charged}; an "
        "unauthorised file must contribute nothing to any number the user is shown"
    )


@pytest.mark.parametrize("tools", [0, 15, 30])
def test_the_budget_counts_the_tools_the_request_carries(tools: int) -> None:
    """`tools` and `instructions` travel on the wire and were never metered.

    The budget only ever looked at `body.input`, so a request's tool schemas -- which the
    provider tokenises like everything else -- were invisible to it. Measured against an
    8k window: 2,348 uncounted characters at 5 tools, 6,843 at 15, and 13,593 at 30, the
    last being 42.5% of the window the budget believed it was managing.

    The overhead is derived from the body itself (`model_dump(exclude={"input"})`) rather
    than from a hand-written list of keys, so a new top-level field cannot silently escape
    it. Three tool counts, so no constant satisfies the row set.
    """
    from open_webui_openrouter_pipe.requests.sanitizer import _request_overhead_chars

    schema = {
        "type": "function",
        "name": "search_documents",
        "description": "Search the corpus and return ranked excerpts with citations.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    }
    payload: dict[str, Any] = {
        "model": "test/model",
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": "hi"}]}],
    }
    if tools:
        payload["tools"] = [dict(schema, name=f"{schema['name']}_{i}") for i in range(tools)]
        payload["instructions"] = "You are a careful assistant."
    body = ResponsesBody.model_validate(payload)

    overhead = _request_overhead_chars(body)

    wire = len(json.dumps(body.model_dump(exclude_none=True), ensure_ascii=False, default=str))
    seen = len(json.dumps(body.input, ensure_ascii=False))
    assert overhead >= (wire - seen) - 15, (
        f"{tools} tool schemas put {wire - seen} chars on the wire and the budget "
        f"accounted for {overhead}; everything the request carries has to be counted. "
        "The 15-char allowance is the JSON punctuation around the `input` key itself, "
        "which is not attributable to the tools and is measured at 11."
    )
    if tools:
        assert overhead > 1_000, (
            f"{tools} tool schemas were charged {overhead} chars; they are not free"
        )


def test_the_tools_a_request_carries_reach_the_budget_not_just_the_helper() -> None:
    """Measuring the overhead is worthless unless the budget spends against it.

    A helper that returns the right number and a floor that ignores it is the shape this
    repo has been bitten by before: the call site reads correctly and the behaviour never
    changes. Asserted on `irreducible_chars` from the real sanitiser pass, with and
    without tool schemas on the same conversation, so the only way to satisfy it is for
    the overhead to arrive where the verdict is computed.
    """
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    ModelFamily.set_dynamic_specs(
        {"test.model": {"context_length": 200_000, "full_model": {"context_length": 200_000}}}
    )
    schema = {
        "type": "function",
        "name": "search_documents",
        "description": "Search the corpus and return ranked excerpts with citations.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
    conversation = [
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "r" * 400},
    ]

    class _Pipe:
        logger = logging.getLogger("test")

    floors = {}
    for label, tools in (("bare", None), ("with-tools", [dict(schema, name=f"t{i}") for i in range(30)])):
        payload: dict[str, Any] = {"model": "test/model", "input": list(conversation)}
        if tools:
            payload["tools"] = tools
        body = ResponsesBody.model_validate(payload)
        outcome = _sanitize_request_input(cast(Any, _Pipe()), body)
        assert outcome is not None
        floors[label] = outcome.irreducible_chars

    assert floors["with-tools"] - floors["bare"] > 4_000, (
        f"the same conversation floored at {floors['bare']} chars bare and "
        f"{floors['with-tools']} with 30 tool schemas attached; the schemas travel on the "
        "wire, so the budget that decides what to trim has to see them"
    )
