from __future__ import annotations

import base64
import json

from open_webui_openrouter_pipe.storage.receive_spool import (
    PartRef,
    ReceiveSpool,
    scan_event,
)


def test_a_scanned_event_spools_the_picture_and_keeps_encrypted_reasoning_intact():
    """A scanned event carries a generated picture and an encrypted reasoning detail
    beside it. The picture spools to disk as a part reference; the reasoning detail's
    `data` is opaque base64 with no media signature, so it must survive byte for byte
    as the string it arrived as - the stored and replayed reasoning is that ciphertext,
    and a spool token in its place would break the next turn.
    """
    picture = b"\x89PNG\r\n\x1a\n" + bytes(range(251)) * 3400
    reasoning_ciphertext = base64.b64encode(b"\xab\xcd\xef" * 24000).decode("ascii")
    assert len(base64.b64decode(reasoning_ciphertext)) > 64 * 1024
    event = {
        "id": "resp-1",
        "object": "response.completed",
        "output": [
            {
                "type": "image_generation_call",
                "id": "img-1",
                "status": "completed",
                "result": [{"b64_json": base64.b64encode(picture).decode("ascii")}],
            },
            {
                "type": "reasoning.encrypted",
                "id": "rs-1",
                "data": reasoning_ciphertext,
            },
        ],
        "usage": {"input_tokens": 5},
    }
    blob = json.dumps(event).encode("utf-8")
    assert len(blob) > 1024 * 1024, "the fixture must cross the scan threshold"

    with ReceiveSpool() as spool:
        scanned = scan_event(blob, spool)
        assert scanned is not None, "the event was not scanned at all"
        entries = scanned["output"]
        picture_entry = entries[0]["result"][0]["b64_json"]
        assert isinstance(picture_entry, PartRef), (
            "the picture was not spooled: the image bytes stayed in the parsed event"
        )
        assert picture_entry.read_all() == picture

        kept = entries[1]["data"]
        assert type(kept) is str, (
            f"the encrypted reasoning's data was replaced by {type(kept).__name__}; "
            "stored and replayed reasoning must stay the exact ciphertext string"
        )
        assert kept == event["output"][1]["data"]
        assert base64.b64decode(kept)[:3] == b"\xab\xcd\xef"
