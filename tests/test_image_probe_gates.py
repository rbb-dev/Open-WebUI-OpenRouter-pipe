from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _recording(name: str) -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures" / f"openrouter_image_{name}.json").read_text()
    )


def test_the_emitted_placement_is_the_one_the_recording_shows_reaching_the_provider():
    from open_webui_openrouter_pipe.integrations.provider_options import merge_provider_options

    flat = _recording("passthrough_flat")
    wrapped = _recording("passthrough_wrapped")
    knob = flat["request"]["provider"]["options"]["recraft"]

    assert flat["status"] == 400, (
        "the flat placement is believed correct because the provider rejected the bogus "
        "value; a recording that shows 200 would mean the value never reached it"
    )
    assert flat["response"]["error"]["metadata"]["provider_name"], (
        "provider_name is what proves the provider itself answered rather than the router"
    )
    assert wrapped["status"] == 200 and wrapped["response"]["data"], (
        "the wrapped placement is believed inert because the same bogus value generated an "
        "image instead of an error; a recording that shows 400 would reverse the decision"
    )

    emitted = merge_provider_options({}, "recraft", knob)
    assert emitted == flat["request"]["provider"], (
        f"the pipe must emit the placement the recording shows reaching the provider. "
        f"emitted={emitted!r}"
    )
    assert emitted != wrapped["request"]["provider"], (
        "re-introducing the parameters wrapper would produce the shape OpenRouter silently "
        "discarded"
    )
