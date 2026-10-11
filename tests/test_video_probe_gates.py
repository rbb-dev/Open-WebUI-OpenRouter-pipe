"""The pipe emits video provider settings in the placement the recorded probe shows the provider honoured."""

from __future__ import annotations

import json
from pathlib import Path


def _video_recording() -> dict:
    return json.loads(
        (
            Path(__file__).parent / "fixtures" / "openrouter_video_passthrough_probe.json"
        ).read_text()
    )


def test_the_emitted_placement_is_the_one_the_recording_shows_the_provider_honoured():
    """The video schema's own guide shows the `parameters` wrapper; the provider ignores it.

    Every placement is accepted with a job id, so status cannot decide this. The recorded
    discriminator is the produced frame: the honoured placement draws the watermark. Without
    this recording the decision rests on a comment, which is how the image side's placement
    was nearly reversed on a reading of the OpenAPI example block.
    """
    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    recorded = _video_recording()
    honoured = [name for name, arm in recorded["arms"].items() if arm["watermark_rendered"]]
    assert honoured == ["flat"], (
        f"the recording must show exactly one placement taking effect; got {honoured}"
    )

    knob = {"watermark": True}
    emitted = VideoGenerationAdapter._normalise_provider_options({"seed": knob})

    assert emitted == recorded["arms"]["flat"]["request_provider"]["options"], (
        f"the pipe must emit the placement the provider honoured. got {emitted!r}"
    )
    assert emitted != recorded["arms"]["wrapped"]["request_provider"]["options"], (
        "re-introducing the parameters wrapper produces the shape the recording shows was "
        "accepted and silently ignored"
    )
