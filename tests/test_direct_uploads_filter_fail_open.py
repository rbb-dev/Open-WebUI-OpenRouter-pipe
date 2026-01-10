from __future__ import annotations

from filters.openrouter_direct_uploads_toggle import Filter


def test_direct_uploads_filter_fails_open_when_model_lacks_file_capability():
    filt = Filter()

    files = [
        {
            "id": "file_1",
            "type": "file",
            "name": "example.pdf",
            "size": 123,
            "content_type": "application/pdf",
        }
    ]
    body = {"files": list(files)}
    metadata: dict = {}
    user = {"valves": filt.UserValves(DIRECT_FILES=True, DIRECT_AUDIO=False, DIRECT_VIDEO=False)}
    model = {
        "info": {
            "meta": {
                "openrouter_pipe": {
                    "capabilities": {"file_input": False, "audio_input": False, "video_input": False}
                }
            }
        }
    }

    result = filt.inlet(body, __metadata__=metadata, __user__=user, __model__=model)

    assert result["files"] == files
    pipe_meta = metadata.get("openrouter_pipe")
    assert isinstance(pipe_meta, dict)
    assert pipe_meta.get("direct_uploads") is None
    warnings = pipe_meta.get("direct_uploads_warnings")
    assert isinstance(warnings, list)
    assert any("Direct file uploads not supported" in str(msg) for msg in warnings)

