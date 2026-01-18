from open_webui_openrouter_pipe import _extract_feature_flags


def test_extract_feature_flags_uses_flat_metadata_shape():
    metadata = {"features": {"web_search": True, "code_interpreter": False}}
    assert _extract_feature_flags(metadata) == {
        "web_search": True,
        "code_interpreter": False,
    }


def test_extract_feature_flags_does_not_assume_nested_by_pipe_id():
    metadata = {"features": {"my.pipe": {"web_search": True}}}
    assert _extract_feature_flags(metadata) == {"my.pipe": {"web_search": True}}

