from open_webui_openrouter_pipe import Pipe


def test_format_final_status_description_includes_cost_tokens_and_tps(pipe_instance):
    pipe = pipe_instance
    usage = {
        "cost": 0.012345,
        "input_tokens": 120,
        "output_tokens": 40,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 5},
    }

    description = pipe._format_final_status_description(
        elapsed=3.21,
        total_usage=usage,
        valves=pipe.valves,
        stream_duration=2.0,
    )

    assert description.startswith("Time: 3.21s  20.0 tps")
    assert "Cost $0.012345" in description
    assert "Total tokens: 160 (Input: 120, Output: 40, Cached: 20, Reasoning: 5)" in description


def test_format_final_status_description_respects_disabled_flag(pipe_instance):
    pipe = pipe_instance
    valves = pipe.Valves(SHOW_FINAL_USAGE_STATUS=False)

    description = pipe._format_final_status_description(
        elapsed=4.5,
        total_usage={},
        valves=valves,
        stream_duration=None,
    )

    assert description == "Thought for 4.5 seconds"
