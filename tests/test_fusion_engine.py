
import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.core.config import _PIPE_METADATA_KEY, Valves
from open_webui_openrouter_pipe.core.errors import OpenRouterAPIError
from open_webui_openrouter_pipe.core.fusion_defaults import FusionRunPlan
from open_webui_openrouter_pipe.requests.fusion_engine import (
    FusionCollector,
    FusionInnerInvocation,
    FusionMemberResult,
    aggregate_sources,
    build_inner_metadata,
    build_inner_valves,
    parse_analysis,
    run_fusion_member,
)

VALID_ANALYSIS = {"consensus": ["c"], "contradictions": [], "partial_coverage": [],
                  "unique_insights": [], "blind_spots": []}


@pytest_asyncio.fixture
async def orchestrator_and_pipe():
    import logging

    from open_webui_openrouter_pipe.requests.orchestrator import RequestOrchestrator

    pipe = Pipe()
    logger = logging.getLogger("test_fusion_engine")
    logger.setLevel(logging.DEBUG)
    orchestrator = RequestOrchestrator(pipe, logger)
    yield orchestrator, pipe
    await pipe.close()


class TestBuildInnerMetadata:
    def test_marker_set_and_ids_dropped(self):
        outer = {
            "chat_id": "chat-1", "message_id": "msg-1", "session_id": "sess-1",
            "model": {"id": "openrouter.openrouter/fusion"},
            _PIPE_METADATA_KEY: {"server_tools": {"web_search": {}}},
            "other": "keep",
        }
        inner = build_inner_metadata(outer)
        assert inner[_PIPE_METADATA_KEY]["fusion_inner"] is True
        assert "chat_id" not in inner and "message_id" not in inner and "model" not in inner
        assert inner["other"] == "keep"

    def test_outer_untouched_deep(self):
        outer: dict[str, Any] = {_PIPE_METADATA_KEY: {"server_tools": {"web_search": {}}}, "chat_id": "c"}
        inner = build_inner_metadata(outer)
        inner[_PIPE_METADATA_KEY]["server_tools"]["web_fetch"] = {}
        assert "web_fetch" not in outer[_PIPE_METADATA_KEY]["server_tools"]

    def test_none_metadata(self):
        assert build_inner_metadata(None)[_PIPE_METADATA_KEY]["fusion_inner"] is True

    def test_unpicklable_shared_objects_survive(self):
        import threading

        lock = threading.Lock()
        outer = {
            "chat_id": "c",
            "tools": {"t": {"callable": lambda: None, "client": lock}},
            "mcp_clients": [lock],
            "files": [{"id": "f1"}],
        }
        inner = build_inner_metadata(outer)
        assert inner[_PIPE_METADATA_KEY]["fusion_inner"] is True
        assert "chat_id" not in inner
        assert inner["tools"] is outer["tools"]
        assert inner["mcp_clients"] is outer["mcp_clients"]
        assert inner["files"] is outer["files"]

    @pytest.mark.asyncio
    async def test_live_future_in_metadata_survives(self):
        from types import SimpleNamespace

        fut = asyncio.get_running_loop().create_future()
        holder = SimpleNamespace(session_future=fut)
        outer = {"tools": {"m": {"client": holder}}}
        inner = build_inner_metadata(outer)
        assert inner["tools"]["m"]["client"] is holder
        fut.cancel()

    def test_per_member_bases_isolated_without_touching_shared(self):
        shared_tools = {"t": {"spec": {"name": "t"}}}
        outer = {"tools": shared_tools, _PIPE_METADATA_KEY: {"server_tools": {"web_search": {}}}}
        first = build_inner_metadata(outer)
        second = build_inner_metadata(outer)
        first[_PIPE_METADATA_KEY]["server_tools"]["web_fetch"] = {}
        first["extra"] = 1
        assert "web_fetch" not in second[_PIPE_METADATA_KEY]["server_tools"]
        assert "web_fetch" not in outer[_PIPE_METADATA_KEY]["server_tools"]
        assert "extra" not in outer
        assert "extra" not in second
        assert outer["tools"] is shared_tools
        assert first["tools"] is shared_tools
        assert second["tools"] is shared_tools


class TestBuildInnerValves:
    def test_loop_cap_and_cost_dump(self):
        valves = Valves(MAX_FUNCTION_CALL_LOOPS=25, COSTS_REDIS_DUMP=True)
        inner = build_inner_valves(valves, max_tool_calls=8)
        assert inner.MAX_FUNCTION_CALL_LOOPS == 8
        assert inner.COSTS_REDIS_DUMP is False
        assert valves.MAX_FUNCTION_CALL_LOOPS == 25

    def test_admin_cap_wins_when_lower(self):
        assert build_inner_valves(Valves(MAX_FUNCTION_CALL_LOOPS=3), max_tool_calls=16).MAX_FUNCTION_CALL_LOOPS == 3


class TestFusionCollector:
    @pytest.mark.asyncio
    async def test_content_reasoning_usage_sources(self):
        q: asyncio.Queue = asyncio.Queue()
        col = FusionCollector("m/x", q)
        await col({"type": "chat:message:delta", "data": {"content": "he"}})
        await col({"type": "fusion_inner:reasoning.delta", "data": {"delta": "think"}})
        await col({"type": "chat:completion", "data": {"usage": {"cost": 0.1}}})
        await col({"type": "source", "data": {"source": {"url": "https://a.example", "name": "A"}}})
        await col("junk")
        assert col.usage == {"cost": 0.1}
        assert col.sources == [{"url": "https://a.example", "title": "A"}]
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items == [("delta", "m/x", "he"), ("reasoning", "m/x", "think")]


class TestParseAnalysis:
    def test_plain_and_fenced(self):
        assert parse_analysis(json.dumps(VALID_ANALYSIS)) == VALID_ANALYSIS
        assert parse_analysis("```json\n" + json.dumps(VALID_ANALYSIS) + "\n```") == VALID_ANALYSIS

    def test_wrong_keys_and_junk(self):
        bad = dict(VALID_ANALYSIS, extra=1)
        assert parse_analysis(json.dumps(bad)) is None
        assert parse_analysis("no json") is None
        assert parse_analysis("") is None


class TestAggregateSources:
    def test_dedupe_and_utm_strip(self):
        r1 = FusionMemberResult("a", "x", None, False, None,
                                ({"url": "https://a.example?utm_source=openai", "title": "A"},))
        r2 = FusionMemberResult("b", "y", None, False, None,
                                ({"url": "https://a.example", "title": "A2"},
                                 {"url": "https://c.example", "title": "C"}))
        assert aggregate_sources([r1, r2]) == [
            {"url": "https://a.example", "title": "A"},
            {"url": "https://c.example", "title": "C"},
        ]


def _prepare_pipe(pipe):
    pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
    pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(return_value=({}, []))


def _invocation(orchestrator, pipe, valves, *, messages=None, enforced=None, catalog=None):
    return FusionInnerInvocation(
        orchestrator=orchestrator,
        messages=messages or [{"role": "user", "content": "What is OpenRouter?"}],
        outer_model_id="openrouter/fusion",
        user={"id": "u1"},
        request=None,
        event_call=None,
        metadata={"chat_id": "outer-chat", "message_id": "outer-msg"},
        tools={},
        valves=valves,
        session=cast(Any, object()),
        pipe_identifier="test-pipe",
        allowlist_norm_ids=set(),
        enforced_norm_ids=set(enforced or set()),
        catalog_norm_ids=set(catalog or set()),
        features={},
        user_id="u1",
    )


def _capture_stream(captured, events=None, fail_first_with=None):
    state = {"calls": 0}

    async def fake_stream(self, session, request_body, **kwargs):
        state["calls"] += 1
        captured.append(dict(request_body))
        if fail_first_with is not None and state["calls"] == 1:
            raise fail_first_with
        for event in events or [
            {"type": "response.created", "response": {"model": request_body.get("model")}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "member answer"},
            {"type": "response.output_text.done", "output_index": 0, "text": "member answer"},
            {"type": "response.completed",
             "response": {"output": [], "usage": {"cost": 0.2, "total_tokens": 9},
                          "model": request_body.get("model")}},
        ]:
            yield event

    return fake_stream


class TestRunFusionMemberReentry:
    async def _run(self, orchestrator_and_pipe, monkeypatch, *,
                   model="openai/gpt-5", registry=None, bypass=True, enforced=None,
                   catalog=None, fail_first_with=None, supports_fc=False,
                   valves=None, server_tools_config=None):
        orchestrator, pipe = orchestrator_and_pipe
        _prepare_pipe(pipe)
        captured: list[dict] = []
        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request",
                            _capture_stream(captured, fail_first_with=fail_first_with))
        if supports_fc:
            from open_webui_openrouter_pipe.models.registry import ModelFamily
            monkeypatch.setattr(ModelFamily, "supports",
                                classmethod(lambda cls, cap, m: cap == "function_calling"))
        valves = valves or Valves()
        inv = _invocation(orchestrator, pipe, valves,
                          enforced=enforced, catalog=catalog)
        if registry is not None:
            inv.tools = registry
        if server_tools_config is None:
            server_tools_config = (
                {"web_search": {"max_results": 5, "search_context_size": "medium"},
                 "web_fetch": {"engine": "auto"}},
                [],
            )
        result = await run_fusion_member(
            pipe, inv, model=model,
            messages=inv.messages, system_prompt="PANEL PROMPT",
            max_tool_calls=8, live_queue=None, bypass_restrictions=bypass,
            server_tools_config=server_tools_config,
        )
        return result, captured

    @pytest.mark.asyncio
    async def test_member_payload_carries_web_server_tools(
        self, orchestrator_and_pipe, monkeypatch
    ):
        result, captured = await self._run(orchestrator_and_pipe, monkeypatch)
        assert result.failed is False
        assert captured, "no outbound request captured"
        tools = captured[0].get("tools") or []
        tool_types = [t.get("type") for t in tools]
        assert "openrouter:web_search" in tool_types
        assert "openrouter:web_fetch" in tool_types
        ws = next(t for t in tools if t.get("type") == "openrouter:web_search")
        assert ws.get("parameters", {}).get("max_results") == 5

    @pytest.mark.asyncio
    async def test_no_web_tools_when_filter_inactive(
        self, orchestrator_and_pipe, monkeypatch
    ):
        result, captured = await self._run(
            orchestrator_and_pipe, monkeypatch, server_tools_config=({}, []),
        )
        assert result.failed is False
        tool_types = [t.get("type") for t in (captured[0].get("tools") or [])]
        assert "openrouter:web_search" not in tool_types

    @pytest.mark.asyncio
    async def test_function_tool_specs_reach_member_payload(
        self, orchestrator_and_pipe, monkeypatch
    ):
        async def mytool(**kwargs):
            return "ok"

        registry = {"mytool": {"type": "function", "callable": mytool,
                               "spec": {"name": "mytool", "parameters": {"type": "object", "properties": {}}}}}
        result, captured = await self._run(
            orchestrator_and_pipe, monkeypatch,
            registry=registry, supports_fc=True,
        )
        assert result.failed is False
        names = [t.get("name") for t in (captured[0].get("tools") or [])]
        assert "mytool" in names

    @pytest.mark.asyncio
    async def test_reasoning_effort_ladder_heals_member(
        self, orchestrator_and_pipe, monkeypatch
    ):
        message = "Invalid reasoning.effort value 'xhigh'. Supported values are: 'low', 'medium', 'high'."
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            upstream_message=message,
            provider_raw={"error": {"param": "reasoning.effort", "code": "unsupported_value",
                                    "type": "invalid_request_error", "message": message}},
        )
        from open_webui_openrouter_pipe.models.registry import ModelFamily

        monkeypatch.setattr(ModelFamily, "supported_parameters",
                            classmethod(lambda cls, m: frozenset({"reasoning"})))
        valves = Valves(REASONING_EFFORT="xhigh")
        result, captured = await self._run(
            orchestrator_and_pipe, monkeypatch,
            model="~google/gemini-flash-latest", fail_first_with=error, valves=valves,
        )
        assert captured[0].get("reasoning", {}).get("effort") == "xhigh"
        assert captured[-1].get("reasoning", {}).get("effort") != "xhigh"
        assert len(captured) >= 2
        assert result.failed is False
        assert result.content == "member answer"

    @pytest.mark.asyncio
    async def test_restriction_rejects_without_bypass(
        self, orchestrator_and_pipe, monkeypatch
    ):
        result, captured = await self._run(
            orchestrator_and_pipe, monkeypatch,
            bypass=False, enforced={"other/model"}, catalog={"other/model"},
        )
        assert result.failed is True
        assert captured == []

    @pytest.mark.asyncio
    async def test_restriction_bypassed_for_preset_members(
        self, orchestrator_and_pipe, monkeypatch
    ):
        result, _captured = await self._run(
            orchestrator_and_pipe, monkeypatch,
            bypass=True, enforced={"other/model"}, catalog={"other/model"},
        )
        assert result.failed is False

    @pytest.mark.asyncio
    async def test_request_id_restored_and_snapshots_suppressed(
        self, orchestrator_and_pipe, monkeypatch
    ):
        import open_webui_openrouter_pipe.streaming.streaming_core as sc_mod
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        snapshots: list[Any] = []

        async def spy(*a, **k):
            snapshots.append((a, k))

        monkeypatch.setattr(sc_mod, "maybe_dump_costs_snapshot", spy)
        valves = Valves(COSTS_REDIS_DUMP=True)
        SessionLogger.request_id.set("outer-req-1")
        result, _captured = await self._run(orchestrator_and_pipe, monkeypatch, valves=valves)
        assert result.failed is False
        assert SessionLogger.request_id.get() == "outer-req-1"
        for args, kwargs in snapshots:
            valves_arg = kwargs.get("valves") or next(
                (a for a in args if hasattr(a, "COSTS_REDIS_DUMP")), None)
            assert valves_arg is not None and valves_arg.COSTS_REDIS_DUMP is False


class TestMemberFailureReasons:
    @pytest.mark.asyncio
    async def test_member_failure_reason_sanitized_and_logged(self, orchestrator_and_pipe, caplog):
        import logging as _logging

        _orchestrator, pipe = orchestrator_and_pipe

        class _Orch:
            async def process_request(self, body, *args, **kwargs):
                raise TypeError("cannot pickle '_asyncio.Future' object")

        inv = _invocation(_Orch(), pipe, Valves())
        with caplog.at_level(_logging.WARNING):
            result = await run_fusion_member(
                pipe, inv, model="m/x", messages=[{"role": "user", "content": "q"}],
                system_prompt="P", max_tool_calls=4, live_queue=None,
                bypass_restrictions=True, server_tools_config=None,
            )
        assert result.failed is True
        reason = result.fail_reason or ""
        assert "TypeError" not in reason
        assert "pickle" not in reason
        assert "interrupted" not in reason
        assert reason == "the model response could not be processed"
        assert any("fusion member" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_answer_maps_to_no_answer_reason(self, orchestrator_and_pipe):
        _orchestrator, pipe = orchestrator_and_pipe

        class _Orch:
            async def process_request(self, body, *args, **kwargs):
                kwargs["outcome_sink"]["error_occurred"] = False
                return ""

        inv = _invocation(_Orch(), pipe, Valves())
        result = await run_fusion_member(
            pipe, inv, model="m/x", messages=[{"role": "user", "content": "q"}],
            system_prompt="P", max_tool_calls=4, live_queue=None,
            bypass_restrictions=True, server_tools_config=None,
        )
        assert result.failed is True
        assert result.fail_reason == "the model returned no answer"

    @pytest.mark.asyncio
    async def test_api_error_maps_to_call_failed(self, orchestrator_and_pipe):
        _orchestrator, pipe = orchestrator_and_pipe

        class _Orch:
            async def process_request(self, body, *args, **kwargs):
                raise OpenRouterAPIError(status=502, reason="upstream said no")

        inv = _invocation(_Orch(), pipe, Valves())
        result = await run_fusion_member(
            pipe, inv, model="m/x", messages=[{"role": "user", "content": "q"}],
            system_prompt="P", max_tool_calls=4, live_queue=None,
            bypass_restrictions=True, server_tools_config=None,
        )
        assert result.failed is True
        assert result.fail_reason == "the model call failed"
        assert "upstream said no" not in (result.fail_reason or "")

    @pytest.mark.asyncio
    async def test_generic_exception_maps_to_before_completing(self, orchestrator_and_pipe):
        _orchestrator, pipe = orchestrator_and_pipe

        class _Orch:
            async def process_request(self, body, *args, **kwargs):
                raise RuntimeError("socket exploded at 10.0.0.7")

        inv = _invocation(_Orch(), pipe, Valves())
        result = await run_fusion_member(
            pipe, inv, model="m/x", messages=[{"role": "user", "content": "q"}],
            system_prompt="P", max_tool_calls=4, live_queue=None,
            bypass_restrictions=True, server_tools_config=None,
        )
        assert result.failed is True
        assert result.fail_reason == "the model call failed before completing"
        assert "socket" not in (result.fail_reason or "")


class TestInnerMessageCopies:
    @pytest.mark.asyncio
    async def test_member_body_shares_content_objects(self, orchestrator_and_pipe):
        _orchestrator, pipe = orchestrator_and_pipe
        captured: list[dict] = []

        class _Orch:
            async def process_request(self, body, *args, **kwargs):
                captured.append(body)
                kwargs["outcome_sink"]["error_occurred"] = False
                return "answer"

        content_blocks = [{"type": "text", "text": "hello"}]
        messages = [{"role": "user", "content": content_blocks}]
        inv = _invocation(_Orch(), pipe, Valves(), messages=messages)
        result = await run_fusion_member(
            pipe, inv, model="openai/gpt-5", messages=messages,
            system_prompt="P", max_tool_calls=4, live_queue=None,
            bypass_restrictions=True, server_tools_config=None,
        )
        assert result.failed is False
        body_messages = captured[0]["messages"]
        assert body_messages[0]["role"] == "system"
        assert body_messages[1] is not messages[0]
        assert body_messages[1]["content"] is content_blocks

    @pytest.mark.asyncio
    async def test_synthesis_messages_share_content_objects(self, monkeypatch, pipe_instance_async):
        import open_webui_openrouter_pipe.requests.fusion_engine as fe

        seen: list[Any] = []

        async def fake(pipe, invocation, **kw):
            seen.append(kw.get("messages"))
            model = kw["model"]
            if model == "j/x":
                return FusionMemberResult(
                    model=model, content=json.dumps(VALID_ANALYSIS), usage=None,
                    failed=False, fail_reason=None,
                )
            return FusionMemberResult(
                model=model, content="draft", usage=None, failed=False, fail_reason=None,
            )

        monkeypatch.setattr(fe, "run_fusion_member", fake)
        from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
        monkeypatch.setattr(FilterManager, "collect_installed_web_tools_config",
                            AsyncMock(return_value=None))
        content_blocks = [{"type": "text", "text": "Q?"}]
        inv = FusionInnerInvocation(
            orchestrator=Mock(), messages=[{"role": "user", "content": content_blocks}],
            outer_model_id="openrouter/fusion", user={"id": "u"}, request=None,
            event_call=None, metadata={}, tools={}, valves=pipe_instance_async.valves,
            session=object(), pipe_identifier="p", user_id="u",
        )
        plan = FusionRunPlan(("m/a",), "j/x", "j/x", 8)
        async for _ev in fe.run_internal_fusion(pipe_instance_async, invocation=inv, plan=plan):
            pass
        synth_messages = seen[-1]
        assert synth_messages is not None
        assert synth_messages[0] is not inv.messages[0]
        assert synth_messages[0]["content"] is content_blocks
        assert synth_messages[-1]["role"] == "system"


class TestRunInternalFusion:
    def _fake_member(self, behaviors):
        calls = []

        async def fake(pipe, invocation, **kw):
            model = kw["model"]
            calls.append(kw)
            queue = kw.get("live_queue")
            spec = behaviors.get(model) or behaviors.get("*") or {}
            for kind, text in spec.get("live", []):
                if queue is not None:
                    await queue.put((kind, model, text))
            return FusionMemberResult(
                model=model, content=spec.get("content", ""), usage=spec.get("usage"),
                failed=spec.get("failed", False), fail_reason=spec.get("reason"),
            )

        return fake, calls

    async def _collect(self, pipe, monkeypatch, behaviors, panel=("m/a", "m/b")):
        import open_webui_openrouter_pipe.requests.fusion_engine as fe

        fake, calls = self._fake_member(behaviors)
        monkeypatch.setattr(fe, "run_fusion_member", fake)
        from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
        monkeypatch.setattr(FilterManager, "collect_installed_web_tools_config",
                            AsyncMock(return_value=None))
        plan = FusionRunPlan(tuple(panel), "j/x", "j/x", 8)
        inv = FusionInnerInvocation(
            orchestrator=Mock(), messages=[{"role": "user", "content": "Q?"}],
            outer_model_id="openrouter/fusion", user={"id": "u"}, request=None,
            event_call=None, metadata={}, tools={}, valves=pipe.valves,
            session=object(), pipe_identifier="p", user_id="u",
        )
        events = []
        async for ev in fe.run_internal_fusion(pipe, invocation=inv, plan=plan):
            events.append(ev)
        return events, calls

    @staticmethod
    def _types(events):
        return [e["type"] for e in events]

    @pytest.mark.asyncio
    async def test_happy_path_sequence_and_usage(self, monkeypatch, pipe_instance_async):
        behaviors = {
            "m/a": {"live": [("delta", "hi a")], "content": "answer A", "usage": {"cost": 0.1}},
            "m/b": {"live": [("reasoning", "think b"), ("delta", "hi b")],
                    "content": "answer B", "usage": {"cost": 0.2}},
            "j/x": {"content": json.dumps(VALID_ANALYSIS), "usage": {"cost": 0.3}},
        }
        events, calls = await self._collect(pipe_instance_async, monkeypatch, behaviors)
        types = self._types(events)
        assert types[0] == "response.created"
        assert types.count("response.fusion_call.panel.added") == 2
        assert types.count("response.fusion_call.panel.completed") == 2
        assert "response.fusion_call.panel.delta" in types
        assert "response.fusion_call.panel.reasoning.delta" in types
        analysis_done = [e for e in events if e["type"] == "response.fusion_call.analysis.completed"]
        assert analysis_done and analysis_done[0]["analysis"] == VALID_ANALYSIS
        assert types[-1] == "response.completed"
        usage = events[-1]["response"]["usage"]
        assert round(usage.get("cost", 0), 6) == round(0.1 + 0.2 + 0.3 + 0.3, 6)
        judge_calls = [c for c in calls if c.get("temperature") == 0.0]
        assert judge_calls and judge_calls[0]["response_format"]["type"] == "json_schema"
        synth_calls = [c for c in calls if "BACKGROUND MATERIAL" in str(c.get("messages"))]
        assert synth_calls

    @pytest.mark.asyncio
    async def test_all_members_fail_skips_judge(self, monkeypatch, pipe_instance_async):
        events, calls = await self._collect(pipe_instance_async, monkeypatch,
                                            {"*": {"failed": True, "reason": "dead"}})
        types = self._types(events)
        assert "response.fusion_call.analysis.in_progress" not in types
        text_done = [e for e in events if e["type"] == "response.output_text.done"]
        assert text_done and "failed" in text_done[-1]["text"]
        assert types[-1] == "response.completed"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_judge_failure_omits_analysis_but_synthesizes(self, monkeypatch, pipe_instance_async):
        behaviors = {
            "m/a": {"content": "answer A", "usage": {"cost": 0.1}},
            "m/b": {"content": "answer B", "usage": {"cost": 0.1}},
            "j/x": {"content": "NOT JSON", "usage": {"cost": 0.2}},
        }
        events, calls = await self._collect(pipe_instance_async, monkeypatch, behaviors)
        types = self._types(events)
        assert "response.fusion_call.analysis.in_progress" in types
        assert "response.fusion_call.analysis.completed" not in types
        assert len([c for c in calls if c.get("temperature") == 0.0]) == 2
        assert types[-1] == "response.completed"
        usage = events[-1]["response"]["usage"]
        assert round(usage.get("cost", 0), 6) == round(0.1 + 0.1 + 0.2 + 0.2 + 0.2, 6)

    @pytest.mark.asyncio
    async def test_duplicate_plan_members_no_hang(self, monkeypatch, pipe_instance_async):
        behaviors = {"m/a": {"content": "answer"}, "j/x": {"content": json.dumps(VALID_ANALYSIS)}}
        events, _calls = await asyncio.wait_for(
            self._collect(pipe_instance_async, monkeypatch, behaviors, panel=("m/a", "m/a")),
            timeout=10,
        )
        assert events[-1]["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_no_function_call_or_native_reasoning_leaks(self, monkeypatch, pipe_instance_async):
        behaviors = {"m/a": {"content": "c"}, "j/x": {"content": json.dumps(VALID_ANALYSIS)}}
        events, _calls = await self._collect(pipe_instance_async, monkeypatch, behaviors, panel=("m/a",))
        for ev in events:
            etype = ev.get("type", "")
            assert "function_call" not in etype
            assert not etype.startswith("response.reasoning")
            item = ev.get("item")
            if isinstance(item, dict):
                assert item.get("type") != "function_call"


class TestInstalledWebToolsConfig:
    def _fake_functions(self, monkeypatch, *, content, is_active=True,
                        stored_valves=None, stored_user=None):
        import types

        import open_webui.models.functions as fx

        row = types.SimpleNamespace(is_active=is_active, content=content)

        monkeypatch.setattr(fx.Functions, "get_function_by_id",
                            AsyncMock(return_value=row), raising=False)
        monkeypatch.setattr(fx.Functions, "get_function_valves_by_id",
                            AsyncMock(return_value=stored_valves or {}), raising=False)
        monkeypatch.setattr(fx.Functions, "get_user_valves_by_id_and_user_id",
                            AsyncMock(return_value=stored_user or {}), raising=False)

    def _rendered_source(self, pipe):
        return pipe._ensure_filter_manager().render_openrouter_web_tools_filter_source()

    @pytest.mark.asyncio
    async def test_installed_filter_config_extracted(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        source = self._rendered_source(pipe)
        self._fake_functions(monkeypatch, content=source,
                             stored_valves={"WEB_SEARCH_MAX_RESULTS": 7})
        cfg = await pipe._ensure_filter_manager().collect_installed_web_tools_config("u1")
        assert cfg is not None
        server_tools, _stop = cfg
        assert server_tools["web_search"]["max_results"] == 7
        assert "web_fetch" not in server_tools

    @pytest.mark.asyncio
    async def test_user_enabled_web_fetch_included(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        source = self._rendered_source(pipe)
        self._fake_functions(monkeypatch, content=source,
                             stored_user={"WEB_FETCH": True})
        cfg = await pipe._ensure_filter_manager().collect_installed_web_tools_config("u1")
        assert cfg is not None
        server_tools, _stop = cfg
        assert "web_fetch" in server_tools
        assert server_tools["web_fetch"].get("engine")

    @pytest.mark.asyncio
    async def test_user_toggle_off_respected(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        source = self._rendered_source(pipe)
        self._fake_functions(monkeypatch, content=source,
                             stored_user={"WEB_SEARCH": False})
        cfg = await pipe._ensure_filter_manager().collect_installed_web_tools_config("u1")
        assert cfg is not None
        server_tools, _stop = cfg
        assert "web_search" not in server_tools

    @pytest.mark.asyncio
    async def test_inactive_filter_returns_none(self, monkeypatch, pipe_instance_async):
        pipe = pipe_instance_async
        self._fake_functions(monkeypatch, content=self._rendered_source(pipe), is_active=False)
        cfg = await pipe._ensure_filter_manager().collect_installed_web_tools_config("u1")
        assert cfg is None


class TestOwuiSurfaceInheritance:
    def test_metadata_channels_survive_into_inner_calls(self):
        outer = {
            "chat_id": "c1", "message_id": "m1",
            "model": {"id": "openrouter.openrouter/fusion"},
            "files": [{"id": "f1", "type": "file"}],
            "tool_servers": [{"url": "http://srv/openapi.json"}],
            "tool_ids": ["my_toolkit"],
            "features": {"web_search": False},
            "variables": {"x": "1"},
        }
        inner = build_inner_metadata(outer)
        assert inner["files"] == [{"id": "f1", "type": "file"}]
        assert inner["tool_servers"] == [{"url": "http://srv/openapi.json"}]
        assert inner["tool_ids"] == ["my_toolkit"]
        assert inner["variables"] == {"x": "1"}
        assert "chat_id" not in inner and "message_id" not in inner and "model" not in inner

    @pytest.mark.asyncio
    async def test_direct_tool_server_registry_engaged_for_members(
        self, orchestrator_and_pipe, monkeypatch
    ):
        orchestrator, pipe = orchestrator_and_pipe
        _prepare_pipe(pipe)
        captured: list[dict] = []
        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _capture_stream(captured))
        registry_calls: list[Any] = []
        executor = pipe._ensure_tool_executor()

        def spy_registry(*a, **k):
            registry_calls.append((a, k))
            return {}, []

        executor._build_direct_tool_server_registry = spy_registry
        inv = _invocation(orchestrator, pipe, Valves())
        inv.metadata = {"chat_id": "c1", "tool_servers": [{"url": "http://srv"}]}
        result = await run_fusion_member(
            pipe, inv, model="a/b", messages=inv.messages, system_prompt="SP",
            max_tool_calls=8, live_queue=None, bypass_restrictions=True,
            server_tools_config=({}, []),
        )
        assert result.failed is False
        assert registry_calls, "direct tool-server registry was not engaged for the inner call"


class TestInterruptionNoticeSuppression:
    @pytest.mark.asyncio
    async def test_inner_cut_stream_keeps_partial_without_notice(
        self, orchestrator_and_pipe, monkeypatch
    ):
        orchestrator, pipe = orchestrator_and_pipe
        _prepare_pipe(pipe)
        cut_stream = [
            {"type": "response.created", "response": {"model": "a/b"}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}},
            {"type": "response.output_text.delta", "output_index": 0, "delta": "partial answer"},
        ]
        captured: list[dict] = []
        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request",
                            _capture_stream(captured, events=cut_stream))
        inv = _invocation(orchestrator, pipe, Valves())
        result = await run_fusion_member(
            pipe, inv, model="a/b", messages=inv.messages, system_prompt="SP",
            max_tool_calls=8, live_queue=None, bypass_restrictions=True,
            server_tools_config=({}, []),
        )
        assert result.failed is False
        assert result.content == "partial answer"
        assert "interrupted" not in result.content.lower()


class TestStageLiveFeedbackEvents:
    def _fake_staged(self, behaviors):
        calls = []

        async def fake(pipe, invocation, **kw):
            calls.append(kw)
            model = kw["model"]
            if kw.get("temperature") == 0.0:
                spec = behaviors.get("judge") or {}
            elif "BACKGROUND MATERIAL" in str(kw.get("messages")):
                spec = behaviors.get("synth") or {}
            else:
                spec = behaviors.get(model) or {}
            queue = kw.get("live_queue")
            for kind, text in spec.get("live", []):
                if queue is not None:
                    await queue.put((kind, model, text))
            return FusionMemberResult(
                model=model, content=spec.get("content", ""), usage=spec.get("usage"),
                failed=spec.get("failed", False), fail_reason=spec.get("reason"),
            )

        return fake, calls

    async def _collect(self, pipe, monkeypatch, behaviors, panel=("m/a",)):
        import open_webui_openrouter_pipe.requests.fusion_engine as fe
        from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

        fake, calls = self._fake_staged(behaviors)
        monkeypatch.setattr(fe, "run_fusion_member", fake)
        monkeypatch.setattr(FilterManager, "collect_installed_web_tools_config",
                            AsyncMock(return_value=None))
        plan = FusionRunPlan(tuple(panel), "j/x", "j/x", 8)
        inv = FusionInnerInvocation(
            orchestrator=Mock(), messages=[{"role": "user", "content": "Q?"}],
            outer_model_id="openrouter/fusion", user={"id": "u"}, request=None,
            event_call=None, metadata={}, tools={}, valves=pipe.valves,
            session=object(), pipe_identifier="p", user_id="u",
        )
        events = []
        async for ev in fe.run_internal_fusion(pipe, invocation=inv, plan=plan):
            events.append(ev)
        return events, calls

    @pytest.mark.asyncio
    async def test_judge_reasoning_streams_and_content_deltas_dropped(
        self, monkeypatch, pipe_instance_async
    ):
        behaviors = {
            "m/a": {"content": "answer A"},
            "judge": {"live": [("reasoning", "weighing the drafts"), ("delta", "{\"cons")],
                      "content": json.dumps(VALID_ANALYSIS)},
            "synth": {"content": "final answer"},
        }
        events, _calls = await self._collect(pipe_instance_async, monkeypatch, behaviors)
        types = [e["type"] for e in events]
        jr = [e for e in events if e["type"] == "response.fusion_call.analysis.reasoning.delta"]
        assert jr and jr[0]["delta"] == "weighing the drafts"
        assert jr[0]["model"] == "j/x"
        assert jr[0]["output_index"] == 0
        assert types.index("response.fusion_call.analysis.reasoning.delta") < types.index(
            "response.fusion_call.analysis.completed")
        panel_deltas = [e for e in events if e["type"] == "response.fusion_call.panel.delta"]
        assert all("cons" not in (e.get("delta") or "") for e in panel_deltas)

    @pytest.mark.asyncio
    async def test_synthesis_in_progress_and_reasoning_stream(
        self, monkeypatch, pipe_instance_async
    ):
        behaviors = {
            "m/a": {"content": "answer A"},
            "judge": {"content": json.dumps(VALID_ANALYSIS)},
            "synth": {"live": [("reasoning", "composing"), ("delta", "final ")],
                      "content": "final answer"},
        }
        events, _calls = await self._collect(pipe_instance_async, monkeypatch, behaviors)
        types = [e["type"] for e in events]
        assert "response.fusion_call.synthesis.in_progress" in types
        sip = next(e for e in events if e["type"] == "response.fusion_call.synthesis.in_progress")
        assert sip["model"] == "j/x"
        assert types.index("response.fusion_call.synthesis.in_progress") > types.index(
            "response.output_item.done")
        assert types.index("response.fusion_call.synthesis.in_progress") < types.index(
            "response.output_text.done")
        sr = [e for e in events if e["type"] == "response.fusion_call.synthesis.reasoning.delta"]
        assert sr and sr[0]["delta"] == "composing" and sr[0]["model"] == "j/x"
        deltas = [e for e in events if e["type"] == "response.output_text.delta"]
        assert deltas and deltas[0]["output_index"] == 1

    @pytest.mark.asyncio
    async def test_all_fail_emits_no_synthesis_stage_events(
        self, monkeypatch, pipe_instance_async
    ):
        behaviors = {"m/a": {"failed": True, "reason": "dead"}}
        events, _calls = await self._collect(pipe_instance_async, monkeypatch, behaviors)
        types = [e["type"] for e in events]
        assert "response.fusion_call.synthesis.in_progress" not in types
        assert "response.fusion_call.synthesis.reasoning.delta" not in types
        assert types[-1] == "response.completed"
