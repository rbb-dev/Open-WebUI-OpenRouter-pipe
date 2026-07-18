from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, NamedTuple

from ..core.config import _PIPE_METADATA_KEY, NO_CONTENT_AFTER_TOOLS_FALLBACK
from ..core.fusion_defaults import (
    DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT,
    DEFAULT_FUSION_PANEL_SYSTEM_PROMPT,
    DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT,
    FusionRunPlan,
)
from ..core.logging_system import SessionLogger
from ..core.utils import merge_usage_stats
from ..models.registry import ModelFamily
from ..structured_task.schema import build_response_format, downgrade_strict_for_provider
from ..tools.tool_executor import _ToolExecutionContext


def build_inner_metadata(metadata: Any) -> dict[str, Any]:
    base = copy.deepcopy(metadata) if isinstance(metadata, dict) else {}
    for key in ("chat_id", "message_id", "model"):
        base.pop(key, None)
    pipe_meta = base.get(_PIPE_METADATA_KEY)
    if not isinstance(pipe_meta, dict):
        pipe_meta = {}
    pipe_meta["fusion_inner"] = True
    base[_PIPE_METADATA_KEY] = pipe_meta
    return base


def build_inner_valves(valves: Any, *, max_tool_calls: int) -> Any:
    loops = min(int(max_tool_calls), int(valves.MAX_FUNCTION_CALL_LOOPS))
    return valves.model_copy(update={
        "MAX_FUNCTION_CALL_LOOPS": loops,
        "COSTS_REDIS_DUMP": False,
    })


class FusionMemberResult(NamedTuple):
    model: str
    content: str
    usage: dict[str, Any] | None
    failed: bool
    fail_reason: str | None
    sources: tuple[dict[str, str], ...] = ()


class FusionCollector:
    def __init__(self, model: str, live_queue: asyncio.Queue | None = None):
        self.model = model
        self.live_queue = live_queue
        self.usage: dict[str, Any] | None = None
        self.sources: list[dict[str, str]] = []

    async def __call__(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        etype = event.get("type")
        raw = event.get("data")
        data = raw if isinstance(raw, dict) else {}
        if etype == "chat:message:delta":
            text = data.get("content")
            if isinstance(text, str) and text and self.live_queue is not None:
                await self.live_queue.put(("delta", self.model, text))
        elif etype == "fusion_inner:reasoning.delta":
            text = data.get("delta")
            if isinstance(text, str) and text and self.live_queue is not None:
                await self.live_queue.put(("reasoning", self.model, text))
        elif etype == "chat:completion":
            usage = data.get("usage")
            if isinstance(usage, dict) and usage:
                self.usage = usage
        elif etype == "source":
            info = data.get("source")
            if isinstance(info, dict):
                url = info.get("url")
                if isinstance(url, str) and url.strip():
                    title = info.get("name")
                    self.sources.append({
                        "url": url.strip(),
                        "title": title.strip() if isinstance(title, str) and title.strip() else url.strip(),
                    })


@dataclass(slots=True)
class FusionInnerInvocation:
    orchestrator: Any
    messages: list
    outer_model_id: str
    user: dict
    request: Any
    event_call: Any
    metadata: Any
    tools: Any
    valves: Any
    session: Any
    pipe_identifier: str
    allowlist_norm_ids: set = field(default_factory=set)
    enforced_norm_ids: set = field(default_factory=set)
    catalog_norm_ids: set = field(default_factory=set)
    features: dict = field(default_factory=dict)
    user_id: str = ""


async def run_fusion_member(
    pipe: Any,
    invocation: FusionInnerInvocation,
    *,
    model: str,
    messages: list,
    system_prompt: str,
    max_tool_calls: int,
    live_queue: asyncio.Queue | None,
    bypass_restrictions: bool,
    server_tools_config: tuple[dict[str, Any], list[dict[str, Any]]] | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
) -> FusionMemberResult:
    inner_body: dict[str, Any] = {
        "model": model,
        "stream": True,
        "messages": [{"role": "system", "content": system_prompt}, *copy.deepcopy(messages)],
    }
    if temperature is not None:
        inner_body["temperature"] = temperature
    if response_format is not None:
        inner_body["response_format"] = response_format
    inner_valves = build_inner_valves(invocation.valves, max_tool_calls=max_tool_calls)
    inner_metadata = build_inner_metadata(invocation.metadata)
    pipe_meta = inner_metadata[_PIPE_METADATA_KEY]
    pipe_meta.pop("server_tools", None)
    pipe_meta.pop("stop_server_tools_when", None)
    if server_tools_config is not None:
        tools_cfg, stop_when = server_tools_config
        if tools_cfg:
            pipe_meta["server_tools"] = copy.deepcopy(tools_cfg)
        if stop_when:
            pipe_meta["stop_server_tools_when"] = copy.deepcopy(stop_when)
    request_token = SessionLogger.request_id.set(f"fusion-inner-{uuid.uuid4().hex[:12]}")
    outer_ctx = pipe._TOOL_CONTEXT.get()
    ctx = None
    token = None
    if outer_ctx is not None:
        ctx = _ToolExecutionContext(
            queue=asyncio.Queue(maxsize=50),
            per_request_semaphore=outer_ctx.per_request_semaphore,
            global_semaphore=outer_ctx.global_semaphore,
            timeout=outer_ctx.timeout,
            batch_timeout=outer_ctx.batch_timeout,
            idle_timeout=outer_ctx.idle_timeout,
            user_id=invocation.user_id,
            event_emitter=None,
            batch_cap=outer_ctx.batch_cap,
            request=outer_ctx.request,
            user=outer_ctx.user,
            metadata=inner_metadata,
            request_id=SessionLogger.request_id.get() or "",
            fusion_inner=True,
            tool_call_budget=max_tool_calls,
        )
        executor = pipe._ensure_tool_executor()
        for _ in range(2):
            ctx.workers.append(asyncio.create_task(executor._tool_worker_loop(ctx)))
        token = pipe._TOOL_CONTEXT.set(ctx)
    collector = FusionCollector(model, live_queue)
    sink: dict[str, Any] = {}
    try:
        result = await invocation.orchestrator.process_request(
            inner_body,
            invocation.user,
            invocation.request,
            collector,
            invocation.event_call,
            inner_metadata,
            invocation.tools,
            None,
            None,
            inner_valves,
            invocation.session,
            model,
            invocation.pipe_identifier,
            set() if bypass_restrictions else set(invocation.allowlist_norm_ids),
            set() if bypass_restrictions else set(invocation.enforced_norm_ids),
            set() if bypass_restrictions else set(invocation.catalog_norm_ids),
            dict(invocation.features or {}),
            user_id=invocation.user_id,
            outcome_sink=sink,
        )
        content = result if isinstance(result, str) else ""
        if "error_occurred" not in sink:
            preview = content.strip().replace("\n", " ")[:160]
            return FusionMemberResult(
                model=model, content="", usage=collector.usage, failed=True,
                fail_reason=preview or "rejected before send",
                sources=tuple(collector.sources),
            )
        failed = (
            bool(sink.get("error_occurred"))
            or not content.strip()
            or content == NO_CONTENT_AFTER_TOOLS_FALLBACK
        )
        reason = sink.get("reason") if failed else None
        return FusionMemberResult(
            model=model, content=content, usage=collector.usage,
            failed=failed, fail_reason=reason if isinstance(reason, str) else None,
            sources=tuple(collector.sources),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return FusionMemberResult(
            model=model, content="", usage=collector.usage,
            failed=True, fail_reason=f"{type(exc).__name__}: {exc}",
            sources=tuple(collector.sources),
        )
    finally:
        SessionLogger.request_id.reset(request_token)
        if token is not None:
            pipe._TOOL_CONTEXT.reset(token)
        if ctx is not None:
            for worker in ctx.workers:
                worker.cancel()
            await asyncio.gather(*ctx.workers, return_exceptions=True)


ANALYSIS_KEYS = frozenset(
    {"consensus", "contradictions", "partial_coverage", "unique_insights", "blind_spots"}
)

_STR_ARRAY = {"type": "array", "items": {"type": "string"}}

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "consensus": _STR_ARRAY,
        "contradictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "stances": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "model": {"type": "string"},
                                "stance": {"type": "string"},
                            },
                            "required": ["model", "stance"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["topic", "stances"],
                "additionalProperties": False,
            },
        },
        "partial_coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "models": _STR_ARRAY,
                    "point": {"type": "string"},
                },
                "required": ["models", "point"],
                "additionalProperties": False,
            },
        },
        "unique_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "insight": {"type": "string"},
                },
                "required": ["model", "insight"],
                "additionalProperties": False,
            },
        },
        "blind_spots": _STR_ARRAY,
    },
    "required": sorted(ANALYSIS_KEYS),
    "additionalProperties": False,
}


def build_analysis_response_format(judge_model: str) -> dict[str, Any]:
    rf = build_response_format(name="fusion_analysis", schema=ANALYSIS_SCHEMA, strict=True)
    return downgrade_strict_for_provider(
        rf, supported_parameters=ModelFamily.supported_parameters(judge_model)
    )


def parse_analysis(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(stripped[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, dict) or set(obj) != ANALYSIS_KEYS:
        return None
    return obj


def resolve_fusion_prompt(valve_value: Any, default: str) -> str:
    text = valve_value if isinstance(valve_value, str) else ""
    return text.strip() and text or default


def latest_user_text(input_items: Any) -> str:
    if isinstance(input_items, str):
        return input_items
    if not isinstance(input_items, list):
        return ""
    for item in reversed(input_items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                text for p in content
                if isinstance(p, dict) and isinstance(text := p.get("text"), str)
            ]
            if parts:
                return "\n".join(parts)
    return ""


def degrade_note(result: FusionMemberResult) -> str:
    reason = (result.fail_reason or "no usable answer").strip()
    return f"*(panel member failed: {reason})*"


def build_judge_input(question: str, results: list[FusionMemberResult]) -> list[dict[str, Any]]:
    blocks: list[str] = []
    for res in results:
        if res.failed:
            blocks.append(
                f"## PANEL ANSWER — model id: {res.model}\n\n"
                f"model {res.model} failed: {res.fail_reason or 'no usable answer'}"
            )
        else:
            blocks.append(f"## PANEL ANSWER — model id: {res.model}\n\n{res.content}")
    payload = (
        f"# USER QUESTION\n\n{question}\n\n# PANEL ANSWERS\n\n" + "\n\n".join(blocks)
    )
    return [{"role": "user", "content": [{"type": "input_text", "text": payload}]}]


def build_synthesis_material(results: list[FusionMemberResult], analysis: dict[str, Any] | None) -> str:
    blocks: list[str] = []
    for res in results:
        if res.failed:
            blocks.append(
                f"### DRAFT — internal model id: {res.model}\n\n"
                f"model {res.model} failed: {res.fail_reason or 'no usable answer'}"
            )
        else:
            blocks.append(f"### DRAFT — internal model id: {res.model}\n\n{res.content}")
    analysis_text = json.dumps(analysis, ensure_ascii=False) if analysis else "(absent)"
    return (
        "[BACKGROUND MATERIAL — prepared before this turn; not visible to the user]\n\n"
        "## DRAFTS\n\n" + "\n\n".join(blocks) + "\n\n## ANALYSIS\n\n" + analysis_text
    )


def aggregate_sources(results: list[FusionMemberResult]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for res in results:
        for src in res.sources:
            url = src.get("url") or ""
            if url.endswith("?utm_source=openai"):
                url = url[: -len("?utm_source=openai")]
            if not url or url in seen:
                continue
            seen.add(url)
            out.append({"url": url, "title": src.get("title") or url})
    return out


async def run_internal_fusion(
    pipe: Any,
    *,
    invocation: FusionInnerInvocation,
    plan: FusionRunPlan,
) -> AsyncGenerator[dict[str, Any], None]:
    valves = invocation.valves
    panel_prompt = resolve_fusion_prompt(
        getattr(valves, "FUSION_PANEL_SYSTEM_PROMPT", ""), DEFAULT_FUSION_PANEL_SYSTEM_PROMPT
    )
    judge_prompt = resolve_fusion_prompt(
        getattr(valves, "FUSION_JUDGE_SYSTEM_PROMPT", ""), DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT
    )
    synthesis_prompt = resolve_fusion_prompt(
        getattr(valves, "FUSION_SYNTHESIS_SYSTEM_PROMPT", ""), DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT
    )
    try:
        web_tools_config = await pipe._ensure_filter_manager().collect_installed_web_tools_config(
            invocation.user_id
        )
    except Exception:
        web_tools_config = None
    item_id = f"st_fusion_internal_{uuid.uuid4().hex[:12]}"
    total_usage: dict[str, Any] = {}
    live_queue: asyncio.Queue = asyncio.Queue()
    member_tasks: list[asyncio.Task] = []

    async def _member_wrapper(member_model: str) -> None:
        try:
            res = await run_fusion_member(
                pipe,
                invocation,
                model=member_model,
                messages=invocation.messages,
                system_prompt=panel_prompt,
                max_tool_calls=plan.max_tool_calls,
                live_queue=live_queue,
                bypass_restrictions=plan.panel_from_preset,
                server_tools_config=web_tools_config,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            res = FusionMemberResult(
                model=member_model, content="", usage=None,
                failed=True, fail_reason=f"{type(exc).__name__}: {exc}",
            )
        await live_queue.put(("member_done", member_model, res))

    try:
        yield {"type": "response.created",
               "response": {"id": f"resp_{item_id}", "model": invocation.outer_model_id,
                            "created_at": time.time()}}
        yield {"type": "response.in_progress", "response": {}}
        yield {"type": "response.output_item.added", "output_index": 0,
               "item": {"id": item_id, "type": "openrouter:fusion", "status": "in_progress"}}
        yield {"type": "response.fusion_call.in_progress", "output_index": 0, "item_id": item_id}
        for member_model in plan.panel_models:
            yield {"type": "response.fusion_call.panel.added", "output_index": 0,
                   "item_id": item_id, "model": member_model}

        for member_model in plan.panel_models:
            member_tasks.append(asyncio.create_task(_member_wrapper(member_model)))

        results: dict[str, FusionMemberResult] = {}
        done_members = 0
        while done_members < len(member_tasks):
            kind, member_model, payload = await live_queue.get()
            if kind == "delta":
                yield {"type": "response.fusion_call.panel.delta", "output_index": 0,
                       "item_id": item_id, "model": member_model, "delta": payload}
            elif kind == "reasoning":
                yield {"type": "response.fusion_call.panel.reasoning.delta", "output_index": 0,
                       "item_id": item_id, "model": member_model, "delta": payload}
            elif kind == "member_done":
                done_members += 1
                results[member_model] = payload
                if payload.usage:
                    total_usage = merge_usage_stats(total_usage, payload.usage)
                content = degrade_note(payload) if payload.failed else payload.content
                yield {"type": "response.fusion_call.panel.completed", "output_index": 0,
                       "item_id": item_id, "model": member_model, "content": content}

        ordered = [results[m] for m in dict.fromkeys(plan.panel_models) if m in results]
        usable = [r for r in ordered if not r.failed]
        analysis: dict[str, Any] | None = None
        question = latest_user_text(invocation.messages)

        if usable:
            yield {"type": "response.fusion_call.analysis.in_progress", "output_index": 0,
                   "item_id": item_id, "judge_model": plan.judge_model}
            judge_queue: asyncio.Queue = asyncio.Queue()

            async def _judge_wrapper(judge_messages: list) -> None:
                try:
                    res = await run_fusion_member(
                        pipe,
                        invocation,
                        model=plan.judge_model,
                        messages=judge_messages,
                        system_prompt=judge_prompt,
                        max_tool_calls=plan.max_tool_calls,
                        live_queue=judge_queue,
                        bypass_restrictions=plan.judge_from_preset,
                        server_tools_config=web_tools_config,
                        temperature=0.0,
                        response_format=build_analysis_response_format(plan.judge_model),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    res = FusionMemberResult(
                        model=plan.judge_model, content="", usage=None,
                        failed=True, fail_reason=f"{type(exc).__name__}: {exc}",
                    )
                await judge_queue.put(("member_done", plan.judge_model, res))

            judge_task = asyncio.create_task(_judge_wrapper(build_judge_input(question, ordered)))
            member_tasks.append(judge_task)
            judge_res = None
            while judge_res is None:
                kind, _judge_model, payload = await judge_queue.get()
                if kind == "reasoning":
                    yield {"type": "response.fusion_call.analysis.reasoning.delta",
                           "output_index": 0, "item_id": item_id,
                           "model": plan.judge_model, "delta": payload}
                elif kind == "member_done":
                    judge_res = payload
            if judge_res.usage:
                total_usage = merge_usage_stats(total_usage, judge_res.usage)
            analysis = parse_analysis(judge_res.content) if not judge_res.failed else None
            if analysis is None and not judge_res.failed:
                repair_task = asyncio.create_task(_judge_wrapper(
                    build_judge_input(question, ordered) + [
                        {"role": "assistant", "content": judge_res.content},
                        {"role": "user", "content": (
                            "Your previous output was not a single valid JSON object with exactly "
                            "the five required keys. Emit ONLY the corrected JSON object now — "
                            "no prose, no fences."
                        )},
                    ]
                ))
                member_tasks.append(repair_task)
                repair_res = None
                while repair_res is None:
                    kind, _judge_model, payload = await judge_queue.get()
                    if kind == "reasoning":
                        yield {"type": "response.fusion_call.analysis.reasoning.delta",
                               "output_index": 0, "item_id": item_id,
                               "model": plan.judge_model, "delta": payload}
                    elif kind == "member_done":
                        repair_res = payload
                if repair_res.usage:
                    total_usage = merge_usage_stats(total_usage, repair_res.usage)
                analysis = parse_analysis(repair_res.content) if not repair_res.failed else None
            if analysis is not None:
                yield {"type": "response.fusion_call.analysis.completed", "output_index": 0,
                       "item_id": item_id, "analysis": analysis}

        yield {"type": "response.fusion_call.completed", "output_index": 0, "item_id": item_id}
        done_item: dict[str, Any] = {
            "id": item_id, "type": "openrouter:fusion", "status": "completed",
            "responses": [
                {"model": r.model, "content": degrade_note(r) if r.failed else r.content}
                for r in ordered
            ],
        }
        if analysis is not None:
            done_item["analysis"] = analysis
        run_sources = aggregate_sources(ordered)
        if run_sources:
            done_item["sources"] = run_sources
        yield {"type": "response.output_item.done", "output_index": 0, "item": done_item}

        synth_text = ""
        if not usable:
            failure_answer = (
                "Every panel member failed to produce an answer, so no deliberated "
                "response is available for this run. Please try again."
            )
            yield {"type": "response.output_item.added", "output_index": 1,
                   "item": {"type": "message"}}
            yield {"type": "response.output_text.done", "output_index": 1,
                   "text": failure_answer}
        else:
            material = build_synthesis_material(ordered, analysis)
            synth_messages = copy.deepcopy(invocation.messages)
            synth_messages.append({"role": "system", "content": material})
            yield {"type": "response.fusion_call.synthesis.in_progress",
                   "output_index": 0, "item_id": item_id, "model": plan.synthesis_model}
            synth_queue: asyncio.Queue = asyncio.Queue()

            async def _synth_wrapper() -> None:
                try:
                    res = await run_fusion_member(
                        pipe,
                        invocation,
                        model=plan.synthesis_model,
                        messages=synth_messages,
                        system_prompt=synthesis_prompt,
                        max_tool_calls=plan.max_tool_calls,
                        live_queue=synth_queue,
                        bypass_restrictions=plan.judge_from_preset,
                        server_tools_config=web_tools_config,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    res = FusionMemberResult(
                        model=plan.synthesis_model, content="", usage=None,
                        failed=True, fail_reason=f"{type(exc).__name__}: {exc}",
                    )
                await synth_queue.put(("member_done", plan.synthesis_model, res))

            synth_task = asyncio.create_task(_synth_wrapper())
            member_tasks.append(synth_task)
            opened = False
            synth_result: FusionMemberResult | None = None
            while synth_result is None:
                kind, _model, payload = await synth_queue.get()
                if kind == "delta":
                    if not opened:
                        yield {"type": "response.output_item.added", "output_index": 1,
                               "item": {"type": "message"}}
                        opened = True
                    synth_text += payload
                    yield {"type": "response.output_text.delta", "output_index": 1,
                           "delta": payload}
                elif kind == "reasoning":
                    yield {"type": "response.fusion_call.synthesis.reasoning.delta",
                           "output_index": 0, "item_id": item_id,
                           "model": plan.synthesis_model, "delta": payload}
                elif kind == "member_done":
                    synth_result = payload
            if synth_result.usage:
                total_usage = merge_usage_stats(total_usage, synth_result.usage)
            if not synth_result.failed and synth_result.content:
                final_text = synth_result.content
            elif synth_text:
                final_text = synth_text
            else:
                final_text = (
                    "The final synthesis step failed, but the panel answers above are "
                    "complete and usable."
                )
            if not opened:
                yield {"type": "response.output_item.added", "output_index": 1,
                       "item": {"type": "message"}}
            yield {"type": "response.output_text.done", "output_index": 1, "text": final_text}

        yield {"type": "response.completed",
               "response": {"model": invocation.outer_model_id, "output": [], "usage": total_usage}}
    finally:
        for task in member_tasks:
            if not task.done():
                task.cancel()
        if member_tasks:
            await asyncio.gather(*member_tasks, return_exceptions=True)


FUSION_INNER_APPLIED_DIALS = (
    "reasoning_preferences",
    "gemini_thinking",
    "anthropic_verbosity",
    "max_output_tokens",
    "capability_tool_gate",
    "context_transforms_state",
)
