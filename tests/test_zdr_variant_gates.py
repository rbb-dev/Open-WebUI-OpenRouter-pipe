"""Issue #56: a routing variant of a ZDR-capable model must pass EVERY ZDR gate.

Reported twice. The first report was `ZDR_ENFORCE` rejecting
`deepseek/deepseek-v4-flash:nitro`; v2.7.1 fixed it by stripping the variant suffix at
that one call site. The reporter came back on v2.7.4 with the same model blocked by a
different gate -- `Restricted by: ZDR_MODELS_ONLY` -- because two other membership checks
still passed the full variant id.

A `:nitro`, `:floor` or `:online` suffix selects how the SAME base model is routed. Its
endpoints are the base model's endpoints, and OpenRouter's ZDR list only ever contains
base ids, so a variant can never appear in it. Any gate comparing the full id therefore
rejects every variant of every ZDR-capable model.

These tests drive all three gates against one registry state. Testing the gates
separately is what allowed the second report: the fix was applied where the bug was seen
rather than where the rule lives, and nothing compared the gates to each other.
"""

from __future__ import annotations

import pytest

from tests.test_request_orchestrator import _consume_stream, _smart_callback

from open_webui_openrouter_pipe.models.registry import ModelFamily
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry as Registry

_BASE = "deepseek/deepseek-v4-flash"
_NOT_ZDR = "openai/gpt-4o"


@pytest.fixture
def zdr_registry(monkeypatch):
    """A ZDR list holding base ids only, which is the shape OpenRouter returns."""
    base = ModelFamily.base_model(_BASE)
    monkeypatch.setattr(Registry, "_zdr_model_ids", {base})
    monkeypatch.setattr(
        Registry, "_specs", {base: {"features": set()}, ModelFamily.base_model(_NOT_ZDR): {"features": set()}}
    )
    return Registry


@pytest.mark.parametrize("variant", ["", ":nitro", ":floor", ":online", ":free"])
def test_every_routing_variant_of_a_zdr_base_is_zdr_capable(zdr_registry, variant):
    """The rule itself, at the one place all three gates consult."""
    assert zdr_registry.is_zdr_capable(f"{_BASE}{variant}") is True, (
        f"{_BASE}{variant} was not recognised as ZDR-capable. OpenRouter's ZDR list "
        "contains base ids only, so a gate comparing the full variant id rejects every "
        "variant of every ZDR-capable model."
    )


@pytest.mark.parametrize("variant", ["", ":nitro", ":floor"])
def test_a_variant_of_a_non_zdr_model_is_still_rejected(zdr_registry, variant):
    """The opposite arm: stripping must not admit a model that is genuinely not ZDR.

    Without this, `return True` satisfies the test above.
    """
    assert zdr_registry.is_zdr_capable(f"{_NOT_ZDR}{variant}") is False


@pytest.mark.parametrize("variant", ["", ":nitro", ":floor"])
def test_the_zdr_models_only_gate_admits_the_same_variants(zdr_registry, variant):
    """The gate the reporter actually hit on v2.7.4.

    Driven through `_restriction_reasons`, the function that produced the
    `Restricted by: ZDR_MODELS_ONLY` line in their screenshot.
    """
    from open_webui_openrouter_pipe import Pipe

    pipe = Pipe()
    pipe.valves.ZDR_MODELS_ONLY = True
    norm_id = ModelFamily.base_model(f"{_BASE}{variant}")
    reasons = pipe._model_restriction_reasons(
        norm_id,
        valves=pipe.valves,
        allowlist_norm_ids={norm_id},
        catalog_norm_ids={norm_id},
    )
    assert "ZDR_MODELS_ONLY" not in reasons, (
        f"{_BASE}{variant} was restricted by ZDR_MODELS_ONLY while ZDR_ENFORCE admits "
        f"it. Two gates over one rule disagreed: {reasons}"
    )


# Every place that asks the ZDR question, pinned in both directions. A census that only
# looks for offenders reports green when it matches NOTHING -- rename `is_zdr_capable`,
# or move a gate into a module the walk does not cover, and `offenders` is empty because
# `seen` is empty. That is the exact failure this file exists for: issue #56 recurred
# because the fix was applied where the bug was seen rather than where the rule lives.
#
# All three parts are load-bearing, and each catches a mutation the other two do not:
#   file      -- a gate relocated into an unscanned module
#   function  -- a gate relocated inside the SAME file, keeping its argument name
#   argument  -- a call site that starts passing a DIFFERENT expression, such as an id
#                already stripped on a line above, which the `offenders` predicate below
#                cannot see because the text holds no literal `split`
# Renaming a local also lands here. That is a reviewed edit, not a bug: confirm the
# argument is still the normalized catalog id, then update the entry.
_ZDR_GATES = {
    ("open_webui_openrouter_pipe/pipe.py", "_apply_model_filters", "norm_id"),
    ("open_webui_openrouter_pipe/pipe.py", "_model_restriction_reasons", "model_norm_id"),
    (
        "open_webui_openrouter_pipe/requests/orchestrator.py",
        "process_request",
        "normalized_model_id",
    ),
}


def test_no_gate_compares_a_zdr_id_without_stripping_the_variant():
    """The rule lives in one place, so the gates cannot drift apart again.

    v2.7.1 stripped at the ZDR_ENFORCE call site and left two others comparing full ids.
    A call site that strips for itself is how that happened, so none may -- and the set
    of call sites is pinned, so one that moves out of view fails instead of vanishing.
    """
    import ast
    import pathlib

    from tests.package_sources import parsed_sources

    package = pathlib.Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"
    seen, offenders = set(), []

    def _visit(node, rel, owner):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            owner = node.name
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name == "is_zdr_capable" and node.args:
                arg = ast.unparse(node.args[0])
                seen.add((rel, owner or "<module>", arg))
                if "rsplit" in arg or "split" in arg:
                    offenders.append(f"{rel}:{node.lineno} passes {arg}")
        for child in ast.iter_child_nodes(node):
            _visit(child, rel, owner)

    for path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        _visit(tree, path.relative_to(package.parent).as_posix(), None)

    assert seen == _ZDR_GATES, (
        "the set of ZDR gates changed; a census that matches nothing reports no "
        "offenders, which is how issue #56 recurred. Each entry is "
        "(file, enclosing function, argument text) -- so a gate that MOVED lands here, "
        "and so does one that now passes a different expression. Renaming a local also "
        "lands here: that is a reviewed edit, not a bug. Confirm the argument is still "
        "the normalized catalog id, then update the entry.\n"
        f"missing: {sorted(_ZDR_GATES - seen)}\n"
        f"unexpected: {sorted(seen - _ZDR_GATES)}"
    )
    assert not offenders, (
        "these call sites strip the variant suffix themselves before asking "
        f"is_zdr_capable, which is how the gates came to disagree: {offenders}. "
        "Reachable only once a stripping site has been added to _ZDR_GATES above -- "
        "i.e. exactly when someone is repairing the assertion instead of the code. Do "
        "not delete this as unreachable."
    )


@pytest.mark.asyncio
async def test_the_stored_user_valves_are_read_once_per_request():
    """One request, one snapshot -- and one database round trip.

    `_stored_user_valves` reaches Open WebUI's `get_user_valves_by_id_and_user_id`, which
    does a full `Users.get_user_by_id` fetch plus a valve decrypt with no cache. It was
    awaited twice per request: once at the pipe entry point and again in the orchestrator's
    ZDR block, which runs on a default install because ZDR_ENFORCE defaults False and
    ALLOW_USER_ZDR_OVERRIDE defaults True.

    Two costs, and the second is the one that matters: the reads are independent
    snapshots, so a valve saved between them leaves the request honouring one for every
    setting and the other for the privacy routing, with nothing able to detect it.

    Asserted as a COUNT of calls to the real Open WebUI reader, driven through a full
    `pipe()` call. Asserting that the orchestrator "uses the passed value" would pass
    against a second read whose result happened to match.
    """
    import open_webui.models.functions as owf
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    calls = {"n": 0}

    class _CountingFunctions:
        async def get_user_valves_by_id_and_user_id(self, _id, _user_id, db=None):
            calls["n"] += 1
            return {"REQUEST_ZDR": False}

    original = owf.Functions
    owf.Functions = _CountingFunctions()
    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"

        async def _emit(_event):
            pass

        with aioresponses() as http:
            http.post("https://openrouter.ai/api/v1/responses", payload={"output": []}, repeat=True)
            http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "openai/gpt-4o-mini", "name": "M"}]},
                repeat=True,
            )
            http.get("https://openrouter.ai/api/v1/endpoints/zdr", payload={"data": []}, repeat=True)
            await pipe.pipe(
                body={"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                __user__={"id": "u1", "valves": Pipe.UserValves()},
                __request__=None,
                __event_emitter__=_emit,
                __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o-mini"}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
    finally:
        owf.Functions = original
        await pipe.close()

    # Equality, not a ceiling: `<= 1` is satisfied by ZERO, and the three early
    # `return supplied` paths in _stored_user_valves make zero reachable -- a guard
    # added ahead of the read, a caching short-circuit, an exception swallowed into the
    # fallback. A test named "read once per request" cannot admit "never read".
    assert calls["n"] == 1, (
        f"the stored user valves were read {calls['n']} times for one request. Each read "
        "is a full user-row fetch, and two reads are two snapshots -- a valve saved "
        "between them splits the request across both."
    )


@pytest.mark.parametrize(
    ("free_is_zdr", "expected"), [(False, False), (True, True)], ids=["not-listed", "listed"]
)
def test_a_real_catalog_variant_answers_for_itself_not_its_base(free_is_zdr, expected):
    """`:free` is a MODEL, `:nitro` is a routing variant, and only one may fall back.

    OpenRouter's catalog lists 24-odd ids carrying `:free` (and `:thinking`) as models in
    their own right, with their own providers, pricing and retention policies. The suffix
    strip added for issue #56 applied to those too, so a `:free` model inherited its paid
    base's ZDR status and appeared under a ZDR-only filter that exists to exclude exactly
    that. Synthetic routing variants have no catalog entry, so the base fallback still
    reaches them.

    Two arms over the same id: a constant answer cannot satisfy both.
    """
    from open_webui_openrouter_pipe.models.registry import ModelFamily, OpenRouterModelRegistry

    base = ModelFamily.base_model("openai/gpt-oss-120b")
    free = ModelFamily.base_model("openai/gpt-oss-120b:free")
    original_specs, original_zdr = OpenRouterModelRegistry._specs, OpenRouterModelRegistry._zdr_model_ids
    try:
        OpenRouterModelRegistry._specs = {base: {}, free: {}}
        OpenRouterModelRegistry._zdr_model_ids = {base} | ({free} if free_is_zdr else set())

        assert OpenRouterModelRegistry.is_zdr_capable("openai/gpt-oss-120b:free") is expected, (
            "a `:free` model is a separate catalog entry served by different providers; "
            "answering for it from the paid base shows non-ZDR models under a ZDR filter"
        )
        assert OpenRouterModelRegistry.is_zdr_capable("openai/gpt-oss-120b:nitro") is True, (
            "`:nitro` has no catalog entry -- it is a routing variant the pipe itself "
            "synthesises -- so it must still resolve through its base (issue #56)"
        )
    finally:
        OpenRouterModelRegistry._specs = original_specs
        OpenRouterModelRegistry._zdr_model_ids = original_zdr


@pytest.mark.asyncio
async def test_a_user_with_no_id_still_reaches_openrouter_with_their_own_settings():
    """Asserted on the outbound payload, because the helper's return type is not the point.

    Open WebUI builds `__user__` as `user.model_dump() if isinstance(user, UserModel)
    else {}` and then writes `params["__user__"]["valves"]` onto it, so a request can
    legitimately carry valves and no id. Checking the private helper's shape would pass
    for a version that returns `{}` or a fresh UserValves -- both of which still drop the
    user's setting. What matters is whether `provider.zdr` reaches OpenRouter.
    """
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
        pipe.valves.ZDR_ENFORCE = False
        pipe.valves.ALLOW_USER_ZDR_OVERRIDE = True

        captured: list[dict] = []
        callback = _smart_callback(captured, "Response")

        async def _emit(_event):
            pass

        with aioresponses() as http:
            http.post("https://openrouter.ai/api/v1/responses", callback=callback, repeat=True)
            http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "openai/gpt-4o-mini", "name": "M"}]},
                repeat=True,
            )
            http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr",
                payload={"data": [{"model_id": "openai/gpt-4o-mini"}]},
                repeat=True,
            )
            result = await pipe.pipe(
                body={
                    "model": "openai/gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                __user__={"valves": Pipe.UserValves(REQUEST_ZDR=True)},
                __request__=None,
                __event_emitter__=_emit,
                __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o-mini"}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
            await _consume_stream(result)
    finally:
        await pipe.close()

    assert captured, "no request reached OpenRouter"
    provider = captured[-1].get("provider") or {}
    assert provider.get("zdr") is True, (
        "a request whose __user__ carried valves but no id was routed WITHOUT Zero Data "
        f"Retention (provider={provider!r}). The user opted in and their setting was "
        "silently replaced by the default."
    )


@pytest.mark.parametrize("requested", [True, False], ids=["zdr-on", "zdr-off"])
@pytest.mark.asyncio
async def test_every_return_path_yields_something_parse_user_valves_understands(requested):
    """A request with no user id must not silently lose the user's settings.

    `_stored_user_valves` has four returns. One of them returned a 2-tuple, which
    `parse_user_valves` recognises as neither a model instance nor a Mapping, so it fell
    through to `model(), []` -- a default-constructed UserValves with an EMPTY rejected
    list. Every field the user had set reverted to its default, and because `rejected`
    was empty the diagnostic loop at the pipe entry point printed nothing. `REQUEST_ZDR`
    went from True to False in total silence.

    Parametrised over both values so a production `return UserValves(REQUEST_ZDR=True)`
    cannot satisfy it, and asserted on the PARSED valve rather than on the return type:
    the shape of the intermediate is not the property, the surviving setting is.
    """
    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import parse_user_valves

    pipe = Pipe()
    try:
        supplied = Pipe.UserValves(REQUEST_ZDR=requested)
        raw = await pipe._stored_user_valves({"valves": supplied})
        parsed, rejected = parse_user_valves(raw, model=Pipe.UserValves)
    finally:
        await pipe.close()

    assert parsed.REQUEST_ZDR is requested, (
        f"a request carrying no user id lost the user's REQUEST_ZDR={requested} "
        f"preference and read {parsed.REQUEST_ZDR} instead. `_stored_user_valves` "
        f"returned {type(raw).__name__}, which parse_user_valves does not recognise, so "
        "every field fell back to its default."
    )
    assert not rejected, (
        f"nothing was unparseable here, yet these fields were reported rejected: {rejected}"
    )


@pytest.mark.asyncio
async def test_a_broken_valve_row_does_not_end_that_users_chat():
    """A row read that fails must not decide the ZDR question.

    Open WebUI returns None from `get_user_valves_by_id_and_user_id` for a single user
    whose settings row will not validate. (A blob that will not DECRYPT is a different
    case: decrypt_valves returns `{}`, and that is handled as unreadable rather than as
    "nothing stored" -- see the sibling that drives it.) The host is
    healthy and every other user is unaffected, but that row stays broken until someone
    repairs it -- so treating the failure as "the user might have asked for ZDR" ends
    that one user's chat indefinitely, for a preference they may never have set.

    A REJECTED FIELD is different: the row was read, REQUEST_ZDR was in it, and it would
    not parse. There the answer is genuinely lost and enforcing is right.

    Its opposite arm is the sibling below: the two end in opposite places -- one
    answered, one refused -- so neither a hardcoded enforce nor a hardcoded allow
    satisfies both.
    """
    import open_webui.models.functions as owf
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    class _RaisingFunctions:
        async def get_user_valves_by_id_and_user_id(self, _id, _user_id, db=None):
            raise RuntimeError("this user's settings row will not validate")

    original = owf.Functions
    owf.Functions = _RaisingFunctions()
    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
        pipe.valves.ZDR_ENFORCE = False
        pipe.valves.ALLOW_USER_ZDR_OVERRIDE = True

        captured: list[dict] = []
        callback = _smart_callback(captured, "Response")
        events: list[dict] = []

        async def _emit(event):
            events.append(event)

        with aioresponses() as http:
            http.post("https://openrouter.ai/api/v1/responses", callback=callback, repeat=True)
            http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "openai/gpt-4o", "name": "M"}]},
                repeat=True,
            )
            http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr",
                payload={"data": [{"model_id": "openai/gpt-4o-mini"}]},
                repeat=True,
            )
            result = await pipe.pipe(
                body={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                __user__={"id": "u1", "valves": Pipe.UserValves()},
                __request__=None,
                __event_emitter__=_emit,
                __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o"}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
            await _consume_stream(result)
    finally:
        owf.Functions = original
        await pipe.close()

    assert captured, (
        "the request never reached OpenRouter: a failed read of one user's valve row "
        "refused a model that is not ZDR-capable, so that user cannot chat at all until "
        "somebody repairs their settings row.\n"
        f"events={[e.get('type') for e in events]}"
    )
    provider = captured[-1].get("provider") or {}
    assert provider.get("zdr") is not True, (
        "ZDR was forced on by a failed row read, which is not evidence that this user "
        f"asked for it (provider={provider!r})"
    )


def _assert_names_the_users_preference(reported: str) -> None:
    """The refusal has to point at the user's own answer, not at the admin valve.

    Both causes end in the same card, and only one of them is switched on here.
    Asserting the label each panel really shows, read at runtime, keeps the two apart
    without pinning the wording -- and a card that names the admin valve instead sends
    an operator to a setting that is already off.
    """
    from open_webui_openrouter_pipe import Pipe

    theirs = Pipe.UserValves.model_fields["REQUEST_ZDR"].title
    admins = Pipe.Valves.model_fields["ZDR_ENFORCE"].title
    assert theirs and admins, "the settings panels show no label for these"
    assert theirs in reported, (
        f"the refusal reported {reported!r}, which never names {theirs!r} -- the label the "
        "user's own panel shows for the answer that could not be read"
    )
    assert admins not in reported, (
        f"the refusal reported {reported!r}, naming {admins!r}. That valve is switched off "
        "in this test, so it leaves the operator with nothing to check and no way to reach "
        "the real cause."
    )


@pytest.mark.asyncio
async def test_an_unparseable_zdr_field_still_enforces_zdr():
    """The opposite arm of the sibling above: here the answer is genuinely lost.

    The row WAS read and REQUEST_ZDR was in it with a value that will not parse, so the
    user may well have opted in and we cannot tell. Routing them to a provider that
    retains the conversation is the one outcome this valve exists to prevent, so a model
    that is not ZDR-capable is refused.
    """
    import open_webui.models.functions as owf
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    class _CorruptFieldFunctions:
        async def get_user_valves_by_id_and_user_id(self, _id, _user_id, db=None):
            return {"REQUEST_ZDR": {"not": "a boolean"}}

    from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter

    reasons: list[str] = []
    real_emit = ErrorFormatter._emit_templated_error

    async def _record(self, emitter, *, variables=None, **kw):
        variables = variables or {}
        reasons.append(variables.get("restriction_reasons", ""))
        return await real_emit(self, emitter, variables=variables, **kw)

    original = owf.Functions
    owf.Functions = _CorruptFieldFunctions()
    ErrorFormatter._emit_templated_error = _record  # pyright: ignore[reportAttributeAccessIssue]
    pipe = Pipe()
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
        pipe.valves.ZDR_ENFORCE = False
        pipe.valves.ALLOW_USER_ZDR_OVERRIDE = True

        captured: list[dict] = []
        callback = _smart_callback(captured, "Response")
        events: list[dict] = []

        async def _emit(event):
            events.append(event)

        with aioresponses() as http:
            http.post("https://openrouter.ai/api/v1/responses", callback=callback, repeat=True)
            http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "openai/gpt-4o", "name": "M"}]},
                repeat=True,
            )
            http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr",
                payload={"data": [{"model_id": "openai/gpt-4o-mini"}]},
                repeat=True,
            )
            result = await pipe.pipe(
                body={
                    "model": "openai/gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                __user__={"id": "u1", "valves": Pipe.UserValves()},
                __request__=None,
                __event_emitter__=_emit,
                __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o"}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
            await _consume_stream(result)
    finally:
        owf.Functions = original
        ErrorFormatter._emit_templated_error = real_emit  # pyright: ignore[reportAttributeAccessIssue]
        await pipe.close()

    assert not captured, (
        "REQUEST_ZDR was present and unparseable, so the user may have opted in to Zero "
        "Data Retention -- but the request was sent to a model that is not ZDR-capable "
        f"anyway: {captured[-1] if captured else None!r}"
    )
    assert reasons, "no restriction was rendered, so the refusal reported nothing"
    _assert_names_the_users_preference(reasons[-1])


async def _reported_restriction_for(stored_row: dict) -> str:
    """Drive one real refusal and hand back the reasons string the card was built from.

    The row is the only thing that varies between calls, and it is stubbed at Open WebUI's
    own reader -- one seam below `_read_user_valves`, which is what classifies it -- so the
    whole chain from the stored value to the rendered label runs for real.
    """
    import open_webui.models.functions as owf
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter

    class _StoredRow:
        async def get_user_valves_by_id_and_user_id(self, _id, _user_id, db=None):
            return dict(stored_row)

    reasons: list[str] = []
    real_emit = ErrorFormatter._emit_templated_error

    async def _record(self, emitter, *, variables=None, **kw):
        variables = variables or {}
        reasons.append(variables.get("restriction_reasons", ""))
        return await real_emit(self, emitter, variables=variables, **kw)

    original = owf.Functions
    pipe = Pipe()
    captured: list[dict] = []
    try:
        owf.Functions = _StoredRow()
        ErrorFormatter._emit_templated_error = _record  # pyright: ignore[reportAttributeAccessIssue]
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
        pipe.valves.ZDR_ENFORCE = False
        pipe.valves.ALLOW_USER_ZDR_OVERRIDE = True

        async def _emit(_event):
            pass

        with aioresponses() as http:
            http.post("https://openrouter.ai/api/v1/responses",
                      callback=_smart_callback(captured, "Response"), repeat=True)
            http.get("https://openrouter.ai/api/v1/models",
                     payload={"data": [{"id": "openai/gpt-4o", "name": "M"}]}, repeat=True)
            http.get("https://openrouter.ai/api/v1/endpoints/zdr",
                     payload={"data": [{"model_id": "openai/gpt-4o-mini"}]}, repeat=True)
            result = await pipe.pipe(
                body={"model": "openai/gpt-4o",
                      "messages": [{"role": "user", "content": "hi"}], "stream": True},
                __user__={"id": "u1", "valves": Pipe.UserValves()},
                __request__=None, __event_emitter__=_emit, __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o"}},
                __tools__=None, __task__=None, __task_body__=None,
            )
            await _consume_stream(result)
    finally:
        owf.Functions = original
        ErrorFormatter._emit_templated_error = real_emit  # pyright: ignore[reportAttributeAccessIssue]
        await pipe.close()

    assert not captured, (
        "the model is not ZDR-capable and ZDR was enforced, so nothing should have been "
        f"sent to OpenRouter: {captured[-1] if captured else None!r}"
    )
    assert reasons, "no restriction was rendered, so the refusal reported nothing"
    return reasons[-1]


@pytest.mark.asyncio
async def test_a_lost_zdr_answer_is_not_reported_as_the_answer_the_user_gave():
    """The two causes end in the same card, and the card must not merge them.

    One user ticked the box. The other has a stored value that will not parse, so what
    they chose is unknown and ZDR is enforced to stay on the safe side of the guess.
    Telling the second user they asked for this hides the only fact that would let them
    fix it -- their saved answer is damaged -- and sends the operator looking for a
    preference nobody set.

    Both arms are rendered in one test and compared to EACH OTHER. Each arm on its own
    only shows that the user's own label is named, and both causes name it, so a single
    arm passes just as happily when the two are collapsed onto one reason.
    """
    from open_webui_openrouter_pipe import Pipe

    chose_it = await _reported_restriction_for({"REQUEST_ZDR": True})
    lost_it = await _reported_restriction_for({"REQUEST_ZDR": {"not": "a boolean"}})

    _assert_names_the_users_preference(chose_it)
    _assert_names_the_users_preference(lost_it)

    theirs = Pipe.UserValves.model_fields["REQUEST_ZDR"].title
    assert theirs, "the user's settings panel shows no label for REQUEST_ZDR"
    assert chose_it != lost_it, (
        f"both causes reported {chose_it!r}. A user whose stored answer could not be read "
        "is being told they chose Zero Data Retention, and there is nothing in the card "
        "that says otherwise."
    )
    assert chose_it == theirs, (
        f"the user who really did tick the box was told {chose_it!r} rather than the plain "
        f"label {theirs!r}, so their own setting now reads as a fault"
    )
    assert lost_it.startswith(theirs) and len(lost_it) > len(theirs), (
        f"the lost answer reported {lost_it!r}, which does not extend the label {theirs!r} "
        "the user's panel shows -- so it names either the wrong setting or no setting"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reader_is_async", [True, False], ids=["async-reader", "sync-reader"])
async def test_the_valve_row_is_read_whether_the_owui_api_is_sync_or_async(reader_is_async):
    """A sync reader must not make the ZDR gate deny every request.

    `Functions.get_user_valves_by_id_and_user_id` is `async` in every Open WebUI I can
    check, but the manifest floor is 0.9.1 and the package carries 57 other bare awaits
    on Open WebUI model APIs. Those degrade when the shape differs; THIS one denies --
    a TypeError here reports the row unreadable, and an unreadable row enforces ZDR,
    which on a default install (ALLOW_USER_ZDR_OVERRIDE=True, ZDR_ENFORCE=False) turns
    every request for a non-ZDR model into "Model restricted" instead of an answer.

    Both arms, because a tolerant call that only ever sees one shape proves nothing
    about the other.
    """
    import open_webui.models.functions as owf

    from open_webui_openrouter_pipe import Pipe

    async def _async_reader(_id, _uid, db=None):
        return {"REQUEST_ZDR": True}

    def _sync_reader(_id, _uid, db=None):
        return {"REQUEST_ZDR": True}

    class _Reader:
        get_user_valves_by_id_and_user_id = staticmethod(
            _async_reader if reader_is_async else _sync_reader
        )

    original = owf.Functions
    owf.Functions = _Reader()
    pipe = Pipe()
    try:
        stored = await pipe._stored_user_valves(
            {"id": "u1", "valves": Pipe.UserValves()}
        )
    finally:
        owf.Functions = original
        await pipe.close()

    assert isinstance(stored, dict) and stored.get("REQUEST_ZDR") is True, (
        f"a {'async' if reader_is_async else 'sync'} valve reader did not yield the "
        f"stored row: {stored!r}. A bare await on a sync reader raises TypeError, which "
        "returns the instance Open WebUI supplied instead -- and that one is "
        "default-constructed whenever its own parse failed, so the user's saved "
        "preference is silently replaced by the default."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "expect_refusal"),
    [
        ({"functions": {"valves": {"PIPE_ID": "gAAAAAB_undecodable"}}}, True),
        ({"ui": {}}, False),
        ({"functions": {"valves": {"PIPE_ID": {}}}}, False),
    ],
    ids=["blob-will-not-decode", "never-saved-any-valves", "unencrypted-empty-dict"],
)
async def test_an_undecodable_user_valve_blob_enforces_zdr(settings, expect_refusal):
    """`{}` from Open WebUI means "nothing stored" OR "the blob failed to decrypt".

    Three arms over one drive, so no constant satisfies it: a hardcoded enforce fails
    arms 2 and 3, a hardcoded allow fails arm 1, and `raw is not None` -- which treats an
    unencrypted install's plain dict as ciphertext -- fails arm 3. The ciphertext travels
    on __user__["settings"], which is what tells the cases apart without a second row
    fetch; `test_the_stored_user_valves_are_read_once_per_request` still reads exactly 1.
    """
    import open_webui.models.functions as owf
    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import Pipe
    from open_webui_openrouter_pipe.core.config import EncryptedStr
    from open_webui_openrouter_pipe.core.error_formatter import ErrorFormatter

    class _EmptyRead:
        async def get_user_valves_by_id_and_user_id(self, _id, _user_id, db=None):
            return {}

    reasons: list[str] = []
    real_emit = ErrorFormatter._emit_templated_error

    async def _record(self, emitter, *, variables=None, **kw):
        variables = variables or {}
        reasons.append(variables.get("restriction_reasons", ""))
        return await real_emit(self, emitter, variables=variables, **kw)

    original = owf.Functions
    owf.Functions = _EmptyRead()
    ErrorFormatter._emit_templated_error = _record  # pyright: ignore[reportAttributeAccessIssue]
    pipe = Pipe()
    captured: list[dict] = []
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
        pipe.valves.ZDR_ENFORCE = False
        pipe.valves.ALLOW_USER_ZDR_OVERRIDE = True
        blob = settings.get("functions", {}).get("valves")
        if blob:
            blob[pipe.id] = blob.pop("PIPE_ID")

        async def _emit(_event):
            pass

        with aioresponses() as http:
            http.post("https://openrouter.ai/api/v1/responses",
                      callback=_smart_callback(captured, "Response"), repeat=True)
            http.get("https://openrouter.ai/api/v1/models",
                     payload={"data": [{"id": "openai/gpt-4o", "name": "M"}]}, repeat=True)
            http.get("https://openrouter.ai/api/v1/endpoints/zdr",
                     payload={"data": [{"model_id": "openai/gpt-4o-mini"}]}, repeat=True)
            result = await pipe.pipe(
                body={"model": "openai/gpt-4o",
                      "messages": [{"role": "user", "content": "hi"}], "stream": True},
                __user__={"id": "u1", "valves": Pipe.UserValves(), "settings": settings},
                __request__=None, __event_emitter__=_emit, __event_call__=None,
                __metadata__={"model": {"id": "openai/gpt-4o"}},
                __tools__=None, __task__=None, __task_body__=None,
            )
            await _consume_stream(result)
    finally:
        owf.Functions = original
        ErrorFormatter._emit_templated_error = real_emit  # pyright: ignore[reportAttributeAccessIssue]
        await pipe.close()

    if expect_refusal:
        assert not captured, (
            "the user has a stored valve blob that will not decode, so their REQUEST_ZDR "
            "answer is unknown -- yet the request went to a model that is not ZDR-capable"
        )
        assert reasons, "no restriction was rendered, so the refusal reported nothing"
        _assert_names_the_users_preference(reasons[-1])
    else:
        assert captured, (
            "a user with no undecodable blob was refused; the guard fires for users who "
            "never opened the valve panel"
        )
        assert (captured[-1].get("provider") or {}).get("zdr") is not True


def test_the_stored_valve_row_is_classified_in_exactly_one_place():
    """One caller, so the undecodable-blob rule cannot be forgotten by a new one.

    The classification lives in `Pipe._read_user_valves`, not at the call sites: a
    second call site pairing `_stored_user_valves` with `parse_user_valves` directly
    would read an undecodable blob as "nothing stored" again, and nothing else in the
    suite would notice. Keyed on the enclosing function, so moving the method is fine
    and duplicating the pairing is not.
    """
    import ast
    import pathlib

    from tests.package_sources import parsed_sources

    package = pathlib.Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"
    callers = set()

    def _visit(node, rel, owner):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            owner = node.name
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name == "_stored_user_valves":
                callers.add((rel, owner or "<module>"))
        for child in ast.iter_child_nodes(node):
            _visit(child, rel, owner)

    for path, _source, tree in parsed_sources("open_webui_openrouter_pipe"):
        _visit(tree, path.relative_to(package.parent).as_posix(), None)

    assert callers == {("open_webui_openrouter_pipe/pipe.py", "_read_user_valves")}, (
        "the stored valve row is read from somewhere other than _read_user_valves, so "
        f"that reader does not get the undecodable-blob classification: {sorted(callers)}"
    )


def test_a_supplied_valves_instance_carries_only_the_fields_the_user_set():
    """`model_fields_set` is what `_merge_valves` reads to decide "the user chose this".

    Open WebUI hands the pipe a `UserValves` instance built from the base class. When a
    plugin contributes a user valve, `Pipe.UserValves` becomes a `create_model` subclass
    while Open WebUI keeps constructing the base -- so `isinstance(raw, model)` is False
    and the instance is round-tripped through a dict. A bare `model_dump()` emits every
    default as well, `model_validate` then marks all of them explicitly set, and one
    field the user actually chose turns into eleven overrides of the admin's globals.

    Two distinct inputs over one path, so no constant answer satisfies both arms.
    """
    from pydantic import create_model

    from open_webui_openrouter_pipe.core.config import UserValves, parse_user_valves

    extended = create_model("UserValves", __base__=UserValves, PLUGIN_X=(bool, False))

    from_instance, _ = parse_user_valves(UserValves(REQUEST_ZDR=True), model=extended)
    assert from_instance.model_fields_set == {"REQUEST_ZDR"}, (
        f"the user set one field; {sorted(from_instance.model_fields_set)} came back as "
        "explicitly set. Every extra name silently overrides the admin's global valve."
    )

    from_mapping, _ = parse_user_valves(
        {"PLUGIN_X": True, "REQUEST_ZDR": True}, model=extended
    )
    assert from_mapping.model_fields_set == {"PLUGIN_X", "REQUEST_ZDR"}, (
        f"a mapping naming two fields yielded {sorted(from_mapping.model_fields_set)}; "
        "the plugin-contributed field must survive the round trip"
    )


@pytest.mark.asyncio
async def test_a_decoded_valve_row_is_never_reported_unreadable():
    """The common case on any encryption-enabled host: decoded fine, ciphertext present.

    Open WebUI stores function valves as ciphertext under
    `user.settings.functions.valves[<pipe id>]` and hands the pipe `user.model_dump()`,
    so EVERY user who has ever saved a valve arrives with a non-empty decode AND a
    non-empty blob. Without the `stored != {}` guard those users are all classified
    unreadable, every field lands in `rejected`, and the orchestrator's
    `"REQUEST_ZDR" in rejected` fail-safe then refuses every non-ZDR model for them --
    on default valves, which is the configuration that reaches it.

    The three sibling arms all drive a reader returning `{}`, so none of them can see
    this: the guard they exercise is the one that fires, not the one that declines to.
    """
    import open_webui.models.functions as owf

    from open_webui_openrouter_pipe import Pipe

    class _DecodesFine:
        async def get_user_valves_by_id_and_user_id(self, _id, _user_id, db=None):
            return {"REQUEST_ZDR": True}

    original = owf.Functions
    owf.Functions = _DecodesFine()
    pipe = Pipe()
    try:
        _valves, rejected = await pipe._read_user_valves(
            {
                "id": "u1",
                "valves": Pipe.UserValves(),
                "settings": {
                    "functions": {"valves": {pipe.id: "gAAAAAB_real_ciphertext"}}
                },
            }
        )
    finally:
        owf.Functions = original
        await pipe.close()

    assert rejected == [], (
        f"a row that decoded to a real value was reported unreadable: {rejected}. On a "
        "host with valve encryption on that is every user who has ever saved a valve, "
        "and each of them is then refused every non-ZDR model."
    )
