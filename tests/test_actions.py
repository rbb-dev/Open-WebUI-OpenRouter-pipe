"""Tests for the pipe_dashboard action registry + dispatcher (authorize-first)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from open_webui_openrouter_pipe.core.config import EncryptedStr, Valves
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import actions


def _user(uid="u1", role="user"):
    return SimpleNamespace(id=uid, role=role)


@pytest.fixture(autouse=True)
def _reset():
    saved = dict(actions.ACTIONS)
    actions._rate_state.clear()
    yield
    actions.ACTIONS.clear()
    actions.ACTIONS.update(saved)
    actions._rate_state.clear()


@pytest.mark.asyncio
async def test_whoami_read(monkeypatch):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=False))
    status, payload = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "whoami", {})
    assert status == 200
    assert payload["result"]["can_view"] is True
    assert payload["result"]["can_act"] is False


@pytest.mark.asyncio
async def test_forbidden_before_validate(monkeypatch):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=False))
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=False))
    status, payload = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "echo", {"bad": 1})
    assert status == 403  # authorize-first: not 400/404


@pytest.mark.asyncio
async def test_unknown_action_after_auth(monkeypatch):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    status, payload = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "nope", {})
    assert status == 404


@pytest.mark.asyncio
async def test_unknown_action_forbidden_when_unauthorized(monkeypatch):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=False))
    status, payload = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "nope", {})
    assert status == 403  # unknown probed by an unauthorized user is 403, not 404


@pytest.mark.asyncio
async def test_echo_write_grant(monkeypatch):
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=True))
    status, payload = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "echo", {"message": "hi"})
    assert status == 200
    assert payload["result"]["message"] == "hi"


@pytest.mark.asyncio
async def test_echo_bad_args(monkeypatch):
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=True))
    status, payload = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "echo", {})
    assert status == 400


@pytest.mark.asyncio
async def test_rate_limited(monkeypatch):
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=True))
    monkeypatch.setattr(actions, "_rate_limited", lambda uid, name: True)
    status, _ = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "echo", {"message": "x"})
    assert status == 429


@pytest.mark.asyncio
async def test_handler_error_500(monkeypatch):
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=True))

    @actions.register_action("boom", permission="write")
    async def boom(pipe, user, args):
        raise RuntimeError("x")

    status, _ = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "boom", {})
    assert status == 500


@pytest.mark.asyncio
async def test_read_action_uses_can_view(monkeypatch):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=False))

    @actions.register_action("peek", permission="read")
    async def peek(pipe, user, args):
        return {"seen": True}

    status, payload = await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "peek", {})
    assert status == 200 and payload["result"]["seen"] is True


@pytest.mark.asyncio
async def test_write_args_audited_read_args_omitted(monkeypatch):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=True))
    calls = []
    monkeypatch.setattr(
        actions, "_audit",
        lambda user, name, outcome, client_ip, args=None: calls.append((name, outcome, client_ip, args)),
    )
    await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "echo", {"message": "secret"}, client_ip="9.9.9.9")
    await actions.dispatch_action(SimpleNamespace(id="p"), _user(), "whoami", {"x": 1})
    echo_ok = next(c for c in calls if c[0] == "echo" and c[1] == "ok")
    whoami_ok = next(c for c in calls if c[0] == "whoami" and c[1] == "ok")
    assert echo_ok[3] == {"message": "secret"}  # write args logged
    assert echo_ok[2] == "9.9.9.9"  # client_ip logged
    assert whoami_ok[3] is None  # read args omitted


def test_audit_routine_ok_is_debug_anomaly_is_warning(monkeypatch):
    calls = []
    monkeypatch.setattr(actions.logger, "debug", lambda *a, **k: calls.append("debug"))
    monkeypatch.setattr(actions.logger, "warning", lambda *a, **k: calls.append("warning"))
    actions._audit(SimpleNamespace(id="u"), "usage_stats", "ok", "1.2.3.4")
    actions._audit(SimpleNamespace(id="u"), "usage_stats", "forbidden", "1.2.3.4")
    assert calls == ["debug", "warning"]


def test_scrub_strips_newlines_and_truncates():
    assert actions._scrub("a\r\nb\nc") == "a  b c"  # CR/LF -> space (no forged log lines)
    assert actions._scrub("x" * 500) == "x" * 200
    assert "\n" not in actions._scrub("evil\naction")
    assert actions._scrub(None) == "None"


class _FakeFunctions:
    def __init__(self, rev=1000, valves=None):
        self.rev = rev
        self.saved = None
        self.valves = {} if valves is None else valves

    async def get_function_by_id(self, id, db=None):
        return SimpleNamespace(updated_at=self.rev)

    async def get_function_valves_by_id(self, id, db=None):
        return self.valves

    async def update_function_valves_by_id(self, id, valves, db=None):
        # Open WebUI commits the write, so the very next read returns it. A double that
        # kept serving the pre-save subset would let a handler read stale values back and
        # still look correct here.
        self.saved = valves
        self.valves = valves
        self.rev += 1
        return SimpleNamespace(updated_at=self.rev)


@pytest.fixture
def fake_functions(monkeypatch):
    import open_webui.models.functions as owf

    fake = _FakeFunctions()
    monkeypatch.setattr(owf, "Functions", fake)
    return fake


def _config_pipe(**valve_kwargs):
    return SimpleNamespace(id="openrouter", valves=Valves(**valve_kwargs))


@pytest.mark.asyncio
async def test_config_get_reads_secret_set_from_store(monkeypatch, fake_functions):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_functions.valves = {"API_KEY": "sk-live-abc"}
    pipe = _config_pipe()
    snapshot = await actions.ACTIONS["config_get"].handler(pipe, _user(), {})
    by_name = {s["name"]: s for s in snapshot["valves"]}
    assert by_name["API_KEY"]["value"] is None
    assert by_name["API_KEY"]["secret_set"] is True
    assert snapshot["rev"] == 1000
    assert snapshot["drift"]["unenriched"] == []


@pytest.mark.asyncio
async def test_config_get_ignores_stale_pinned_secret(monkeypatch, fake_functions):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    fake_functions.valves = {}
    pipe = _config_pipe(API_KEY="sk-stale-pinned")
    snapshot = await actions.ACTIONS["config_get"].handler(pipe, _user(), {})
    by_name = {s["name"]: s for s in snapshot["valves"]}
    assert by_name["API_KEY"]["secret_set"] is False


@pytest.mark.asyncio
async def test_config_set_persists_edit_and_bumps_rev(fake_functions):
    pipe = _config_pipe()
    result = await actions.ACTIONS["config_set"].handler(
        pipe, _user(), {"edits": {"MAX_CONCURRENT_REQUESTS": 250}, "rev": 1000}
    )
    assert result["saved"] == 1
    assert result["rev"] == 1001
    assert set(fake_functions.saved) == {"MAX_CONCURRENT_REQUESTS"}
    assert fake_functions.saved["MAX_CONCURRENT_REQUESTS"] == 250


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["RATE_LIMIT_TEMPLATE", "CONNECTION_ERROR_TEMPLATE"])
async def test_clearing_a_template_puts_the_original_back_in_the_config_box(fake_functions, name):
    """Save a mangled template, clear the box, save again -- the box shows the built-in text.

    This is the whole affordance end to end through the real config_set/config_get handlers and a
    stand-in valve store: what an admin reads back after clearing is the factory wording, ready to
    edit. Two valves with different factory texts, so a handler that returned one fixed string
    could not pass both.
    """
    factory = Valves.model_fields[name].get_default(call_default_factory=True)
    pipe = _config_pipe()

    async def box_value():
        snapshot = await actions.ACTIONS["config_get"].handler(pipe, _user(), {})
        return {spec["name"]: spec for spec in snapshot["valves"]}[name]["value"]

    await actions.ACTIONS["config_set"].handler(
        pipe, _user(), {"edits": {name: "MANGLED {oops"}, "rev": fake_functions.rev}
    )
    fake_functions.valves = fake_functions.saved
    assert await box_value() == "MANGLED {oops"

    cleared = await actions.ACTIONS["config_set"].handler(
        pipe, _user(), {"edits": {name: "   \n  "}, "rev": fake_functions.rev}
    )
    fake_functions.valves = fake_functions.saved
    assert await box_value() == factory
    assert cleared["values"][name] == factory, (
        "the save response is the only thing the editor sees before it re-renders; without the "
        f"restored text in it the box goes blank and the admin thinks nothing happened: {cleared!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "typed", "stored"),
    [
        ("MAX_CONCURRENT_REQUESTS", "250", 250),
        ("MODEL_ID", "anthropic/*", "anthropic/*"),
    ],
)
async def test_a_save_answers_with_the_value_the_store_now_holds(fake_functions, name, typed, stored):
    """The response carries the coerced, persisted value -- not the raw string that was typed.

    The editor adopts these as its new baseline, so anything the server normalises on the way in
    has to come back or the box keeps showing what was typed. One numeric valve where the typed
    string and the stored value differ in TYPE, one string valve where they do not.
    """
    pipe = _config_pipe()
    result = await actions.ACTIONS["config_set"].handler(
        pipe, _user(), {"edits": {name: typed}, "rev": fake_functions.rev}
    )
    assert result["values"] == {name: stored}, result


@pytest.mark.asyncio
async def test_a_save_never_answers_with_a_secret(monkeypatch, fake_functions):
    """A secret is write-only to the browser; echoing it back would hand it to anyone watching."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pipe = _config_pipe()
    result = await actions.ACTIONS["config_set"].handler(
        pipe,
        _user(),
        {
            "edits": {"API_KEY": "sk-brand-new", "MODEL_ID": "anthropic/claude-sonnet-*"},
            "rev": fake_functions.rev,
        },
    )
    assert "API_KEY" not in result["values"], result
    assert result["values"] == {"MODEL_ID": "anthropic/claude-sonnet-*"}, result
    assert "sk-brand-new" not in repr(result), result


@pytest.mark.asyncio
async def test_config_set_preserves_unedited_secret(fake_functions):
    fake_functions.valves = {"API_KEY": "sk-keep-me"}
    pipe = _config_pipe()
    await actions.ACTIONS["config_set"].handler(
        pipe, _user(), {"edits": {"MODEL_ID": "anthropic/*"}, "rev": 1000}
    )
    assert set(fake_functions.saved) == {"API_KEY", "MODEL_ID"}
    assert EncryptedStr.decrypt(fake_functions.saved["API_KEY"]) == "sk-keep-me"


@pytest.mark.asyncio
async def test_config_set_aborts_when_current_read_fails(fake_functions):
    fake_functions.valves = None
    pipe = _config_pipe()
    with pytest.raises(Exception):
        await actions.ACTIONS["config_set"].handler(
            pipe, _user(), {"edits": {"MODEL_ID": "x"}, "rev": 1000}
        )
    assert fake_functions.saved is None


@pytest.mark.asyncio
async def test_config_set_conflict_returns_fresh_and_skips_write(fake_functions):
    fake_functions.rev = 2000
    pipe = _config_pipe()
    result = await actions.ACTIONS["config_set"].handler(
        pipe, _user(), {"edits": {"MODEL_ID": "x"}, "rev": 1000}
    )
    assert result["conflict"] is True
    assert result["rev"] == 2000
    assert fake_functions.saved is None


@pytest.mark.asyncio
async def test_config_set_treats_an_unreadable_rev_as_a_conflict(fake_functions, caplog):
    """A rev the server cannot read must block the write, not wave it through."""
    import logging as _logging

    pipe = _config_pipe()

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("database unavailable")

    original = fake_functions.get_function_by_id
    fake_functions.get_function_by_id = _boom
    try:
        with caplog.at_level(_logging.WARNING):
            result = await actions.ACTIONS["config_set"].handler(
                pipe, _user(), {"edits": {"MODEL_ID": "x"}, "rev": 1000}
            )
    finally:
        fake_functions.get_function_by_id = original

    assert result["conflict"] is True, result
    assert fake_functions.saved is None, "the write proceeded without a rev check"
    assert any(
        "concurrent-edit protection is unavailable" in m for m in caplog.messages
    ), caplog.messages


@pytest.mark.asyncio
async def test_config_set_invalid_value_raises_before_write(fake_functions):
    pipe = _config_pipe()
    with pytest.raises(Exception):
        await actions.ACTIONS["config_set"].handler(
            pipe, _user(), {"edits": {"REASONING_EFFORT": "bogus"}, "rev": 1000}
        )
    assert fake_functions.saved is None


@pytest.mark.asyncio
async def test_config_set_forbidden_for_non_operator(monkeypatch, fake_functions):
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=False))
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    status, _ = await actions.dispatch_action(
        _config_pipe(), _user(), "config_set", {"edits": {"MODEL_ID": "x"}, "rev": 1000}
    )
    assert status == 403
    assert fake_functions.saved is None


@pytest.mark.asyncio
async def test_config_get_allowed_for_viewer(monkeypatch, fake_functions):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=False))
    status, payload = await actions.dispatch_action(_config_pipe(), _user(), "config_get", {})
    assert status == 200
    assert "valves" in payload["result"]


@pytest.mark.asyncio
async def test_config_set_emits_config_changed_on_save(monkeypatch, fake_functions):
    spy = AsyncMock()
    monkeypatch.setattr(actions, "emit_config_changed", spy)
    result = await actions.ACTIONS["config_set"].handler(
        _config_pipe(), _user(), {"edits": {"MAX_CONCURRENT_REQUESTS": 250}, "rev": 1000}
    )
    spy.assert_awaited_once_with(result["rev"])


@pytest.mark.asyncio
async def test_config_set_no_emit_on_conflict(monkeypatch, fake_functions):
    fake_functions.rev = 2000
    spy = AsyncMock()
    monkeypatch.setattr(actions, "emit_config_changed", spy)
    result = await actions.ACTIONS["config_set"].handler(
        _config_pipe(), _user(), {"edits": {"MODEL_ID": "x"}, "rev": 1000}
    )
    assert result["conflict"] is True
    spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_config_set_no_emit_when_no_edits(monkeypatch, fake_functions):
    spy = AsyncMock()
    monkeypatch.setattr(actions, "emit_config_changed", spy)
    result = await actions.ACTIONS["config_set"].handler(
        _config_pipe(), _user(), {"edits": {}, "rev": 1000}
    )
    assert result["saved"] == 0
    spy.assert_not_awaited()


def test_validate_optional_key_absent_is_valid():
    ok, err = actions._validate({}, {"force": actions.optional(bool)})
    assert ok is True
    assert err == ""


def test_validate_optional_key_present_right_type():
    ok, _ = actions._validate({"force": True}, {"force": actions.optional(bool)})
    assert ok is True


def test_validate_optional_key_present_wrong_type():
    ok, err = actions._validate({"force": "yes"}, {"force": actions.optional(bool)})
    assert ok is False
    assert "force" in err


def test_validate_required_tuple_union():
    schema = {"rev": (int, str)}
    assert actions._validate({"rev": 5}, schema)[0] is True
    assert actions._validate({"rev": "5"}, schema)[0] is True
    assert actions._validate({"rev": 5.0}, schema)[0] is False
    assert actions._validate({}, schema)[0] is False


def test_validate_optional_tuple_union():
    schema = {"file_id": str, "rev": actions.optional((int, str))}
    assert actions._validate({"file_id": "f"}, schema)[0] is True
    assert actions._validate({"file_id": "f", "rev": "x"}, schema)[0] is True
    assert actions._validate({"file_id": "f", "rev": 1.5}, schema)[0] is False


def test_validate_required_plain_type_unchanged():
    schema = {"message": str}
    assert actions._validate({"message": "hi"}, schema)[0] is True
    assert actions._validate({}, schema)[0] is False
    assert actions._validate({"message": 3}, schema)[0] is False


# ── update actions ───────────────────────────────────────────────────────────


class _FakeUpdateService:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.raise_error: Exception | None = None
        self.row_valves = {}
        self.stored_read_ok = True

    async def _row_valves(self):
        return dict(self.row_valves)

    async def _row_valves_checked(self):
        return dict(self.row_valves), self.stored_read_ok

    async def check(self, *, force=False):
        self.calls.append(("check", {"force": force}))
        if self.raise_error:
            raise self.raise_error
        return {"enabled": True, "update_available": True, "rev": 7}

    async def apply(self, args, *, actor, actor_id, request):
        self.calls.append(("apply", {"args": dict(args), "actor": actor, "actor_id": actor_id, "request": request}))
        if self.raise_error:
            raise self.raise_error
        return {"ok": True, "from_version": "a", "to_version": "b"}

    async def restore(self, args, *, actor, actor_id, request):
        self.calls.append(("restore", {"args": dict(args), "request": request}))
        if self.raise_error:
            raise self.raise_error
        return {"ok": True, "from_version": "b", "to_version": "a"}

    async def snapshot_delete(self, args):
        self.calls.append(("snapshot_delete", {"args": dict(args)}))
        if self.raise_error:
            raise self.raise_error
        return {"ok": True, "snapshots": []}


def _update_pipe(svc, enabled=True):
    svc.row_valves = {"PIPE_DASHBOARD_UPDATE_ENABLE": enabled}
    valves = SimpleNamespace(PIPE_DASHBOARD_UPDATE_ENABLE=enabled)
    plugin = SimpleNamespace(plugin_id="pipe-dashboard", update_service=svc)
    registry = SimpleNamespace(_plugins=[plugin])
    return SimpleNamespace(id="p", valves=valves, _plugin_registry=registry)


def _req():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))


@pytest.fixture()
def update_env(monkeypatch):
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(actions, "can_act", AsyncMock(return_value=True))
    svc = _FakeUpdateService()
    return SimpleNamespace(svc=svc, pipe=_update_pipe(svc))


@pytest.mark.asyncio
async def test_update_check_delegates_and_passes_force(update_env):
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(role="admin"), "update_check", {"force": True}, request=_req()
    )
    assert status == 200
    assert payload["result"]["update_available"] is True
    assert update_env.svc.calls == [("check", {"force": True})]


@pytest.mark.asyncio
async def test_update_check_force_downgraded_for_non_admin(update_env):
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(role="user"), "update_check", {"force": True}, request=_req()
    )
    assert status == 200
    assert update_env.svc.calls == [("check", {"force": False})]


@pytest.mark.asyncio
async def test_update_gate_uses_row_valves_over_ctx(update_env):
    update_env.pipe.valves.PIPE_DASHBOARD_UPDATE_ENABLE = True

    async def _row_valves_checked():
        return {"PIPE_DASHBOARD_UPDATE_ENABLE": False}, True

    update_env.svc._row_valves_checked = _row_valves_checked
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(role="admin"), "update_check", {}, request=_req()
    )
    assert status == 200
    assert payload["result"] == {"enabled": False}
    assert update_env.svc.calls == []


@pytest.mark.asyncio
async def test_update_check_disabled_short_circuits(update_env):
    update_env.svc.row_valves = {"PIPE_DASHBOARD_UPDATE_ENABLE": False}
    update_env.pipe.valves.PIPE_DASHBOARD_UPDATE_ENABLE = False
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(), "update_check", {}, request=_req()
    )
    assert status == 200
    assert payload["result"] == {"enabled": False}
    assert update_env.svc.calls == []


@pytest.mark.asyncio
async def test_update_gate_denies_when_the_persisted_valve_is_unreadable(update_env, caplog):
    """An operator's disable must not be overridden by an in-memory True on a read failure."""
    import logging as _logging

    # The real UpdateService never raises here -- it catches the DB error itself and
    # falls back to the in-memory valves -- so a stub that raises proves only a branch
    # the real API cannot reach. What it reports instead is stored_read_ok=False.
    update_env.svc.stored_read_ok = False
    update_env.pipe.valves.PIPE_DASHBOARD_UPDATE_ENABLE = True

    with caplog.at_level(_logging.WARNING):
        status, payload = await actions.dispatch_action(
            update_env.pipe, _user(), "update_check", {}, request=_req()
        )

    assert status == 200
    assert payload["result"] == {"enabled": False}, payload
    assert update_env.svc.calls == [], "an update action ran behind an unverifiable gate"
    assert any(
        "refusing update actions" in m for m in caplog.messages
    ), caplog.messages


@pytest.mark.asyncio
async def test_update_writes_require_admin(update_env):
    for name, args in (
        ("update_apply", {"rev": 1}),
        ("update_restore", {"rev": 1, "file_id": "f"}),
        ("update_snapshot_delete", {"file_id": "f", "sha256": "s"}),
    ):
        status, payload = await actions.dispatch_action(
            update_env.pipe, _user(role="user"), name, args, request=_req()
        )
        assert status == 200
        assert payload["result"]["error"] == "forbidden"
    assert update_env.svc.calls == []


@pytest.mark.asyncio
async def test_update_writes_disabled_gate(update_env):
    update_env.svc.row_valves = {"PIPE_DASHBOARD_UPDATE_ENABLE": False}
    update_env.pipe.valves.PIPE_DASHBOARD_UPDATE_ENABLE = False
    for name, args in (
        ("update_apply", {"rev": 1}),
        ("update_restore", {"rev": 1, "file_id": "f"}),
        ("update_snapshot_delete", {"file_id": "f", "sha256": "s"}),
    ):
        actions._rate_state.clear()
        status, payload = await actions.dispatch_action(
            update_env.pipe, _user(role="admin"), name, args, request=_req()
        )
        assert status == 200
        assert payload["result"]["error"] == "disabled"
    assert update_env.svc.calls == []


@pytest.mark.asyncio
async def test_update_apply_happy_passes_request_and_actor(update_env):
    req = _req()
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(uid="boss", role="admin"), "update_apply",
        {"rev": "7", "compressed": True}, request=req,
    )
    assert status == 200
    assert payload["result"]["ok"] is True
    kind, call = update_env.svc.calls[0]
    assert kind == "apply"
    assert call["args"] == {"rev": "7", "compressed": True}
    assert call["actor"] == "boss"
    assert call["actor_id"] == "boss"
    assert call["request"] is req


@pytest.mark.asyncio
async def test_update_apply_maps_update_error_to_result(update_env):
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.update_service import UpdateError

    update_env.svc.raise_error = UpdateError("stale_rev", "row changed")
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(role="admin"), "update_apply", {"rev": 1}, request=_req()
    )
    assert status == 200
    assert payload["result"]["error"] == "stale_rev"
    assert "row changed" in payload["result"]["message"]


@pytest.mark.asyncio
async def test_update_apply_needs_request_fails_closed(update_env):
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(role="admin"), "update_apply", {"rev": 1}, request=None
    )
    assert status == 400
    assert update_env.svc.calls == []


@pytest.mark.asyncio
async def test_update_apply_schema_rejects_bad_compressed(update_env):
    status, _payload = await actions.dispatch_action(
        update_env.pipe, _user(role="admin"), "update_apply",
        {"rev": 1, "compressed": "yes"}, request=_req(),
    )
    assert status == 400


@pytest.mark.asyncio
async def test_update_snapshot_delete_happy(update_env):
    status, payload = await actions.dispatch_action(
        update_env.pipe, _user(role="admin"), "update_snapshot_delete",
        {"file_id": "f", "sha256": "s"}, request=_req(),
    )
    assert status == 200
    assert payload["result"]["ok"] is True
    assert update_env.svc.calls[0][0] == "snapshot_delete"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_rev", [None, 1000, "whatever-the-client-had"])
async def test_a_row_open_webui_cannot_read_blocks_the_write(fake_functions, caplog, client_rev):
    """Open WebUI returns None on a DB fault; it does not raise.

    `Functions.get_function_by_id` is `try: ... except Exception: return None`
    (models/functions.py), so `_current_config_rev`'s except arm is unreachable for a
    real fault. The pipe got None, logged nothing, and `config_get` handed the client
    `rev: null` -- which the shipped client stores and echoes back verbatim. Server-side
    `client_rev` was then None, so `client_rev is not None` was False and the write went
    through with NO concurrent-edit check, which is the state the guard exists to deny.

    Parametrised over what the caller sent, including the None the server itself
    produced: an unreadable revision must block regardless of the caller.
    """
    import logging as _logging

    pipe = _config_pipe()

    async def _missing_row(*_args, **_kwargs):
        return None

    original = fake_functions.get_function_by_id
    fake_functions.get_function_by_id = _missing_row
    try:
        with caplog.at_level(_logging.WARNING):
            result = await actions.ACTIONS["config_set"].handler(
                pipe, _user(), {"edits": {"MODEL_ID": "x"}, "rev": client_rev}
            )
    finally:
        fake_functions.get_function_by_id = original

    assert result.get("conflict") is True, (
        f"client_rev={client_rev!r}: the write was accepted while the server could not "
        "read the stored revision, so a concurrent edit would be silently overwritten"
    )
    assert fake_functions.saved is None, (
        f"client_rev={client_rev!r}: config was persisted despite the conflict"
    )
    assert any("config revision is unknown" in m for m in caplog.messages), (
        "nothing told the operator why the write was refused"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("in_memory_enable", [True, False])
@pytest.mark.parametrize(
    ("outcome", "readable"),
    [
        ("returns-None", False),
        ("returns-empty-dict", True),
        ("returns-a-value", True),
        ("raises", False),
    ],
)
async def test_the_real_service_reports_an_unreadable_valve_read(
    monkeypatch, in_memory_enable, outcome, readable
):
    """Drives the real `_row_valves_checked` across every outcome the store produces.

    The version of this test that existed used a fake which RAISED, and raising is the
    one behaviour Open WebUI's API is guaranteed never to exhibit:
    `Functions.get_function_valves_by_id` catches Exception itself and returns None, and
    `decrypt_valves` returns `{}` for every non-error input including a missing row. So
    a guard keyed on the exception could never fire in production while this test stayed
    green -- the store was unreadable and the update action ran anyway.

    `None` and `{}` are therefore the two rows that matter, and they must give OPPOSITE
    answers: None is a swallowed DB error, `{}` is a healthy read of an absent override.
    A guard that denies on both is as wrong as one that denies on neither, which is why
    "returns-empty-dict" asserts readable rather than being left out. The raising row
    stays as a defensive case for a future Open WebUI that lets one escape.

    Parametrised over the in-memory value because that is the trap: with an in-memory
    True and an unreadable store, the old code returned True and let the action run.
    """
    import open_webui.models.functions as owf

    from open_webui_openrouter_pipe.plugins.pipe_dashboard.update_service import UpdateService

    class _Store:
        async def get_function_valves_by_id(self, _id, db=None):
            if outcome == "raises":
                raise RuntimeError("database unavailable")
            if outcome == "returns-None":
                return None
            if outcome == "returns-empty-dict":
                return {}
            return {"PIPE_DASHBOARD_UPDATE_ENABLE": True}

    monkeypatch.setattr(owf, "Functions", _Store())
    valves = SimpleNamespace(
        PIPE_DASHBOARD_UPDATE_ENABLE=in_memory_enable,
        PIPE_DASHBOARD_UPDATE_AUTO=False,
        PIPE_DASHBOARD_UPDATE_REPO="",
    )
    pipe = SimpleNamespace(id="openrouter", valves=valves)

    svc = UpdateService.__new__(UpdateService)
    svc._pipe = lambda: pipe
    svc._valves = lambda: valves
    svc._functions = lambda: _Store()

    merged, stored_read_ok = await svc._row_valves_checked()
    assert stored_read_ok is readable, (
        f"a store that {outcome} was reported as "
        f"{'readable' if stored_read_ok else 'unreadable'}. Open WebUI returns None for "
        "a swallowed DB error and {} for an absent override; collapsing the two either "
        "lets an unverified in-memory True authorise an update, or denies every update "
        "on a perfectly healthy store."
    )
    expected = True if outcome == "returns-a-value" else in_memory_enable
    assert merged.get("PIPE_DASHBOARD_UPDATE_ENABLE") is expected, (
        "the merged dict should carry the stored value where there is one and the "
        "in-memory fallback otherwise; only the flag says whether it is verified"
    )
