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
    monkeypatch.setattr(actions._pd_actions_log, "debug", lambda *a, **k: calls.append("debug"))
    monkeypatch.setattr(actions._pd_actions_log, "warning", lambda *a, **k: calls.append("warning"))
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
        self.saved = valves
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
