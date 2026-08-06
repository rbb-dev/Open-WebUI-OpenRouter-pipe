"""Tests for the pipe_dashboard authorization chokepoint (authz)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import authz


def _pipe(pid="function_openrouter"):
    return SimpleNamespace(id=pid)


def _seam(*, model=None, read_ok=True, write_ok=False, verified=True, user=None,
          bypass_admin=True, grantee="u1"):
    if model is None:
        model = Mock(user_id="admin1")

    async def check_model_access(u, m, bypass_filter=False):
        if bypass_filter:
            return None
        if not read_ok:
            raise HTTPException(status_code=403)

    def get_verified_user(u):
        if not verified:
            raise HTTPException(status_code=401)
        return u

    users = Mock()
    users.get_user_by_id = AsyncMock(return_value=user)
    models = Mock()
    models.get_model_by_id = AsyncMock(return_value=model)
    grants = Mock()

    async def has_access(*, user_id, resource_type, resource_id, permission="read", **_kw):
        # CONSUMES user_id rather than recording it. A recorder proves which argument
        # was passed and leaves the decision free, so the next widening simply moves to
        # whichever argument the newest recorder forgot -- which is how this file grew
        # three separate recorders. Consuming it means a wrong identity changes the
        # returned boolean, and the ordinary `is False` assertions do the work.
        if user_id != grantee:
            return False
        if (resource_type, resource_id) != ("model", authz.model_id(_pipe())):
            return False
        return {"read": read_ok, "write": write_ok}.get(permission, False)

    grants.has_access = has_access
    return SimpleNamespace(
        Users=users, Models=models, AccessGrants=grants,
        get_verified_user=get_verified_user, check_model_access=check_model_access,
        BYPASS_ADMIN=bypass_admin, BYPASS_MODEL=False,
    )


def test_model_id():
    assert authz.model_id(_pipe("function_x")) == "function_x.pipe-dashboard"
    assert authz.model_id(SimpleNamespace(id=None)) is None


@pytest.mark.asyncio
async def test_can_view_granted(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(read_ok=True, user=user))
    assert await authz.can_view(user, _pipe()) is True


@pytest.mark.asyncio
async def test_can_view_denied_no_grant(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(read_ok=False, user=user))
    assert await authz.can_view(user, _pipe()) is False


@pytest.mark.asyncio
async def test_pending_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="pending")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(verified=False, user=user))
    assert await authz.can_view(user, _pipe()) is False
    assert await authz.can_act(user, _pipe()) is False


@pytest.mark.asyncio
async def test_can_act_write_grant(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(write_ok=True, user=user))
    assert await authz.can_act(user, _pipe()) is True


@pytest.mark.asyncio
async def test_can_act_read_only_user_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(read_ok=True, write_ok=False, model=Mock(user_id="someone_else"), user=user),
    )
    assert await authz.can_act(user, _pipe()) is False


@pytest.mark.asyncio
async def test_owner_can_act(monkeypatch):
    user = SimpleNamespace(id="owner1", role="user")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(write_ok=False, model=Mock(user_id="owner1"), user=user),
    )
    assert await authz.can_act(user, _pipe()) is True


@pytest.mark.asyncio
async def test_admin_bypass_write(monkeypatch):
    user = SimpleNamespace(id="a1", role="admin")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(write_ok=False, model=Mock(user_id="someone_else"), user=user),
    )
    assert await authz.can_act(user, _pipe()) is True


@pytest.mark.asyncio
async def test_none_user_denied(monkeypatch):
    monkeypatch.setattr(authz, "_owui", lambda: _seam())
    assert await authz.can_view(None, _pipe()) is False


@pytest.mark.asyncio
async def test_missing_pipe_id_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(user=user))
    assert await authz.can_view(user, SimpleNamespace(id=None)) is False


@pytest.mark.asyncio
async def test_owui_import_failure_denies(monkeypatch):
    def boom():
        raise ImportError("no owui")

    monkeypatch.setattr(authz, "_owui", boom)
    assert await authz.can_view(SimpleNamespace(id="u1", role="user"), _pipe()) is False


@pytest.mark.asyncio
async def test_write_missing_model_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(model=None, user=user))
    seam = _seam(user=user)
    seam.Models.get_model_by_id = AsyncMock(return_value=None)
    monkeypatch.setattr(authz, "_owui", lambda: seam)
    assert await authz.can_act(user, _pipe()) is False


@pytest.mark.asyncio
async def test_the_operator_check_demands_a_write_grant(monkeypatch):
    """Pins WHICH permission is asked for, not merely the answer.

    Open WebUI's AccessGrants.has_access takes `permission` with a default of "read",
    so an operator check that passes "read" -- or that simply omits the argument --
    still returns True for anyone holding a read grant. Every viewer becomes an
    operator, and nothing about the call looks wrong at the call site.

    Asserting the returned decision cannot catch that: with a read grant present and a
    write grant absent, "asked for read" and "asked for write" differ only in the
    argument. So this records the argument.
    """
    user = SimpleNamespace(id="u1", role="user")
    asked: list[str] = []

    seam = _seam(read_ok=True, write_ok=False, model=Mock(user_id="someone_else"), user=user)
    inner = seam.AccessGrants.has_access

    async def recording(**kwargs):
        asked.append(kwargs.get("permission", "read"))
        return await inner(**kwargs)

    seam.AccessGrants.has_access = recording
    monkeypatch.setattr(authz, "_owui", lambda: seam)

    granted = await authz.can_act(user, _pipe())

    assert asked == ["write"], (
        f"can_act asked Open WebUI for {asked!r}. Anything but 'write' grants operator "
        "rights to every user holding a read grant -- and an omitted permission "
        "defaults to 'read' upstream, so dropping the argument has the same effect."
    )
    assert granted is False


@pytest.mark.asyncio
async def test_the_view_check_does_not_bypass_the_model_acl(monkeypatch):
    """Pins the bypass_filter argument, the read gate's equivalent of `permission`.

    Open WebUI's check_model_access does nothing at all when bypass_filter is True, so
    passing True there admits every verified user regardless of the model's ACL, and
    the call still reads as an access check.
    """
    user = SimpleNamespace(id="u1", role="user")
    asked: list[bool] = []

    seam = _seam(read_ok=False, user=user)
    inner = seam.check_model_access

    async def recording(u, m, bypass_filter=False):
        asked.append(bypass_filter)
        return await inner(u, m, bypass_filter=bypass_filter)

    seam.check_model_access = recording
    monkeypatch.setattr(authz, "_owui", lambda: seam)

    granted = await authz.can_view(user, _pipe())

    assert asked == [False], (
        f"can_view called check_model_access with bypass_filter={asked!r}. True makes "
        "the call a no-op, so every verified user sees the dashboard whatever the "
        "model's ACL says."
    )
    assert granted is False


@pytest.mark.asyncio
async def test_admin_bypass_is_refused_when_the_operator_disables_it(monkeypatch):
    """BYPASS_ADMIN_ACCESS_CONTROL was never exercised as False.

    Upstream defaults it True, so this only bites a deployment that deliberately
    hardens it -- whose setting would then be silently ignored.
    """
    user = SimpleNamespace(id="a1", role="admin")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(write_ok=False, model=Mock(user_id="someone_else"), user=user,
                      bypass_admin=False),
    )
    assert await authz.can_act(user, _pipe()) is False, (
        "an admin was granted operator rights with BYPASS_ADMIN_ACCESS_CONTROL off and "
        "no write grant; the operator's hardening is being ignored"
    )


@pytest.mark.asyncio
async def test_the_operator_check_asks_about_the_dashboard_model(monkeypatch):
    """Pins WHICH resource, not just which permission.

    Recording `permission=` proves the call demands write. It says nothing about what
    the write is on. Swapping resource_type to "pipe" and resource_id to the bare pipe
    id passes every other assertion here, and grants dashboard operator rights to
    anyone holding write on the base pipe -- regardless of the dashboard model's ACL.
    """
    user = SimpleNamespace(id="u1", role="user")
    asked: list[tuple] = []

    seam = _seam(read_ok=True, write_ok=True, model=Mock(user_id="someone_else"), user=user)
    inner = seam.AccessGrants.has_access

    async def recording(**kwargs):
        asked.append((
            kwargs.get("user_id"),
            kwargs.get("resource_type"),
            kwargs.get("resource_id"),
            kwargs.get("permission", "read"),
        ))
        return await inner(**kwargs)

    seam.AccessGrants.has_access = recording
    monkeypatch.setattr(authz, "_owui", lambda: seam)

    await authz.can_act(user, _pipe())

    assert asked == [("u1", "model", authz.model_id(_pipe()), "write")], (
        f"can_act asked {asked!r}. All four must be right: the REQUESTING user, the "
        "dashboard MODEL id, and a WRITE permission. Any one wrong lets a grant "
        "belonging to someone else, or on something else, confer operator rights."
    )


def test_the_open_webui_seam_resolves_the_real_symbols():
    """Every other test in this file monkeypatches `_owui` wholesale.

    So the imports inside it, and the signatures behind them, are executed by nothing.
    Swapping check_model_access for a same-named function without a `bypass_filter`
    parameter raises TypeError into the blanket handler and denies every user, with the
    whole suite green.

    The real modules are loaded FROM DISK rather than imported, because conftest
    installs stub packages for `open_webui.models` with an empty `__path__` -- a plain
    import here resolves the stub and the check skips itself into uselessness.
    """
    import ast
    import sysconfig
    from pathlib import Path as _Path

    root = None
    for base in (sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]):
        candidate = _Path(base) / "open_webui"
        if candidate.is_dir():
            root = candidate
            break
    assert root is not None, "the installed open_webui package could not be located on disk"

    def _params(path: _Path, qualname: str) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        want = qualname.split(".")[-1]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == want:
                a = node.args
                return [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
        raise AssertionError(f"{qualname} not found in {path}")

    cma = _params(root / "utils" / "access_control" / "__init__.py", "check_model_access")
    assert "bypass_filter" in cma, (
        f"check_model_access{cma} has no bypass_filter parameter, so authz's call raises "
        "TypeError into its blanket handler and every user is denied"
    )

    ha = _params(root / "models" / "access_grants.py", "has_access")
    for name in ("user_id", "resource_type", "resource_id", "permission"):
        assert name in ha, f"AccessGrants.has_access{ha} lost {name!r}"

    for rel, symbol in (
        (_Path("models") / "models.py", "get_model_by_id"),
        (_Path("models") / "users.py", "get_user_by_id"),
        (_Path("utils") / "auth.py", "get_verified_user"),
    ):
        _params(root / rel, symbol)


def test_authz_imports_each_symbol_from_the_module_that_defines_it():
    """The signature check above proves Open WebUI's API is intact.

    It does not prove authz points at it. `open_webui.utils.models` also exports a
    `check_model_access`, with no `bypass_filter` parameter -- importing that one
    raises TypeError into the blanket handler and denies every user, and a
    signature check on the RIGHT module stays green throughout.
    """
    import ast
    from pathlib import Path as _Path

    src = (
        _Path(__file__).resolve().parents[1]
        / "open_webui_openrouter_pipe" / "plugins" / "pipe_dashboard" / "authz.py"
    ).read_text(encoding="utf-8")

    imported: dict[str, str] = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("open_webui"):
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module or ""

    expected = {
        "check_model_access": "open_webui.utils.access_control",
        "AccessGrants": "open_webui.models.access_grants",
        "Models": "open_webui.models.models",
        "Users": "open_webui.models.users",
        "get_verified_user": "open_webui.utils.auth",
        "BYPASS_ADMIN_ACCESS_CONTROL": "open_webui.config",
        "BYPASS_MODEL_ACCESS_CONTROL": "open_webui.env",
    }
    wrong = {
        name: (imported.get(name), module)
        for name, module in expected.items()
        if imported.get(name) != module
    }
    assert not wrong, (
        "authz imports these from a different module than the one that defines the "
        f"signature it calls (found, expected): {wrong}. A same-named symbol elsewhere "
        "raises TypeError into the blanket handler, which denies every user."
    )


@pytest.mark.asyncio
async def test_a_write_grant_held_by_someone_else_confers_nothing(monkeypatch):
    """The decision, not the arguments.

    A grant exists and is a write grant -- it just belongs to a different principal.
    Asking Open WebUI about the wrong identity (say the model's own owner, which is
    true by construction) makes every caller an operator, and the call site still reads
    correctly: right permission, right resource, right model id.
    """
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(write_ok=True, grantee="somebody_else",
                      model=Mock(user_id="owner1"), user=user),
    )
    assert await authz.can_act(user, _pipe()) is False, (
        "can_act granted operator rights from a write grant belonging to a different "
        "principal -- it is not asking about the requesting user"
    )
