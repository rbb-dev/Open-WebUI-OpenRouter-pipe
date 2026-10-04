"""Open WebUI's auth chain, transcribed, as the seam the dashboard action route binds to.

The action route is mounted with two FastAPI dependencies: a bare-bearer precondition
and Open WebUI's own `get_verified_user` (`utils/auth.py:175`, `:393`, `:570-576`). A
test that wants to reach the endpoint therefore has to put a user where Open WebUI puts
one -- `request.state.user` (`utils/auth.py:434`, `:486`) -- and the only honest way to
do that without a live Open WebUI is to install the dependency OWUI's own code would
resolve to.

Importing `open_webui.utils.auth` for real is not an option here: on its own it runs
Open WebUI's DB bootstrap (`Creating new primary key with 'id' and 'user_id'`), which
every collection on the box would pay for. So the arms are transcribed -- the
`HTTPBearer(auto_error=False)` credential, the cookie fallback (`utils/auth.py:408-409`),
the `request.state.token` fallback (`:412-413`), the `sk-` API-key arm (`:419-434`,
`:509-565`), the trusted-identity comparison and the `{user, admin}` role gate.

The trusted-identity valve is read from the `open_webui.env` module in `sys.modules` at
request time rather than captured, so a row that binds `WEBUI_AUTH_TRUSTED_EMAIL_HEADER`
the way an operator's process binds it -- on that module -- is what the comparison reads.
Passing `trusted_header=` binds it the same way.

The module carries `decode_token` and `is_valid_token` beside the dependency names so a
row can state a property -- a cookie-only request is refused before any identity work --
against a route bound either way.

Nothing here is a production helper; it is the seam every test in this group mounts the
route onto.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import AsyncMock

from fastapi import BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer

#: `open_webui.constants.ERROR_MESSAGES`, the strings a refusal on this route can carry.
UNAUTHORIZED = "401 Unauthorized"
ACCESS_PROHIBITED = (
    "You do not have permission to access this resource. "
    "Please contact your administrator for assistance."
)
API_KEY_NOT_ALLOWED = "Use of API key is not enabled in the environment."
INVALID_TOKEN = "Invalid token"

TRUSTED_EMAIL_HEADER = "WEBUI_AUTH_TRUSTED_EMAIL_HEADER"

SIGNING_KEY = "b1300-probe-signing-key-32-bytes-long"

DEFAULT_USER = types.SimpleNamespace(id="u1", role="user", email="u1@example.com")
DEFAULT_API_KEY_USER = types.SimpleNamespace(id="key1", role="admin", email="key@example.com")


def env_module() -> Any:
    """The `open_webui.env` this process has bound, created on first use."""
    env = sys.modules.get("open_webui.env")
    if env is None:
        env = types.ModuleType("open_webui.env")
        sys.modules["open_webui.env"] = env
    return env


def token_for(user_id: str) -> str:
    """A credential the transcribed `decode_token` accepts, naming `user_id`."""
    import jwt

    return jwt.encode({"id": user_id}, SIGNING_KEY, algorithm="HS256")


def install_open_webui_auth(
    monkeypatch: Any,
    *,
    user: Any = DEFAULT_USER,
    valid: bool = True,
    api_keys: bool = False,
    api_key_user: Any = DEFAULT_API_KEY_USER,
    trusted_header: str | None = None,
    entered: list[str] | None = None,
    decode: Any = None,
    decoded: list[str] | None = None,
) -> types.ModuleType:
    """Install `open_webui.utils.auth` and `open_webui.models.users` as OWUI's own.

    `decoded` receives every credential the transcribed `decode_token` is handed, which
    is how a header-parse row proves the token that survived the parse is the one the
    caller wrote; `decode` replaces the decoder for a row about a token no real
    decoder would accept. A `user` that is callable is resolved once per request, for a
    row whose principals differ per call.
    """
    seen = entered if entered is not None else []
    recorded = decoded if decoded is not None else []
    auth: Any = types.ModuleType("open_webui.utils.auth")
    auth.bearer_security = HTTPBearer(auto_error=False)
    fired: list[Any] = []
    resolve_user: Any = user if callable(user) else (lambda _request: user)
    holder: dict[str, Any] = {}
    users_mod: Any = sys.modules.get("open_webui.models.users")
    if users_mod is None:
        users_mod = types.ModuleType("open_webui.models.users")
        sys.modules["open_webui.models.users"] = users_mod
    monkeypatch.setattr(
        users_mod,
        "Users",
        types.SimpleNamespace(
            update_last_active_by_id=AsyncMock(return_value=True),
        ),
        raising=False,
    )

    def jwt_decode_token(token: str) -> Any:
        import jwt

        return jwt.decode(token, SIGNING_KEY, algorithms=["HS256"])

    def decode_token(token: str) -> Any:
        recorded.append(token)
        if decode is not None:
            return decode(token)
        return jwt_decode_token(token)

    async def is_valid_token(_data: Any, _redis: Any) -> bool:
        return valid

    async def get_current_user(
        request: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        auth_token: Any = Depends(auth.bearer_security),
    ) -> Any:
        seen.append("get_current_user")
        token = auth_token.credentials if auth_token is not None else None
        if token is None and "token" in request.cookies:
            token = request.cookies.get("token")
        if token is None and getattr(request.state, "token", None):
            token = request.state.token.credentials
        if token is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if token.startswith("sk-"):
            holder["user"] = api_key_user
            if api_key_user is None:
                raise HTTPException(status_code=401, detail=INVALID_TOKEN)
            if not api_keys:
                raise HTTPException(status_code=403, detail=API_KEY_NOT_ALLOWED)
            await users_mod.Users.update_last_active_by_id(api_key_user.id)
            request.state.user = api_key_user
            request.state.auth_type = "api_key"
            return api_key_user
        try:
            data = decode_token(token)
        except Exception:
            data = None
        if not data or not data.get("id"):
            raise HTTPException(status_code=401, detail=UNAUTHORIZED)
        if not valid:
            raise HTTPException(status_code=401, detail=UNAUTHORIZED)
        resolved = resolve_user(request)
        if resolved is None:
            raise HTTPException(status_code=401, detail=UNAUTHORIZED)
        valve = getattr(env_module(), TRUSTED_EMAIL_HEADER, "")
        if valve:
            claimed = request.headers.get(valve, "").lower()
            if claimed and resolved.email != claimed:
                raise HTTPException(status_code=401, detail=UNAUTHORIZED)
        holder["user"] = resolved
        fired.append(
            asyncio.ensure_future(users_mod.Users.update_last_active_by_id(resolved.id))
        )
        request.state.user = resolved
        request.state.auth_type = "jwt"
        return resolved

    def get_verified_user(user: Any = Depends(get_current_user)) -> Any:
        seen.append("get_verified_user")
        if user.role not in ("user", "admin"):
            raise HTTPException(status_code=401, detail=ACCESS_PROHIBITED)
        return user

    auth.fired_tasks = fired
    auth.decode_token = decode_token
    auth.is_valid_token = is_valid_token
    auth.get_current_user = get_current_user
    auth.get_verified_user = get_verified_user

    async def get_user_by_id(_id: str) -> Any:
        return holder.get("user")

    async def get_user_by_api_key(_key: str) -> Any:
        return api_key_user

    users_mod.Users.get_user_by_id = get_user_by_id
    users_mod.Users.get_user_by_api_key = get_user_by_api_key
    if trusted_header is not None:
        monkeypatch.setattr(env_module(), TRUSTED_EMAIL_HEADER, trusted_header, raising=False)
    monkeypatch.setitem(sys.modules, "open_webui.utils.auth", auth)
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_mod)
    return auth