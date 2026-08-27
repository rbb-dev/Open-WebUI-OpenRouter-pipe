"""Every outbound HTTP request in the package is accounted for.

The model-icon download was the one place that issued a GET to a catalog-supplied URL
without an address check, while three other externally-supplied URLs went through the
address gate. Nothing said the fourth was missing, because nothing enumerated the fetch
sites at all.

This enumerates them. The sites are DERIVED, not listed: a module's HTTP session names
are worked out from its own annotations and assignments, and every HTTP-verb call on one
of those names is a site. A new `session.get` therefore appears here the moment it is
written, and must be either inside a function that vets its address or recorded below
with a reason someone had to type.

What the derivation is, precisely:

- A name is a session if it is annotated `aiohttp.ClientSession` / `httpx.AsyncClient`
  (through `| None`, `Optional[...]` and string annotations), if it is bound by such a
  constructor, or if it is assigned from a name already known to be one. Attribute
  targets count, so `self._session = session` in `__init__` carries through. The same
  derivation, over a different type table, works out which names hold an `asyncio.Queue`
  -- because `queue.get()` and `queue.put()` are outbound-SHAPED and have to be told
  apart from a request by something better than a hand-written list.
- A site is `<session>.<verb>(...)` for the HTTP verbs.
- Anything else that is outbound-shaped -- consumed by the async machinery of its own
  function, and named for an HTTP verb -- whose receiver is NEITHER a derived session NOR
  a derived queue FAILS, unless it is listed in `NOT_A_SESSION` with a reason. The census
  used to `continue` past any receiver it could not reduce to a dotted name, which
  silently dropped `self._session().get(url)`, `clients[0].get(url)` and every other
  shape with a call or a subscript in it -- including the idiom `update_service.py`
  itself uses to reach its session. Failing closed is what makes "the census is complete"
  checkable rather than assumed.
- Receivers that merely read `session`-ish and are not outbound-shaped
  (`SessionLogger.session_id.get()`, a dict `payload.get(k)`, a SQLAlchemy
  `session.query(...).delete()`) are neither awaited nor entered, so they do not appear.

Note that the RATCHET does not use the outbound-shaped filter at all: `session.get(url)`
on a derived session name is a fetch site however its result is consumed. The filter only
decides what goes in the fail-closed bucket, whose whole population is receivers the
derivation could not type.

What it CANNOT see, stated rather than papered over.

- Whether the URL that reaches a given call is externally supplied, and whether a
  function that vets one address then fetches a different one. Both need data flow.
- A synchronous client (`requests.get(url)`), which is not awaited and is not a
  dependency of this package.
- The difference between a `@property` and its `@x.setter`: one name in one class is one
  key, so a getter that vets covers a setter that does not. Three such pairs exist on
  this tree and none of them is a fetch site. The check below that no two definitions
  share a key fails the moment a collision appears that is not one of those pairs.
- Consumption through a receiver that is NOT an imported module:
  `t = create_task(client.get(u))` and `await ensure_future(client.get(u))` called bare,
  `tg.create_task(client.get(u))` on a task group, and
  `await stack.enter_async_context(client.get(u))` on an exit stack; plus
  `functools.partial(client.get, u)` and `getattr(client, 'get')(u)`, where there is no
  HTTP-verb call node to find at all, and a coroutine built in a nested `def` and awaited
  by its caller. Argument positions ARE consumed when the callee is `<module>.<attr>` for
  a module imported in that file, which is what covers `asyncio.gather(a.get(x),
  a.get(y))`, `asyncio.gather(*[a.get(x), a.get(y)])`, `asyncio.wait({a.get(x)})`,
  `asyncio.wait_for(...)` and `asyncio.shield(...)` written with explicit
  arguments rather than a comprehension. Measured, not guessed: restricting it to module
  receivers leaves the fail-closed bucket at 8 entries, where marking every call in every
  argument position puts 24 more `dict.get` receivers in it, each needing a hand-written
  reason nobody would re-read. On a receiver the derivation CAN type, all of these are
  fetch sites already, because the ratchet ignores consumption entirely.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.package_sources import REPO_ROOT, parsed_sources

SCAN_ROOTS = ("open_webui_openrouter_pipe", "filters")

SESSION_TYPES = frozenset({"ClientSession", "AsyncClient"})

QUEUE_TYPES = frozenset({"Queue", "LifoQueue", "PriorityQueue"})

HTTP_VERBS = frozenset(
    {"get", "post", "put", "patch", "delete", "head", "options", "request", "stream",
     "send", "ws_connect"}
)

VETTING_HELPERS = frozenset(
    {
        "_is_safe_url",
        "_prepare_pinned_request",
        "_request_ips_blocking",
        "_resolve_validated_ips",
        "_validated_ips_for_host",
        "_hop_is_refused",
        "_fetchable",
        "_vet_payload_addresses",
        "_vetted_reference_urls",
        "_validate_passthrough_urls",
    }
)

# Reviewed, with the reason each one is not an address check. Every entry was read at the
# call site: the claim is that the URL reaching that request cannot be chosen by a remote
# party, so there is no address to vet. Keyed by class and function rather than by line,
# so moving code does not churn the table, and renaming either one is a deliberate
# re-review.
EXEMPT: dict[str, str] = {
    "open_webui_openrouter_pipe/api/gateway/chat_completions_adapter.py::ChatCompletionsAdapter::send_openai_chat_completions_streaming_request":
        "OPENROUTER_API_BASE_URL from valves plus the literal '/chat/completions'",
    "open_webui_openrouter_pipe/api/gateway/chat_completions_adapter.py::ChatCompletionsAdapter::send_openai_chat_completions_nonstreaming_request":
        "OPENROUTER_API_BASE_URL from valves plus the literal '/chat/completions'",
    "open_webui_openrouter_pipe/api/gateway/responses_adapter.py::ResponsesAdapter::send_openai_responses_streaming_request::_producer":
        "OPENROUTER_API_BASE_URL from valves plus the literal '/responses'",
    "open_webui_openrouter_pipe/api/gateway/responses_adapter.py::ResponsesAdapter::send_openai_responses_nonstreaming_request":
        "OPENROUTER_API_BASE_URL from valves plus the literal '/responses'",
    "open_webui_openrouter_pipe/integrations/image_client.py::OpenRouterImageClient::list_models":
        "self._base_url plus the literal '/models?output_modalities=image'",
    "open_webui_openrouter_pipe/integrations/image_client.py::OpenRouterImageClient::endpoints":
        "self._base_url plus '/images/models/<id>/endpoints'; the model id lands in the path",
    "open_webui_openrouter_pipe/integrations/image_client.py::OpenRouterImageClient::generate":
        "self._base_url plus the literal '/images'",
    "open_webui_openrouter_pipe/integrations/video_client.py::OpenRouterVideoClient::list_models":
        "self._base_url plus the literal '/videos/models'",
    "open_webui_openrouter_pipe/integrations/video_client.py::OpenRouterVideoClient::model_modalities":
        "self._base_url plus '/models/<slug>/endpoints'; the slug lands in the path",
    "open_webui_openrouter_pipe/integrations/video_client.py::OpenRouterVideoClient::submit":
        "self._base_url plus the literal '/videos'",
    "open_webui_openrouter_pipe/integrations/video_client.py::OpenRouterVideoClient::status":
        "VideoClient.poll_url refuses any candidate that is not self._base_url or a path "
        "under its origin, and falls back to '<base>/videos/<job_id>'",
    "open_webui_openrouter_pipe/integrations/media_relay.py::relay_to_public_url":
        "_ENDPOINTS is a two-entry literal table of file hosts; 'host' selects a row and "
        "an unknown host raises before any request",
    "open_webui_openrouter_pipe/models/catalog_manager.py::ModelCatalogManager::_fetch_frontend_model_catalog":
        "the module constant _OPENROUTER_FRONTEND_MODELS_URL, whole",
    "open_webui_openrouter_pipe/models/catalog_manager.py::ModelCatalogManager::_fetch_model_endpoints":
        "_OPENROUTER_MODEL_ENDPOINTS_URL_TEMPLATE; the slug lands in the path",
    "open_webui_openrouter_pipe/models/registry.py::OpenRouterModelRegistry::_refresh":
        "OPENROUTER_API_BASE_URL from valves plus the literal '/models'",
    "open_webui_openrouter_pipe/models/registry.py::OpenRouterModelRegistry::_fetch_zdr_model_ids":
        "OPENROUTER_API_BASE_URL from valves plus the literal '/endpoints/zdr'",
    "open_webui_openrouter_pipe/pipe.py::Pipe::_ping_openrouter":
        "OPENROUTER_API_BASE_URL from valves plus the literal '/models?limit=1'",
}

# Helpers that take the URL as a parameter, so nothing inside them can vet it, and whose
# exemption rests entirely on there being exactly one caller. That claim is checked, not
# read: a second caller appearing is how this kind of reason rots.
SOLE_CALLER: dict[str, str] = {}

# Outbound-SHAPED calls whose receiver is neither a session nor a queue this file can
# derive. Each was read at the call site. Keyed by receiver AND enclosing function, so a
# second one at a new site is a new decision rather than something an old entry licenses.
NOT_A_SESSION: dict[str, str] = {
    "open_webui_openrouter_pipe/pipe.py::Pipe::_shutdown_tool_context::_graceful::context.queue.put":
        "ToolContext.queue is the asyncio.Queue the tool workers drain; the sentinel None "
        "is pushed once per worker and `context.queue.join()` follows",
    "open_webui_openrouter_pipe/tools/tool_executor.py::ToolExecutor::_execute_function_calls::context.queue.put":
        "the same ToolContext.queue, carrying a queued tool call to a worker",
    "open_webui_openrouter_pipe/plugins/pipe_dashboard/dashboard_publisher.py::run_dashboard_publisher::client.delete":
        "the redis client from get_redis(); `worker_key` is this worker's own presence "
        "key and `delete` here is DEL, not an HTTP verb",
    "open_webui_openrouter_pipe/tools/tool_executor.py::ToolExecutor::_tool_worker_loop::context.queue.get":
        "the same ToolContext.queue as the `put` entries above, read by a worker; bound "
        "to `get_coro` first so it can be awaited bare or under `asyncio.wait_for`",
    "open_webui_openrouter_pipe/integrations/video.py::VideoGenerationAdapter::_acquire_message_lock::self._pipe._video_message_locks.get":
        "a dict of asyncio.Lock keyed by (chat, message); this is dict.get returning the "
        "lock for a key, taken while holding _video_message_locks_dict_lock",
    "open_webui_openrouter_pipe/integrations/video.py::VideoGenerationAdapter::_try_acquire_user_slot::self._pipe._video_user_locks.get":
        "a dict of asyncio.Lock keyed by user id; dict.get, then `async with lock` guards "
        "the per-user concurrency count",
    "open_webui_openrouter_pipe/integrations/video.py::VideoGenerationAdapter::_add_user_active_job::self._pipe._video_user_locks.get":
        "the same per-user asyncio.Lock dict; dict.get, then `async with lock` guards the "
        "active-job set",
    "open_webui_openrouter_pipe/integrations/video.py::VideoGenerationAdapter::_release_user_slot::self._pipe._video_user_locks.get":
        "the same per-user asyncio.Lock dict; dict.get, then `async with lock` releases "
        "the slot",
}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _expr_key(node: ast.AST) -> str | None:
    """A dotted name for Name/Attribute chains, and nothing for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        try:
            return ast.unparse(node)
        except Exception:
            return None
    return None


def _annotation_names(node: ast.AST) -> set[str]:
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            try:
                found |= _annotation_names(ast.parse(sub.value, mode="eval").body)
            except SyntaxError:
                continue
    return found


def _is_annotation_of(annotation: ast.AST | None, types: frozenset[str]) -> bool:
    return annotation is not None and bool(_annotation_names(annotation) & types)


def _is_constructor_of(node: ast.AST, types: frozenset[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name in types


def derived_names(tree: ast.Module, types: frozenset[str]) -> set[str]:
    """Names in this module holding one of `types`, by annotation or by assignment."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and _is_annotation_of(node.annotation, types):
            names.add(node.arg)
        elif isinstance(node, ast.AnnAssign) and _is_annotation_of(node.annotation, types):
            key = _expr_key(node.target)
            if key:
                names.add(key)

    for _ in range(len(names) + 8):
        before = len(names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = node.value
                if _is_constructor_of(value, types) or _expr_key(value) in names:
                    for target in node.targets:
                        key = _expr_key(target)
                        if key:
                            names.add(key)
            elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                ctx = node.context_expr
                if _is_constructor_of(ctx, types) or _expr_key(ctx) in names:
                    key = _expr_key(node.optional_vars)
                    if key:
                        names.add(key)
        if len(names) == before:
            break
    return names


def session_names(tree: ast.Module) -> set[str]:
    return derived_names(tree, SESSION_TYPES)


def _imported_modules(tree: ast.Module) -> set[str]:
    """Names bound by `import x` / `import x.y as z` in this file.

    A module receiver is what tells `asyncio.gather(a.get(u))` apart from
    `store(body.get(k))`: the first fans out awaitables, the second is a dict lookup
    handed to a local. `from x import y` is deliberately NOT here -- `y` is usually a
    function, and treating it as a module reopens the 24 `dict.get` receivers that
    marking every argument position produced.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _function_index(tree: ast.Module) -> dict[ast.AST, list[str]]:
    """Every node mapped to the chain of CLASS and function names enclosing it.

    The class is in the chain because without it two same-named methods in two classes
    in one module are one key, and a key is what an EXEMPT reason and a `vets` verdict
    both hang off. Measured on this tree before the class was added: seven names
    collided, `storage/multimodal.py::__init__` three ways -- and appending a class with
    an unvetted POST to a caller-supplied URL next to an exempt sibling left the whole
    file green, with the new site reported as EXEMPT rather than as missed.

    What remains keyed together, stated rather than papered over: a `@property` and its
    `@x.setter` share a name inside one class, so they share a chain. `_vets` unions the
    chains it finds, so a getter that vets covers a setter that does not. Neither half of
    such a pair is a fetch site anywhere in this tree, and telling them apart needs the
    decorator, which is a different key rather than a longer one.
    """
    index: dict[ast.AST, list[str]] = {}

    def walk(node: ast.AST, chain: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = [*chain, child.name]
                index[child] = inner
                walk(child, inner)
            else:
                index[child] = chain
                walk(child, chain)

    walk(tree, [])
    return index


def _outbound_shaped(tree: ast.Module) -> set[int]:
    """Calls the async machinery of their own function consumes.

    Every HTTP request in this package is awaited or entered with `async with`, because
    both aiohttp and httpx require it. A `dict.get(...)` never is, which is what keeps
    the fail-closed bucket down to a handful of entries instead of the 2244 that
    dropping this filter altogether produces on this tree.

    Five consumption SHAPES count, and none of them names a library function:

    - the call is awaited, entered, or iterated with `async for`, as a statement or as
      the `async for` clause of a comprehension -- the direct case;
    - the call is the element of a comprehension or generator expression inside one of
      those, including the key or the value of a dict comprehension.
      `await asyncio.gather(*(recv.get(u) for u in urls))` is the fan-out idiom
      `models/catalog_manager.py` and `integrations/video_catalog.py` already write, and
      it was invisible: the census saw only `asyncio.gather`, whose receiver is a module;
    - the call sits in an argument position of a consumed call whose callee is
      `<module>.<attr>` for a module imported in that file, `Starred` unwrapped and
      tuple/list/set literals descended into. That is the same fan-out written out:
      `asyncio.gather(recv.get(a), recv.get(b))`, `asyncio.gather(*[recv.get(a),
      recv.get(b)])`, `asyncio.wait({recv.get(a)})`, `asyncio.wait_for(recv.get(u), 1)`,
      `asyncio.shield(recv.get(u))`. Restricted to module receivers because
      `await store(body.get(k))` must stay out;
    - the call is bound to a local that is later consumed one of those ways, through
      `=`, an annotated `=`, a walrus, or one element of a tuple target;
    - all of the above, applied again to whatever those bindings turn up.

    Naming `gather`, `create_task` and `ensure_future` instead would be a list of
    library entry points, and the next one written -- `TaskGroup.create_task`, or a
    project helper taking a coroutine -- would not be on it. Shapes are enumerable;
    consumers are not. The residual escapes are recorded in the module docstring
    rather than papered over, because closing them costs the bucket 24 more `dict.get`
    entries and buys nothing that is not already covered by the receiver derivation.
    """
    shaped: set[int] = set()
    entered: set[str] = set()
    modules = _imported_modules(tree)

    def _elements_of_comprehensions(root: ast.AST) -> None:
        for sub in ast.walk(root):
            if isinstance(
                sub, (ast.GeneratorExp, ast.ListComp, ast.SetComp)
            ) and isinstance(sub.elt, ast.Call):
                shaped.add(id(sub.elt))
            elif isinstance(sub, ast.DictComp):
                for part in (sub.key, sub.value):
                    if isinstance(part, ast.Call):
                        shaped.add(id(part))

    def _arguments_of(call: ast.Call) -> None:
        func = call.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in modules
        ):
            return
        for arg in [*call.args, *(kw.value for kw in call.keywords)]:
            _mark_argument(arg)

    def _mark_argument(node: ast.AST) -> None:
        if isinstance(node, ast.Starred):
            _mark_argument(node.value)
        elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for element in node.elts:
                _mark_argument(element)
        elif isinstance(node, ast.Call):
            shaped.add(id(node))
            _arguments_of(node)

    def _consume(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            shaped.add(id(node))
            _arguments_of(node)
        else:
            key = _expr_key(node)
            if key:
                entered.add(key)
        _elements_of_comprehensions(node)

    for node in ast.walk(tree):
        if isinstance(node, ast.Await):
            _consume(node.value)
        elif isinstance(node, ast.withitem):
            _consume(node.context_expr)
        elif isinstance(node, ast.AsyncFor):
            _consume(node.iter)
        elif isinstance(node, ast.comprehension) and node.is_async:
            _consume(node.iter)

    for _ in range(len(entered) + 8):
        before = (len(shaped), len(entered))
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            elif isinstance(node, ast.NamedExpr):
                targets, value = [node.target], node.value
            if value is None:
                continue
            flat: list[ast.AST] = []
            for target in targets:
                flat.extend(
                    target.elts
                    if isinstance(target, (ast.Tuple, ast.List))
                    else [target]
                )
            names = {k for k in (_expr_key(t) for t in flat) if k}
            if not names & entered:
                continue
            bound = (
                value.elts
                if isinstance(value, (ast.Tuple, ast.List)) and len(flat) > 1
                else [value]
            )
            for item in bound:
                _consume(item)
        if (len(shaped), len(entered)) == before:
            break
    return shaped


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _leaves(node: ast.AST) -> bool:
    return any(
        isinstance(sub, (ast.Return, ast.Raise, ast.Continue, ast.Break))
        for sub in ast.walk(node)
    )


def _decisive_expressions(func: ast.AST) -> list[ast.AST]:
    """Expressions whose value decides whether the function carries on.

    A verdict that reaches one of these gates something. A verdict that reaches none of
    them was computed and thrown away -- which is exactly what `await self._is_safe_url(url)`
    with the result discarded does, and what the previous name-appearance rule could not
    tell apart from a real check.
    """
    decisive: list[ast.AST] = []
    for node in ast.walk(func):
        if isinstance(node, (ast.Return, ast.Raise)):
            decisive.append(node)
        elif isinstance(node, (ast.If, ast.While)):
            if any(_leaves(stmt) for stmt in [*node.body, *node.orelse]):
                decisive.append(node.test)
        elif isinstance(node, (ast.Assert, ast.IfExp)):
            decisive.append(node.test)
    return decisive


def _vets(func: ast.AST) -> bool:
    decisive = _decisive_expressions(func)
    decisive_ids = {id(sub) for region in decisive for sub in ast.walk(region)}
    decisive_names = {
        sub.id
        for region in decisive
        for sub in ast.walk(region)
        if isinstance(sub, ast.Name)
    }

    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or _call_name(node) not in VETTING_HELPERS:
            continue
        if id(node) in decisive_ids:
            return True

    for node in ast.walk(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            carries = any(
                isinstance(sub, ast.Call) and _call_name(sub) in VETTING_HELPERS
                for sub in ast.walk(node.value)
            )
            if not carries:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if {t.id for t in targets if isinstance(t, ast.Name)} & decisive_names:
                return True
    return False


def _vetting_chains(tree: ast.Module, index: dict[ast.AST, list[str]]) -> set[tuple[str, ...]]:
    return {
        tuple(chain)
        for node, chain in index.items()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _vets(node)
    }


def vetting_functions(files: tuple[tuple[Path, str, ast.Module], ...]) -> set[str]:
    """Qualified names of every function whose address-check verdict gates something."""
    found: set[str] = set()
    for path, _source, tree in files:
        index = _function_index(tree)
        for chain in _vetting_chains(tree, index):
            found.add("::".join([_rel(path), *chain]))
    return found


def outbound_fetch_sites(
    files: tuple[tuple[Path, str, ast.Module], ...],
) -> list[tuple[str, int, bool, str]]:
    """(key, line, vets, receiver) for every HTTP call on a derived session name.

    Callable on synthetic sources, which is what lets a test prove the matcher still
    matches. A scan of the real tree can only ever show the population is clean, and a
    matcher narrowed to nothing shows exactly the same thing.
    """
    sites: list[tuple[str, int, bool, str]] = []
    for path, _source, tree in files:
        names = session_names(tree)
        if not names:
            continue
        index = _function_index(tree)
        vetting_chains = _vetting_chains(tree, index)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in HTTP_VERBS:
                continue
            key = _expr_key(node.func.value)
            if key is None or key not in names:
                continue
            chain = index.get(node, [])
            vets = any(
                tuple(chain[: length + 1]) in vetting_chains for length in range(len(chain))
            )
            qualified = "::".join([_rel(path), *chain]) if chain else _rel(path)
            sites.append((qualified, node.lineno, vets, f"{key}.{node.func.attr}"))
    return sites


def unclassified_outbound_calls(
    files: tuple[tuple[Path, str, ast.Module], ...],
) -> list[tuple[str, int, str]]:
    """(key, line, receiver.verb) for outbound-shaped calls on an unclassified receiver.

    These are what the census cannot reduce to a session or a queue. `self._session()
    .get(url)` lands here, and so does `clients[0].post(url)`. The population on the real
    tree has to be empty after NOT_A_SESSION is subtracted.
    """
    found: list[tuple[str, int, str]] = []
    for path, _source, tree in files:
        sessions = session_names(tree)
        queues = derived_names(tree, QUEUE_TYPES)
        shaped = _outbound_shaped(tree)
        index = _function_index(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in HTTP_VERBS or id(node) not in shaped:
                continue
            key = _expr_key(node.func.value)
            if key is not None and (key in sessions or key in queues):
                continue
            try:
                receiver = ast.unparse(node.func.value)
            except Exception:  # pragma: no cover - unparse is total on parsed trees
                receiver = "<unparseable>"
            chain = index.get(node, [])
            qualified = "::".join([_rel(path), *chain]) if chain else _rel(path)
            found.append((qualified, node.lineno, f"{receiver}.{node.func.attr}"))
    return found


def _real_files() -> tuple[tuple[Path, str, ast.Module], ...]:
    files: tuple[tuple[Path, str, ast.Module], ...] = ()
    for root in SCAN_ROOTS:
        files = files + parsed_sources(root)
    return files


def _real_sites() -> list[tuple[str, int, bool, str]]:
    return outbound_fetch_sites(_real_files())


def _synthetic(source: str, name: str = "scratch.py") -> tuple[tuple[Path, str, ast.Module], ...]:
    return ((REPO_ROOT / "open_webui_openrouter_pipe" / name, source, ast.parse(source)),)


def _callers_of(site: str) -> set[str]:
    """Qualified names of every function calling the bare function named by `site`.

    Across BOTH scan roots, not just the module the helper is defined in. A private
    helper reached by `from .actions import dispatch_action` and then called bare is an
    idiom this package already uses -- `plugins/pipe_dashboard/http_routes.py` does it --
    so a check that parses one module would report "one caller" for a helper with two,
    which is the precise claim a delegated exemption rests on.
    """
    _module_rel, function = site.rsplit("::", 1)
    callers: set[str] = set()
    for path, _source, tree in _real_files():
        index = _function_index(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == function:
                    callers.add("::".join([_rel(path), *index.get(node, [])]))
    return callers


def test_every_outbound_fetch_is_vetted_or_reviewed():
    """The ratchet. A new fetch site with no address check fails here and nowhere else."""
    offenders = [
        (key, line, receiver)
        for key, line, vets, receiver in _real_sites()
        if not vets and key not in EXEMPT
    ]

    assert not offenders, (
        "these outbound requests neither vet their address nor carry a reviewed "
        "exemption:\n"
        + "\n".join(f"  {key}  line {line}  ({receiver})" for key, line, receiver in offenders)
        + "\n\nIf the URL can be chosen by a remote party, go through "
        "MultimodalHandler._vetted_get, which vets and pins every hop. If it cannot, add "
        "the site to EXEMPT in this file with the reason, which is a line someone has to "
        "write and a reviewer gets to disagree with."
    )


def test_every_outbound_shaped_call_is_a_session_or_declared_not_one():
    """Fails closed. A receiver the census cannot classify is a hole, not a pass.

    The previous version skipped any receiver that was not a plain dotted name, so
    `async with self._session().get(url)` -- the idiom `update_service.py` uses to reach
    its own session -- was invisible, along with every subscripted, parenthesised or
    call-returning receiver.
    """
    unclassified = [
        (key, line, receiver)
        for key, line, receiver in unclassified_outbound_calls(_real_files())
        if f"{key}::{receiver}" not in NOT_A_SESSION
    ]

    assert not unclassified, (
        "these calls are shaped like an outbound request and the census cannot tell "
        "what they are called on:\n"
        + "\n".join(f"  {key}  line {line}  ({receiver})" for key, line, receiver in unclassified)
        + "\n\nIf it is an HTTP session, bind it to a name the derivation can see. If it "
        "is not, add it to NOT_A_SESSION with the reason."
    )


def test_no_not_a_session_entry_is_stale():
    present = {
        f"{key}::{receiver}"
        for key, _line, receiver in unclassified_outbound_calls(_real_files())
    }
    stale = sorted(set(NOT_A_SESSION) - present)
    assert not stale, (
        f"these entries no longer name an unclassified call: {stale}. Delete them; a "
        "stale entry silently covers whatever is written at that name next."
    )


VETTING_FETCH_SITES = (
    "open_webui_openrouter_pipe/storage/multimodal.py::MultimodalHandler::_vetted_get",
    "open_webui_openrouter_pipe/storage/multimodal.py::MultimodalHandler::_download_remote_url",
    "open_webui_openrouter_pipe/storage/multimodal.py::MultimodalHandler::_download_remote_url_streaming",
)


@pytest.mark.parametrize("expected", VETTING_FETCH_SITES)
def test_the_census_still_finds_the_sites_that_vet_their_own_address(expected):
    """The three the EXEMPT table cannot speak for, because they carry no exemption.

    A matcher that stopped matching would pass the ratchet with nothing to report. It is
    caught for the other seventeen sites by `test_no_exemption_is_stale` below, which
    names each one -- these three vet, so they are absent from that table and need saying
    here. That leaves no numeric floor anywhere in this file.

    `assert len(sites) >= 20` against a tree holding exactly 20 had no slack in the one
    direction this codebase is meant to move: folding the two identical request blocks in
    `integrations/image_client.py` into one helper takes the count to 19 and reddens a
    check whose message blames the matcher. The same refactor still reddens the EXEMPT
    table, which is correct and says so -- the site moved to a function no reviewed reason
    covers, and re-pointing the reason is the review.
    """
    keys = {key for key, _line, _vets, _receiver in _real_sites()}

    assert expected in keys, f"{expected} is a known fetch site and was not found"


def test_the_one_transport_that_can_be_pointed_anywhere_is_a_vetting_site():
    """The defect this file was written for, stated as a property of the tree.

    The model-icon and release-asset downloads reach the network through this one
    function, so it is the only place a catalog- or release-supplied address is dialled.
    """
    vetted = {key for key, _line, vets, _receiver in _real_sites() if vets}
    assert "open_webui_openrouter_pipe/storage/multimodal.py::MultimodalHandler::_vetted_get" in vetted


@pytest.mark.parametrize("entry", sorted(set(EXEMPT) | set(SOLE_CALLER)))
def test_no_exemption_is_stale(entry):
    """A removed fetch site must not leave a licence behind for the next one.

    Also the "the matcher still matches" check for every site that carries a reason:
    a derivation that collapsed to nothing fails all seventeen of these rows, each
    naming the site it lost. Parametrised so it names them one at a time rather than
    printing a list of everything at once.
    """
    keys = {key for key, _line, _vets, _receiver in _real_sites()}
    assert entry in keys, (
        f"{entry} no longer names a fetch site. Delete the entry, or re-point it at the "
        "function the request moved to; a stale exemption silently covers whatever is "
        "written at that name next."
    )


def test_no_exemption_is_also_vetted():
    """An exemption over a site that vets is a reason nobody will re-read."""
    redundant = sorted(
        key for key, _line, vets, _receiver in _real_sites() if vets and key in EXEMPT
    )
    assert not redundant, f"these vet their address and do not need an exemption: {redundant}"


def test_every_exemption_carries_a_reason():
    thin = sorted(key for key, reason in EXEMPT.items() if len(reason.strip()) < 20)
    assert not thin, f"these exemptions say nothing: {thin}"


def test_an_exemption_that_rests_on_its_one_caller_has_exactly_that_caller():
    """The delegated reason, checked instead of believed.

    A helper that takes the URL as a parameter cannot vet anything itself, so such an
    exemption is entirely a claim about who calls it, and a second caller appearing is
    how that kind of reason rots. Iterated rather than parametrised: the table is empty
    today, and an empty parametrisation reports as a SKIP, which reads like a check that
    could not run rather than a table with nothing in it.
    """
    for site, caller in sorted(SOLE_CALLER.items()):
        assert site in EXEMPT, f"{site} claims a sole caller but carries no reason"
        assert _callers_of(site) == {caller}, (
            f"{site.rsplit('::', 1)[1]} is called from {sorted(_callers_of(site))}; the "
            f"exemption only covers {caller}. Vet the URL in the new caller too, or move "
            "the check into the transport."
        )


def test_the_named_sole_caller_is_a_real_function():
    """A typo in the table would make the check above vacuous in one direction."""
    for _site, caller in sorted(SOLE_CALLER.items()):
        module_rel, *_chain = caller.split("::")
        tree = ast.parse((REPO_ROOT / module_rel).read_text(encoding="utf-8"))
        index = _function_index(tree)
        defined = {
            "::".join([module_rel, *names])
            for node, names in index.items()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert caller in defined, f"{caller} is not a function in {module_rel}"


def test_the_sole_caller_check_sees_callers_in_other_modules():
    """The table is empty, so the machinery it rests on needs its own evidence.

    `_audit` is a PRIVATE helper defined in `plugins/pipe_dashboard/actions.py`, imported
    by name into `http_routes.py` and called bare there. A check that parsed only the
    defining module would miss that caller entirely and report a sole caller for a helper
    that has more than one -- silently licensing exactly the drift the table exists to
    catch. Chosen over a grep for the name, which also hits the import line and the
    tables in this file.
    """
    callers = _callers_of(
        "open_webui_openrouter_pipe/plugins/pipe_dashboard/actions.py::_audit"
    )
    modules = {key.split("::", 1)[0] for key in callers}

    assert "open_webui_openrouter_pipe/plugins/pipe_dashboard/http_routes.py" in modules, (
        f"only found callers in {sorted(modules)}; a cross-module caller of a bare "
        "helper is invisible, which is the one thing SOLE_CALLER claims to check"
    )
    assert len(modules) > 1, sorted(modules)


# ── the matcher, driven against sources that ARE offenders ───────────────────


def test_the_matcher_reports_an_unvetted_session_get():
    """Driven against source that IS an offender, so 'no offenders' means something."""
    source = (
        "import aiohttp\n"
        "async def grab(session: aiohttp.ClientSession, url: str):\n"
        "    async with session.get(url) as resp:\n"
        "        return await resp.read()\n"
    )
    sites = outbound_fetch_sites(_synthetic(source))
    assert [(key.rsplit('::', 1)[1], vets) for key, _line, vets, _r in sites] == [("grab", False)]


def test_the_matcher_reports_a_vetted_session_get_as_vetted():
    source = (
        "import aiohttp\n"
        "class H:\n"
        "    async def grab(self, session: aiohttp.ClientSession, url: str):\n"
        "        if not await self._is_safe_url(url):\n"
        "            return None\n"
        "        async with session.get(url) as resp:\n"
        "            return await resp.read()\n"
    )
    sites = outbound_fetch_sites(_synthetic(source))
    assert [vets for _key, _line, vets, _r in sites] == [True]


@pytest.mark.parametrize(
    ("body", "vets"),
    [
        ("        await self._is_safe_url(url)\n", False),
        ("        self._is_safe_url(url)\n", False),
        ("        verdict = await self._is_safe_url(url)\n", False),
        ("        if not await self._is_safe_url(url):\n            return None\n", True),
        (
            "        verdict = await self._is_safe_url(url)\n"
            "        if not verdict:\n"
            "            return None\n",
            True,
        ),
        (
            "        verdict = await self._is_safe_url(url)\n"
            "        seen[url] = verdict\n"
            "        return verdict\n",
            True,
        ),
        (
            "        safe = await self._is_safe_url(url)\n"
            "        if not safe:\n"
            "            raise RuntimeError(url)\n",
            True,
        ),
    ],
    ids=[
        "awaited-and-discarded",
        "called-and-discarded",
        "bound-and-ignored",
        "gates-a-return",
        "bound-then-gates-a-return",
        "returned-as-the-verdict",
        "gates-a-raise",
    ],
)
def test_a_verdict_only_counts_when_it_decides_something(body, vets):
    """Calling the check is not checking.

    The rule used to be that the helper's NAME appeared anywhere in the function body,
    so replacing the icon guard with a bare `await self._is_safe_url(url)` -- verdict
    discarded, request issued regardless -- left the whole file green.
    """
    source = (
        "import aiohttp\n"
        "class H:\n"
        "    async def grab(self, session: aiohttp.ClientSession, url: str):\n"
        f"{body}"
        "        async with session.get(url) as resp:\n"
        "            return await resp.read()\n"
    )
    sites = outbound_fetch_sites(_synthetic(source))
    assert [v for _key, _line, v, _r in sites] == [vets]


def test_the_matcher_follows_a_session_stored_on_self():
    source = (
        "import aiohttp\n"
        "class C:\n"
        "    def __init__(self, session: aiohttp.ClientSession):\n"
        "        self._session = session\n"
        "    async def go(self, url):\n"
        "        async with self._session.post(url) as resp:\n"
        "            return resp\n"
    )
    sites = outbound_fetch_sites(_synthetic(source))
    assert [(key.rsplit('::', 1)[1], receiver) for key, _l, _v, receiver in sites] == [
        ("go", "self._session.post")
    ]


def test_two_same_named_methods_in_two_classes_are_two_sites():
    """The class is part of the key, so a sibling's exemption cannot cover a new method.

    Before it was, both of these were `scratch.py::grab` -- one key, one EXEMPT reason
    and one `vets` verdict between them. `_vetting_chains` is a SET of chains, so the
    vetted one put `grab` in it and the unvetted one read as vetted; adding a class with
    an unvetted POST to a caller-supplied URL next to an exempt sibling on the real tree
    left every check in this file green.

    Both halves are asserted. Distinct keys alone would still pass if the verdict were
    shared, and the shared verdict is the half that hides a real request.
    """
    source = (
        "import aiohttp\n"
        "class Vetted:\n"
        "    async def grab(self, session: aiohttp.ClientSession, url: str):\n"
        "        if not await self._is_safe_url(url):\n"
        "            return None\n"
        "        async with session.get(url) as resp:\n"
        "            return await resp.read()\n"
        "class Unvetted:\n"
        "    async def grab(self, session: aiohttp.ClientSession, url: str):\n"
        "        async with session.post(url) as resp:\n"
        "            return await resp.read()\n"
    )

    sites = outbound_fetch_sites(_synthetic(source))

    assert sorted((key.split("::", 1)[1], vets) for key, _l, vets, _r in sites) == [
        ("Unvetted::grab", False),
        ("Vetted::grab", True),
    ], sites


def test_the_real_tree_has_no_two_definitions_under_one_key():
    """The same property where it actually bit: `storage/multimodal.py` has three.

    `UnfetchableAddress.__init__`, `_VettedResolver.__init__` and
    `MultimodalHandler.__init__` were one key, and so were six other names across the
    package. A synthetic pair proves the derivation distinguishes classes; this proves
    the tree it is pointed at no longer collapses -- which is what EXEMPT and
    VETTING_FETCH_SITES are indexed by.
    """
    offenders: dict[str, list[int]] = {}
    for path, _source, tree in _real_files():
        seen: dict[str, list[int]] = {}
        for node, chain in _function_index(tree).items():
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                seen.setdefault("::".join([_rel(path), *chain]), []).append(node.lineno)
        for key, lines in seen.items():
            if len(lines) > 1 and not _is_a_property_pair(tree, lines):
                offenders[key] = sorted(lines)

    assert not offenders, (
        "these keys name more than one function definition, so an EXEMPT reason or a "
        f"vetting verdict written for one silently covers the other: {offenders}"
    )


def _is_a_property_pair(tree: ast.Module, lines: list[int]) -> bool:
    """A `@property` and its `@x.setter`: one name, one class, two definitions.

    Recorded as the residual in `_function_index`'s docstring rather than fixed, because
    telling them apart needs the decorator in the key. Neither half is a fetch site
    anywhere in this tree; this predicate is what keeps that claim honest by failing the
    check above the moment a collision appears that is NOT one of these.
    """
    decorated = {
        node.lineno: [ast.unparse(d) for d in node.decorator_list]
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    marks = [decorated.get(line, []) for line in lines]
    return all(marks) and any("property" in "".join(m) for m in marks)


def test_a_websocket_upgrade_is_a_fetch_site_like_every_other_verb():
    """`session.ws_connect(url)` dials whatever URL it is handed, like `get` does.

    It was in neither bucket: not an HTTP verb, so not a site, and not outbound-shaped
    for the fail-closed check either. The package writes none today, which is exactly
    when adding it is free.
    """
    source = (
        "import aiohttp\n"
        "async def stream(session: aiohttp.ClientSession, url: str):\n"
        "    async with session.ws_connect(url) as ws:\n"
        "        return await ws.receive()\n"
    )

    sites = outbound_fetch_sites(_synthetic(source))

    assert [(receiver, vets) for _k, _l, vets, receiver in sites] == [
        ("session.ws_connect", False)
    ]


def test_the_matcher_follows_an_httpx_client_bound_by_async_with():
    source = (
        "import httpx\n"
        "async def go(url):\n"
        "    async with httpx.AsyncClient() as client:\n"
        "        async with client.stream('GET', url) as resp:\n"
        "            return resp\n"
    )
    sites = outbound_fetch_sites(_synthetic(source))
    assert [receiver for _k, _l, _v, receiver in sites] == ["client.stream"]


@pytest.mark.parametrize(
    "receiver",
    [
        "self._session()",
        "clients[0]",
        "session or fallback",
        "get_session()",
        "self.sessions['primary']",
        "_pool.pop()",
    ],
    ids=["call-on-self", "subscript", "boolean-or", "bare-call", "dict-lookup", "pop"],
)
def test_an_unclassifiable_receiver_fails_closed(receiver):
    """Every shape the old `isinstance(receiver, (Name, Attribute))` guard dropped.

    A panel added `async with self._session().get(url)` to `update_service.py` and the
    whole file still reported 16 passed.
    """
    source = (
        "async def grab(url):\n"
        f"    async with ({receiver}).get(url) as resp:\n"
        "        return await resp.read()\n"
    )
    found = unclassified_outbound_calls(_synthetic(source))
    assert [r for _k, _l, r in found] == [f"{receiver}.get"]


def test_a_derived_session_is_not_in_the_unclassified_bucket():
    """Otherwise every real fetch site would have to be listed as 'not a session'."""
    source = (
        "import aiohttp\n"
        "async def grab(session: aiohttp.ClientSession, url: str):\n"
        "    async with session.get(url) as resp:\n"
        "        return await resp.read()\n"
    )
    assert unclassified_outbound_calls(_synthetic(source)) == []


def test_a_derived_queue_is_not_in_the_unclassified_bucket():
    """`await queue.get()` is outbound-shaped and is not a request."""
    source = (
        "import asyncio\n"
        "async def pump():\n"
        "    queue = asyncio.Queue()\n"
        "    await queue.put(1)\n"
        "    return await queue.get()\n"
    )
    assert unclassified_outbound_calls(_synthetic(source)) == []


@pytest.mark.parametrize(
    "source",
    [
        "class L:\n    session_id = None\nx = L.session_id.get()\n",
        "def f(session):\n    return session.query(int).filter(True).delete()\n",
        "def f(session: dict):\n    return session.get('viewer')\n",
        "async def f(client):\n    await client.delete('worker-key')\n",
    ],
    ids=["contextvar", "sqlalchemy", "dict", "redis"],
)
def test_the_matcher_ignores_lookalikes_that_are_not_http(source):
    """The receivers a grep for `session.get(` reports and this must not."""
    assert outbound_fetch_sites(_synthetic(source)) == []


def test_the_matcher_ignores_examples_written_inside_docstrings():
    """core/timing_logger.py shows `async with session.post(...)` in two docstrings."""
    source = (
        "import aiohttp\n"
        "def mark(session: aiohttp.ClientSession):\n"
        '    """Usage:\n\n'
        "    async with session.post(url) as resp:\n"
        "        pass\n"
        '    """\n'
        "    return None\n"
    )
    assert outbound_fetch_sites(_synthetic(source)) == []


@pytest.mark.parametrize(
    "source",
    [
        "async def grab(url):\n"
        "    ctx = self._session().get(url)\n"
        "    async with ctx as resp:\n"
        "        return await resp.read()\n",
        "async def grab(url):\n"
        "    coro = clients[0].post(url)\n"
        "    return await coro\n",
        "async def grab(url):\n"
        "    first = get_session().get(url)\n"
        "    second = first\n"
        "    async with second as resp:\n"
        "        return resp\n",
    ],
    ids=["bound-then-entered", "bound-then-awaited", "bound-twice"],
)
def test_a_request_bound_to_a_local_first_is_still_outbound_shaped(source):
    """The same request in two statements instead of one.

    `ctx = self._session().get(url)` then `async with ctx` was silent, while the identical
    call written inline tripped the fail-closed bucket -- so the census could be satisfied
    by moving a line. The local-binding propagation this needs is the one `derived_names`
    already does for session names, so the shape was recognised for one purpose and not
    the other.
    """
    found = unclassified_outbound_calls(_synthetic(source))
    assert len(found) == 1, found


@pytest.mark.parametrize(
    "body",
    [
        "    return await asyncio.gather(*(recv.get(u) for u in urls))\n",
        "    return await asyncio.gather(*[recv.get(u) for u in urls])\n",
        "    async with asyncio.TaskGroup():\n"
        "        await asyncio.gather(*(recv.get(u) for u in urls))\n",
        "    async with contextlib.AsyncExitStack() as stack:\n"
        "        await stack.enter_async_context(\n"
        "            asyncio.gather(*(recv.get(u) for u in urls))\n"
        "        )\n",
        "    return [r async for r in recv.get(urls[0])]\n",
        "    if (ctx := recv.get(urls[0])):\n"
        "        async with ctx as resp:\n"
        "            return resp\n",
        "    first, _rest = recv.get(urls[0]), urls[1:]\n"
        "    return await first\n",
    ],
    ids=[
        "gather-over-a-genexp",
        "gather-over-a-listcomp",
        "gather-inside-a-task-group",
        "gather-through-an-exit-stack",
        "async-for",
        "walrus-then-entered",
        "tuple-target-then-awaited",
    ],
)
def test_a_request_fanned_out_rather_than_awaited_inline_is_still_shaped(body):
    """The census fell open on the package's own fan-out idiom.

    `await asyncio.gather(*(x.get(u) for u in urls))` is what `catalog_manager` and
    `video_catalog` already write. The census saw a call whose receiver was the module
    `asyncio`, and the request inside the generator was consumed by nothing it could
    see -- so an HTTP GET on a receiver the derivation cannot type went in no bucket at
    all, and the file reported clean.

    Every row here is a SHAPE, not a library function: a comprehension element inside
    something consumed, an `async for`, a walrus, a tuple target. `asyncio.gather` is
    incidental to all of them, which is the point -- writing the same fan-out through a
    project helper instead has to trip the same wire.
    """
    source = f"async def grab(recv, urls):\n{body}"

    found = unclassified_outbound_calls(_synthetic(source))

    assert [r for _k, _l, r in found] == ["recv.get"], found


@pytest.mark.parametrize(
    "body",
    [
        "    return await asyncio.gather(recv.get(urls[0]))\n",
        "    return await asyncio.wait_for(recv.get(urls[0]), 1)\n",
        "    return await asyncio.shield(recv.get(urls[0]))\n",
        "    return await asyncio.wait_for(fut=recv.get(urls[0]), timeout=1)\n",
        "    return await asyncio.gather(*{u: recv.get(u) for u in urls}.values())\n",
        "    return await asyncio.gather(*{recv.get(u): u for u in urls}.keys())\n",
        "    return await asyncio.gather(asyncio.shield(recv.get(urls[0])))\n",
        "    return await asyncio.gather(*[recv.get(urls[0])])\n",
        "    return await asyncio.gather(*(recv.get(urls[0]),))\n",
        "    return await asyncio.wait({recv.get(urls[0])})\n",
        "    return await asyncio.gather(*[asyncio.shield(recv.get(urls[0]))])\n",
    ],
    ids=[
        "gather-with-an-explicit-argument",
        "wait-for",
        "shield",
        "keyword-argument",
        "dictcomp-value",
        "dictcomp-key",
        "nested-module-calls",
        "starred-list-literal",
        "starred-tuple-literal",
        "set-literal",
        "container-around-a-nested-module-call",
    ],
)
def test_a_fan_out_written_with_explicit_arguments_is_still_shaped(body):
    """The same fan-out, spelled out instead of comprehended.

    `_consume` marked the consumed node itself and `_elements_of_comprehensions` covered
    only Generator/List/Set comps, so an HTTP-verb call in an ARGUMENT position, or in a
    dict comprehension, landed in no bucket at all. Measured on the real tree before this:
    the inline form was caught, and
    `await asyncio.gather(v._raw_session().get(a), v._raw_session().get(b))` in
    `update_service.py` left the file green.

    `import asyncio` is part of every source here, deliberately: the rule is keyed on
    modules imported in the file being scanned, which is what keeps
    `await store(body.get(k))` -- a local, not a module -- out of the bucket.
    """
    source = f"import asyncio\nasync def grab(recv, urls):\n{body}"

    found = unclassified_outbound_calls(_synthetic(source))

    assert [r for _k, _l, r in found] == ["recv.get"], found


@pytest.mark.parametrize(
    "fan_out",
    [
        "asyncio.gather(recv.get(urls[0]), recv.post(urls[1]))",
        "asyncio.gather(*[recv.get(urls[0]), recv.post(urls[1])])",
        "asyncio.gather(*(recv.get(urls[0]), recv.post(urls[1])))",
        "asyncio.gather(*[recv.get(urls[0]), *[recv.post(urls[1])]])",
    ],
    ids=[
        "explicit-arguments",
        "starred-list-literal",
        "starred-tuple-literal",
        "a-starred-literal-inside-a-starred-literal",
    ],
)
def test_every_leg_of_a_multi_argument_fan_out_is_shaped(fan_out):
    """One leg being caught is not the property; every leg is.

    A rule that stopped at the first argument would satisfy the parametrised test above
    on every row.

    The container rows are the fifth member of the fan-out family, and the one that was
    still open: `Starred` was unwrapped, but the unwrapped node was only marked when it
    was itself a call, so the legs of a starred LITERAL were never reached. Measured on
    the real tree with an unvetted two-leg mirror fetch injected into `update_service`:
    written as a starred genexp the census reddened, and the same code as
    `gather(*[a, b])` left all 77 tests green.
    """
    source = (
        "import asyncio\n"
        "async def grab(recv, urls):\n"
        f"    first, second = await {fan_out}\n"
        "    return first, second\n"
    )

    found = unclassified_outbound_calls(_synthetic(source))

    assert sorted(r for _k, _l, r in found) == ["recv.get", "recv.post"], found


def test_a_container_literal_handed_to_a_local_is_not_descended_into():
    """The bound on descending into literals: the module-receiver rule still gates it.

    `await store([recv.get(u)])` is the same list in the same position, handed to a LOCAL
    rather than to an imported module. Dropping that guard is what puts 24 `dict.get`
    receivers in the fail-closed bucket, so this is the row that separates descending
    into a fan-out's legs from descending into everything.

    Written with an unclassifiable receiver rather than a dict lookup deliberately: if
    the guard were dropped, `recv.get` is exactly what would appear.
    """
    source = (
        "async def grab(recv, urls, store):\n"
        "    return await store([recv.get(urls[0])])\n"
    )

    assert unclassified_outbound_calls(_synthetic(source)) == []


def test_an_argument_that_is_not_a_call_does_not_become_an_entered_name():
    """The bound on the argument rule, and the reason it is not `_consume` recursing.

    Marking the NAMES in a module call's argument list as entered leaks module-wide:
    `asyncio.to_thread(fn, x)` would enter `fn`, and every `fn = something.get(k)`
    anywhere in the same file then becomes outbound-shaped. Measured on the real tree,
    that put `item.tool_cfg.get` and `self._pipe._video_active_tasks.get` in the
    fail-closed bucket -- two dict lookups, neither of them a request.
    """
    source = (
        "import asyncio\n"
        "async def grab(cfg, key):\n"
        "    fn = cfg.get('callable')\n"
        "    return await asyncio.to_thread(fn, key)\n"
    )

    assert unclassified_outbound_calls(_synthetic(source)) == []


def test_a_dict_lookup_passed_to_something_awaited_is_not_outbound_shaped():
    """The bound on the rule above, asserted rather than hoped for.

    Marking every call inside an awaited expression would sweep in `await
    store(body.get('x'))` and, measured on this tree, 24 more `dict.get` receivers --
    each needing a hand-written reason in NOT_A_SESSION. A comprehension ELEMENT is a
    narrower shape than "anywhere inside", and this is the case that separates them.
    """
    source = (
        "async def grab(store, body):\n"
        "    return await store(body.get('x'))\n"
    )

    assert unclassified_outbound_calls(_synthetic(source)) == []


def test_a_local_that_is_never_entered_or_awaited_is_not_outbound_shaped():
    """Otherwise the propagation would sweep in every `x = d.get(k)` in the package."""
    source = (
        "def grab(payload):\n"
        "    value = payload.get('key')\n"
        "    return value\n"
    )
    assert unclassified_outbound_calls(_synthetic(source)) == []
