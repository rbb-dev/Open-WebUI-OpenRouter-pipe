"""Health command — show subsystem status badges."""

from __future__ import annotations

from ..command_registry import register_command
from ..context import CommandContext
from ..formatters import markdown_table


@register_command("health", summary="Show system health status", category="Diagnostics")
async def handle_health(ctx: CommandContext) -> str:
    """Display health status for all major subsystems."""
    pipe = ctx.pipe
    rows: list[list[str]] = []

    # Circuit breaker
    cb = getattr(pipe, "_circuit_breaker", None)
    if cb is not None:
        open_users = len(getattr(cb, "_open_circuits", {}))
        cb_status = "OPEN" if open_users > 0 else "OK"
        rows.append(["Circuit Breaker", cb_status, f"{open_users} open circuit(s)"])
    else:
        rows.append(["Circuit Breaker", "N/A", "Not initialized"])

    # Redis
    redis_enabled = getattr(pipe, "_redis_enabled", False)
    redis_client = getattr(pipe, "_redis_client", None)
    if redis_enabled and redis_client:
        rows.append(["Redis", "OK", "Connected"])
    elif redis_enabled:
        rows.append(["Redis", "DOWN", "Enabled but no client"])
    else:
        rows.append(["Redis", "OFF", "Disabled"])

    # Request queue
    queue = getattr(pipe, "_request_queue", None)
    if queue is not None:
        qsize = queue.qsize()
        rows.append(["Request Queue", "OK", f"{qsize} pending"])
    else:
        rows.append(["Request Queue", "N/A", "Not initialized"])

    # Model catalog
    catalog = getattr(pipe, "_catalog_manager", None)
    if catalog is not None:
        rows.append(["Model Catalog", "OK", "Loaded"])
    else:
        rows.append(["Model Catalog", "N/A", "Not initialized"])

    # Artifact store
    store = getattr(pipe, "_artifact_store", None)
    if store is not None:
        has_session = getattr(store, "_session_factory", None) is not None
        has_model = getattr(store, "_item_model", None) is not None
        if has_session and has_model:
            rows.append(["Artifact Store", "OK", "DB connected"])
        elif has_session:
            rows.append(["Artifact Store", "PARTIAL", "Session but no model"])
        else:
            rows.append(["Artifact Store", "STANDBY", "Not yet initialized"])
    else:
        rows.append(["Artifact Store", "N/A", "Not created"])

    # HTTP session
    http_session = getattr(pipe, "_http_session", None)
    if http_session is not None and not http_session.closed:
        rows.append(["HTTP Session", "OK", "Active"])
    else:
        rows.append(["HTTP Session", "STANDBY", "Not active"])

    header = "## System Health\n"
    table = markdown_table(["Subsystem", "Status", "Details"], rows)
    return header + table
