"""Tool execution subsystem.

This package contains tool-related functionality:
- tool_executor: Tool call execution orchestrator and direct tool server registry
- tool_worker: Worker loop and batching logic for tool execution
- tool_schema: JSON schema strictification for structured outputs
- tool_registry: Tool registration, collision handling, and spec building

The tool subsystem manages the full lifecycle of function calling from
schema validation through execution and result handling.

NOTE: Imports are not eagerly loaded to avoid triggering Open WebUI database
initialization during package import. Import directly from submodules as needed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tool_executor import _QueuedToolCall, _ToolExecutionContext, ToolExecutor
    from .tool_worker import _tool_worker_loop, _can_batch_tool_calls, _args_reference_call
    from .tool_schema import _strictify_schema, _strictify_schema_impl
    from .tool_registry import build_tools, _dedupe_tools, _build_collision_safe_tool_specs_and_registry

__all__ = [
    "_QueuedToolCall",
    "_ToolExecutionContext",
    "ToolExecutor",
    "_tool_worker_loop",
    "_can_batch_tool_calls",
    "_args_reference_call",
    "_strictify_schema",
    "_strictify_schema_impl",
    "build_tools",
    "_dedupe_tools",
    "_build_collision_safe_tool_specs_and_registry",
]
