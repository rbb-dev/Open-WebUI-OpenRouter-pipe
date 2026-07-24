"""Tool execution subsystem.

This package contains tool-related functionality:
- tool_executor: Tool call execution orchestrator, worker loop, and direct tool server registry
- tool_schema: JSON schema strictification for structured outputs
- tool_registry: Tool registration, collision handling, and spec building

The tool subsystem manages the full lifecycle of function calling from
schema validation through execution and result handling.

NOTE: Imports are not eagerly loaded to avoid triggering Open WebUI database
initialization during package import. Import directly from submodules as needed.
"""
