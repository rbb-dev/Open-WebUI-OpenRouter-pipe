"""OpenRouter pipe for Open WebUI.

This package provides the OpenRouter integration for Open WebUI, including:
- Domain subsystems: persistence, multimodal, streaming
- Infrastructure modules: config, registry, transforms, errors, helpers
- Tool subsystem: tool_registry, tool_schema, tool_executor
- Orchestrator: pipe (main Pipe class)

IMPORTANT: This module uses LAZY LOADING to avoid import-time side effects.
All imports are deferred until actually accessed via __getattr__.
This prevents triggering Open WebUI's heavy initialization (Alembic, embeddings, etc.)
during simple imports like `from open_webui_openrouter_pipe import Pipe`.
"""

from typing import TYPE_CHECKING, Any

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _get_version
except ImportError:
    __version__ = "2.7.3"  # Fallback if not installed as package
else:
    try:
        __version__ = _get_version("open-webui-openrouter-pipe")
    except PackageNotFoundError:
        __version__ = "2.7.3"  # Fallback if not installed as package

# -----------------------------------------------------------------------------
# Type hints only (no runtime import)
# -----------------------------------------------------------------------------

if TYPE_CHECKING:
    from .api.transforms import (
        CompletionsBody,
        ResponsesBody,
        _apply_disable_native_websearch_to_payload,
        _apply_identifier_valves_to_payload,
        _apply_model_fallback_to_payload,
        _apply_openrouter_trace_to_payload,
        _apply_provider_routing_params_to_payload,
        _filter_openrouter_request,
        _responses_payload_to_chat_completions_payload,
        _strip_disable_model_settings_params,
    )
    from .core.config import (
        _OPENROUTER_REFERER,
        _PIPE_RUNTIME_ID,
        DEFAULT_AUTHENTICATION_ERROR_TEMPLATE,
        DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE,
        DEFAULT_NETWORK_TIMEOUT_TEMPLATE,
        DEFAULT_OPENROUTER_ERROR_TEMPLATE,
        DEFAULT_RATE_LIMIT_TEMPLATE,
        LOGGER,
        EncryptedStr,
        UserValves,
        Valves,
        _detect_runtime_pipe_id,
        _select_openrouter_http_referer,
    )
    from .core.errors import (
        OpenRouterAPIError,
        StatusMessages,
        _build_error_template_values,
        _build_openrouter_api_error,
        _classify_retryable_http_error,
        _extract_openrouter_error_details,
        _read_rag_file_constraints,
        _resolve_error_model_context,
        _retry_after_seconds,
        _unwrap_config_value,
    )
    from .core.logging_system import (
        SessionLogger,
        _SessionLogArchiveJob,
        write_session_log_archive,
    )
    from .core.utils import (
        _OPEN_WEBUI_CONFIG_MODULE,
        _coerce_bool,
        _coerce_positive_int,
        _extract_feature_flags,
        _extract_marker_ulid,
        _get_open_webui_config_module,
        _iter_marker_spans,
        _normalize_optional_str,
        _normalize_string_list,
        _pretty_json,
        _render_error_template,
        _safe_json_loads,
        _sanitize_path_component,
        _serialize_marker,
        _template_value_present,
        contains_marker,
        merge_usage_stats,
        split_text_by_markers,
        wrap_code_block,
    )
    from .models.registry import (
        ModelFamily,
        OpenRouterModelRegistry,
        sanitize_model_id,
    )
    from .pipe import Pipe, _PipeJob
    from .requests import NonStreamingAdapter, TaskModelAdapter
    from .requests.debug import _debug_print_error_response, _debug_print_request
    from .storage.multimodal import (
        MultimodalHandler,
        _extract_openrouter_og_image,
        _guess_image_mime_type,
    )
    from .storage.owui_files import (
        extract_internal_file_id,
        is_internal_file_url,
    )
    from .storage.persistence import (
        _ENCRYPTED_PAYLOAD_VERSION,
        _PAYLOAD_FLAG_LZ4,
        _PAYLOAD_FLAG_PLAIN,
        ULID_LENGTH,
        ArtifactStore,
        _sanitize_table_fragment,
        generate_item_id,
        normalize_persisted_item,
    )
    from .streaming.event_emitter import EventEmitterHandler
    from .streaming.streaming_core import StreamingHandler, _wrap_event_emitter
    from .tools.tool_executor import _QueuedToolCall, _ToolExecutionContext
    from .tools.tool_registry import (
        _dedupe_tools,
        _responses_spec_from_owui_tool_cfg,
        build_tools,
    )
    from .tools.tool_schema import _classify_function_call_artifacts, _strictify_schema

    # Open WebUI / FastAPI re-exports are resolved lazily at runtime via __getattr__.
    # Define them for type checkers so __all__ is consistent without importing heavy deps.
    upload_file_handler: Any
    run_in_threadpool: Any


# -----------------------------------------------------------------------------
# Public API - All lazy loaded
# -----------------------------------------------------------------------------

__all__ = [
    "DEFAULT_AUTHENTICATION_ERROR_TEMPLATE",
    "DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE",
    "DEFAULT_NETWORK_TIMEOUT_TEMPLATE",
    "DEFAULT_OPENROUTER_ERROR_TEMPLATE",
    "DEFAULT_RATE_LIMIT_TEMPLATE",
    "LOGGER",
    "ULID_LENGTH",
    "_ENCRYPTED_PAYLOAD_VERSION",
    "_OPENROUTER_REFERER",
    "_OPEN_WEBUI_CONFIG_MODULE",
    "_PAYLOAD_FLAG_LZ4",
    "_PAYLOAD_FLAG_PLAIN",
    "_PIPE_RUNTIME_ID",
    # Domain subsystems
    "ArtifactStore",
    "CompletionsBody",
    "EncryptedStr",
    "EventEmitterHandler",
    "ModelFamily",
    "MultimodalHandler",
    "NonStreamingAdapter",
    # Error handling
    "OpenRouterAPIError",
    # Model registry
    "OpenRouterModelRegistry",
    # Main classes
    "Pipe",
    # Data transforms
    "ResponsesBody",
    # Logging
    "SessionLogger",
    "StatusMessages",
    "StreamingHandler",
    "TaskModelAdapter",
    "UserValves",
    "Valves",
    # Internal (for testing)
    "_PipeJob",
    "_QueuedToolCall",
    "_SessionLogArchiveJob",
    "_ToolExecutionContext",
    # Version
    "__version__",
    "_apply_disable_native_websearch_to_payload",
    "_apply_identifier_valves_to_payload",
    "_apply_model_fallback_to_payload",
    "_apply_openrouter_trace_to_payload",
    "_apply_provider_routing_params_to_payload",
    "_build_error_template_values",
    "_build_openrouter_api_error",
    "_classify_function_call_artifacts",
    "_classify_retryable_http_error",
    "_coerce_bool",
    "_coerce_positive_int",
    "_debug_print_error_response",
    "_debug_print_request",
    "_dedupe_tools",
    "_detect_runtime_pipe_id",
    "_extract_feature_flags",
    "_extract_marker_ulid",
    "_extract_openrouter_error_details",
    "_extract_openrouter_og_image",
    "_filter_openrouter_request",
    "_get_open_webui_config_module",
    "_guess_image_mime_type",
    "_iter_marker_spans",
    "_normalize_optional_str",
    "_normalize_string_list",
    "_pretty_json",
    "_read_rag_file_constraints",
    "_render_error_template",
    "_resolve_error_model_context",
    "_responses_payload_to_chat_completions_payload",
    "_responses_spec_from_owui_tool_cfg",
    "_retry_after_seconds",
    "_safe_json_loads",
    "_sanitize_path_component",
    "_sanitize_table_fragment",
    "_select_openrouter_http_referer",
    # Utility functions
    "_serialize_marker",
    # Helper functions
    "_strictify_schema",
    "_strip_disable_model_settings_params",
    "_template_value_present",
    "_unwrap_config_value",
    "_wrap_event_emitter",
    # Tool subsystem
    "build_tools",
    "contains_marker",
    "extract_internal_file_id",
    "generate_item_id",
    "is_internal_file_url",
    "merge_usage_stats",
    "normalize_persisted_item",
    "run_in_threadpool",
    "sanitize_model_id",
    "split_text_by_markers",
    # Open WebUI components
    "upload_file_handler",
    "wrap_code_block",
    "write_session_log_archive",
]


# -----------------------------------------------------------------------------
# Lazy Loading Implementation
# -----------------------------------------------------------------------------

# Cache for loaded attributes
_cache: dict = {}

# Mapping of attribute name to (module_path, attr_name_in_module)
_LAZY_IMPORTS = {
    # Core config
    "Valves": (".core.config", "Valves"),
    "UserValves": (".core.config", "UserValves"),
    "EncryptedStr": (".core.config", "EncryptedStr"),
    "_PIPE_RUNTIME_ID": (".core.config", "_PIPE_RUNTIME_ID"),
    "_OPENROUTER_REFERER": (".core.config", "_OPENROUTER_REFERER"),
    "DEFAULT_OPENROUTER_ERROR_TEMPLATE": (".core.config", "DEFAULT_OPENROUTER_ERROR_TEMPLATE"),
    "DEFAULT_NETWORK_TIMEOUT_TEMPLATE": (".core.config", "DEFAULT_NETWORK_TIMEOUT_TEMPLATE"),
    "DEFAULT_RATE_LIMIT_TEMPLATE": (".core.config", "DEFAULT_RATE_LIMIT_TEMPLATE"),
    "DEFAULT_AUTHENTICATION_ERROR_TEMPLATE": (".core.config", "DEFAULT_AUTHENTICATION_ERROR_TEMPLATE"),
    "DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE": (".core.config", "DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE"),
    "_detect_runtime_pipe_id": (".core.config", "_detect_runtime_pipe_id"),
    "LOGGER": (".core.config", "LOGGER"),
    "_select_openrouter_http_referer": (".core.config", "_select_openrouter_http_referer"),

    # Core errors
    "OpenRouterAPIError": (".core.errors", "OpenRouterAPIError"),
    "StatusMessages": (".core.errors", "StatusMessages"),
    "_build_error_template_values": (".core.errors", "_build_error_template_values"),
    "_unwrap_config_value": (".core.errors", "_unwrap_config_value"),
    "_retry_after_seconds": (".core.errors", "_retry_after_seconds"),
    "_classify_retryable_http_error": (".core.errors", "_classify_retryable_http_error"),
    "_extract_openrouter_error_details": (".core.errors", "_extract_openrouter_error_details"),
    "_resolve_error_model_context": (".core.errors", "_resolve_error_model_context"),
    "_build_openrouter_api_error": (".core.errors", "_build_openrouter_api_error"),
    "_read_rag_file_constraints": (".core.errors", "_read_rag_file_constraints"),

    # Core utils
    "_get_open_webui_config_module": (".core.utils", "_get_open_webui_config_module"),
    "_OPEN_WEBUI_CONFIG_MODULE": (".core.utils", "_OPEN_WEBUI_CONFIG_MODULE"),
    "_safe_json_loads": (".core.utils", "_safe_json_loads"),
    "_extract_feature_flags": (".core.utils", "_extract_feature_flags"),
    "_render_error_template": (".core.utils", "_render_error_template"),
    "_sanitize_path_component": (".core.utils", "_sanitize_path_component"),
    "_pretty_json": (".core.utils", "_pretty_json"),
    "_template_value_present": (".core.utils", "_template_value_present"),
    "_normalize_optional_str": (".core.utils", "_normalize_optional_str"),
    "merge_usage_stats": (".core.utils", "merge_usage_stats"),
    "wrap_code_block": (".core.utils", "wrap_code_block"),
    "_serialize_marker": (".core.utils", "_serialize_marker"),
    "contains_marker": (".core.utils", "contains_marker"),
    "split_text_by_markers": (".core.utils", "split_text_by_markers"),
    "_coerce_positive_int": (".core.utils", "_coerce_positive_int"),
    "_coerce_bool": (".core.utils", "_coerce_bool"),
    "_normalize_string_list": (".core.utils", "_normalize_string_list"),
    "_extract_marker_ulid": (".core.utils", "_extract_marker_ulid"),
    "_iter_marker_spans": (".core.utils", "_iter_marker_spans"),

    # API transforms
    "ResponsesBody": (".api.transforms", "ResponsesBody"),
    "CompletionsBody": (".api.transforms", "CompletionsBody"),
    "_apply_disable_native_websearch_to_payload": (".api.transforms", "_apply_disable_native_websearch_to_payload"),
    "_apply_identifier_valves_to_payload": (".api.transforms", "_apply_identifier_valves_to_payload"),
    "_responses_payload_to_chat_completions_payload": (".api.transforms", "_responses_payload_to_chat_completions_payload"),
    "_filter_openrouter_request": (".api.transforms", "_filter_openrouter_request"),
    "_strip_disable_model_settings_params": (".api.transforms", "_strip_disable_model_settings_params"),
    "_apply_model_fallback_to_payload": (".api.transforms", "_apply_model_fallback_to_payload"),
    "_apply_openrouter_trace_to_payload": (".api.transforms", "_apply_openrouter_trace_to_payload"),
    "_apply_provider_routing_params_to_payload": (".api.transforms", "_apply_provider_routing_params_to_payload"),
    "_get_disable_param": (".api.transforms", "_get_disable_param"),
    "_model_params_to_dict": (".api.transforms", "_model_params_to_dict"),

    # Models registry
    "OpenRouterModelRegistry": (".models.registry", "OpenRouterModelRegistry"),
    "ModelFamily": (".models.registry", "ModelFamily"),
    "sanitize_model_id": (".models.registry", "sanitize_model_id"),

    # Requests
    "_debug_print_request": (".requests.debug", "_debug_print_request"),
    "_debug_print_error_response": (".requests.debug", "_debug_print_error_response"),
    "NonStreamingAdapter": (".requests.nonstreaming_adapter", "NonStreamingAdapter"),
    "TaskModelAdapter": (".requests.task_model_adapter", "TaskModelAdapter"),

    # Storage persistence
    "normalize_persisted_item": (".storage.persistence", "normalize_persisted_item"),
    "ArtifactStore": (".storage.persistence", "ArtifactStore"),
    "generate_item_id": (".storage.persistence", "generate_item_id"),
    "_PAYLOAD_FLAG_LZ4": (".storage.persistence", "_PAYLOAD_FLAG_LZ4"),
    "_PAYLOAD_FLAG_PLAIN": (".storage.persistence", "_PAYLOAD_FLAG_PLAIN"),
    "_sanitize_table_fragment": (".storage.persistence", "_sanitize_table_fragment"),
    "ULID_LENGTH": (".storage.persistence", "ULID_LENGTH"),
    "_ENCRYPTED_PAYLOAD_VERSION": (".storage.persistence", "_ENCRYPTED_PAYLOAD_VERSION"),

    # Storage multimodal
    "extract_internal_file_id": (".storage.owui_files", "extract_internal_file_id"),
    "MultimodalHandler": (".storage.multimodal", "MultimodalHandler"),
    "_guess_image_mime_type": (".storage.multimodal", "_guess_image_mime_type"),
    "_extract_openrouter_og_image": (".storage.multimodal", "_extract_openrouter_og_image"),
    "is_internal_file_url": (".storage.owui_files", "is_internal_file_url"),

    # Tools
    "_classify_function_call_artifacts": (".tools.tool_schema", "_classify_function_call_artifacts"),
    "_strictify_schema": (".tools.tool_schema", "_strictify_schema"),
    "_responses_spec_from_owui_tool_cfg": (".tools.tool_registry", "_responses_spec_from_owui_tool_cfg"),
    "build_tools": (".tools.tool_registry", "build_tools"),
    "_dedupe_tools": (".tools.tool_registry", "_dedupe_tools"),

    # Streaming
    "_wrap_event_emitter": (".streaming.streaming_core", "_wrap_event_emitter"),
    "StreamingHandler": (".streaming.streaming_core", "StreamingHandler"),
    "EventEmitterHandler": (".streaming.event_emitter", "EventEmitterHandler"),

    # Logging
    "SessionLogger": (".core.logging_system", "SessionLogger"),
    "_SessionLogArchiveJob": (".core.logging_system", "_SessionLogArchiveJob"),
    "write_session_log_archive": (".core.logging_system", "write_session_log_archive"),

    # Pipe (heavy - triggers OWUI)
    "Pipe": (".pipe", "Pipe"),
    "_PipeJob": (".pipe", "_PipeJob"),

    # Tool executor
    "_QueuedToolCall": (".tools.tool_executor", "_QueuedToolCall"),
    "_ToolExecutionContext": (".tools.tool_executor", "_ToolExecutionContext"),

    # Plugins
    "PluginBase": (".plugins.base", "PluginBase"),
    "PluginContext": (".plugins.base", "PluginContext"),
    "PluginRegistry": (".plugins.registry", "PluginRegistry"),
}


def __getattr__(name: str):
    """Lazy-load all module attributes on first access.

    This prevents triggering Open WebUI's heavy initialization (Alembic,
    sentence-transformers, langchain, etc.) during simple imports.
    """
    # Return cached value if available
    if name in _cache:
        return _cache[name]

    # Handle lazy imports
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        _cache[name] = value
        globals()[name] = value  # Also cache in globals for faster subsequent access
        return value

    # Handle Open WebUI re-exports (may be None if not available)
    if name == "upload_file_handler":
        try:
            from open_webui.routers.files import upload_file_handler as _handler
            _cache[name] = _handler
            globals()[name] = _handler  # cache for faster subsequent access
            return _handler
        except ImportError:
            _cache[name] = None
            return None

    if name == "run_in_threadpool":
        try:
            from fastapi.concurrency import run_in_threadpool as _run
            _cache[name] = _run
            globals()[name] = _run  # cache for faster subsequent access
            return _run
        except ImportError:
            _cache[name] = None
            return None

    # Handle submodule access (e.g., open_webui_openrouter_pipe.errors)
    submodules = {
        "errors": ".core.errors",
        "config": ".core.config",
        "utils": ".core.utils",
        "logging_system": ".core.logging_system",
        "circuit_breaker": ".core.circuit_breaker",
        "error_formatter": ".core.error_formatter",
        "persistence": ".storage.persistence",
        "multimodal": ".storage.multimodal",
        "transforms": ".api.transforms",
        "registry": ".models.registry",
    }
    if name in submodules:
        import importlib
        module = importlib.import_module(submodules[name], __name__)
        _cache[name] = module
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
