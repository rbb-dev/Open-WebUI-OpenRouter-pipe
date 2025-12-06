"""Domain layer for OpenRouter Responses Pipe.

Business logic and core domain models.
Depends on core layer only (never imports from adapters).
"""

from .types import *
from .registry import OpenRouterModelRegistry, ModelFamily
from .history import transform_messages_to_input
from .multimodal import (
    get_effective_remote_file_limit_mb,
    validate_base64_size,
    parse_data_url,
    download_remote_file,
    emit_status,
)
from .tools import (
    build_tools,
    _strictify_schema,
    _dedupe_tools,
)
