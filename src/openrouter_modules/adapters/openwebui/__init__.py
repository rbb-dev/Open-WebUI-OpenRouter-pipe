"""Open WebUI adapters.

Integration modules for Open WebUI-specific functionality:
- Artifact persistence (SQLAlchemy + Redis)
- Event emitters
- File upload handlers
- Tool execution wrappers
"""

from .persistence import (
    ArtifactPersistence,
    _sanitize_table_fragment,
    _extract_internal_file_id,
)
from .events import (
    wrap_event_emitter,
    merge_usage_stats,
    wrap_code_block,
)
from .file_handler import (
    get_user_by_id,
    get_file_by_id,
    infer_file_mime_type,
    inline_internal_file_url,
    read_file_record_base64,
    encode_file_path_base64,
    upload_to_owui_storage,
    is_youtube_url,
    is_safe_url,
    is_safe_url_blocking,
)

__all__ = [
    "ArtifactPersistence",
    "_sanitize_table_fragment",
    "_extract_internal_file_id",
    "wrap_event_emitter",
    "merge_usage_stats",
    "wrap_code_block",
    "get_user_by_id",
    "get_file_by_id",
    "infer_file_mime_type",
    "inline_internal_file_url",
    "read_file_record_base64",
    "encode_file_path_base64",
    "upload_to_owui_storage",
    "is_youtube_url",
    "is_safe_url",
    "is_safe_url_blocking",
]
