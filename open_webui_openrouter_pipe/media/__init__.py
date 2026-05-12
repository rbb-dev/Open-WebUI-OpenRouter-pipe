"""Generic media helpers (image conversion, frame extraction, thumbnails).

Feature-agnostic — designed for reuse by future image-gen task-model features.
"""
from __future__ import annotations

from .frame_extraction import (
    ExtractedFrame,
    FrameExtractionError,
    VideoMetadata,
    extract_frame,
    probe_video,
)
from .image_conversion import composite_on_white, normalise_mime
from .thumbnail import Thumbnail, make_thumbnail

__all__ = [
    "composite_on_white",
    "normalise_mime",
    "ExtractedFrame",
    "FrameExtractionError",
    "VideoMetadata",
    "extract_frame",
    "probe_video",
    "Thumbnail",
    "make_thumbnail",
]
