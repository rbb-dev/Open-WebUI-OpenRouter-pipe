from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class VideoGenerationError(RuntimeError):
    pass


@dataclass(slots=True)
class VideoGenerationResult:

    content: str
    status_description: str
    usage: dict[str, Any] = field(default_factory=dict)
    job_id: str = ""
    file_id: str | None = None
    failed: bool = False


@dataclass(slots=True)
class VideoLifecycleResult(VideoGenerationResult):

    elapsed: float = 0.0
    model_id: str = ""
    output_mime: str = ""


@dataclass(slots=True)
class DownloadedVideo:

    path: Path
    mime_type: str
    size_bytes: int
