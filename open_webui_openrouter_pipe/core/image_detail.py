from __future__ import annotations

from typing import Any

DEFAULT_IMAGE_DETAIL = "auto"


def image_detail_or_auto(detail: Any) -> str:
    return detail if isinstance(detail, str) and detail else DEFAULT_IMAGE_DETAIL
