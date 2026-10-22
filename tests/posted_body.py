from __future__ import annotations

import json
from typing import Any


def posted_json(kwargs: dict[str, Any], default: Any = None) -> Any:
    data = kwargs.get("data")
    if data is not None and hasattr(data, "decode"):
        return json.loads(data.decode())
    return kwargs.get("json", default)
