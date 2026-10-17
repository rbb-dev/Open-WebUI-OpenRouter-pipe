from __future__ import annotations

import sys
import threading
from typing import Any

_PROCESS_UPGRADE_LOCK = threading.Lock()


def _upgrade_process_state(key: str, current_type: type[Any]) -> Any:
    stored = sys.modules.get(key)
    if type(stored) is current_type:
        return stored
    with _PROCESS_UPGRADE_LOCK:
        stored = sys.modules.get(key)
        if type(stored) is current_type:
            return stored
        replacement = current_type()
        previous = sys.modules.get(key)
        if previous is not None:
            replacement.adopt_from(previous)
        sys.modules[key] = replacement  # type: ignore[assignment]
        return replacement
