from __future__ import annotations

import tempfile
from pathlib import Path


def _sorter_stream(events, *, dedupe, prior=None):
    from open_webui_openrouter_pipe.logging.session_log_manager import _EventSorter

    with tempfile.TemporaryDirectory() as tmp:
        sorter = _EventSorter(Path(tmp))
        try:
            for pos, evt in enumerate(prior or []):
                sorter.add("", -1, pos + 1, evt)
            for pos, evt in enumerate(events):
                sorter.add("", 0, pos, evt)
            return list(sorter.stream("", dedupe))
        finally:
            sorter.close()
