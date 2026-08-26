#!/usr/bin/env python3
"""Generate plugins/pipe_dashboard/_socketio_client.py from the vendored socket.io client.

    python scripts/build_pipe_dashboard_socketio.py

Reads scripts/vendor/socket.io.min.js (SHA-384 pinned, same vendor file the
Fusion build inlines), strips the sourceMappingURL, and writes the client as a
Python string constant so bundled deployments carry it without file I/O.
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOCKETIO_VENDOR = PROJECT_ROOT / "scripts" / "vendor" / "socket.io.min.js"
SOCKETIO_SHA384 = "sha384-kzavj5fiMwLKzzD1f8S7TeoVIEi7uKHvbTA3ueZkrzYq75pNQUiUi6Dy98Q3fxb0"
TARGET_PY = PROJECT_ROOT / "open_webui_openrouter_pipe" / "plugins" / "pipe_dashboard" / "_socketio_client.py"

HEADER = '''"""Generated module — vendored socket.io client (socket.io-client v4.8.3).

Regenerate with ``python scripts/build_pipe_dashboard_socketio.py``; do not edit.
"""

from __future__ import annotations

'''


def _sha384(text: str) -> str:
    return "sha384-" + base64.b64encode(hashlib.sha384(text.encode("utf-8")).digest()).decode("ascii")


def main() -> None:
    src = SOCKETIO_VENDOR.read_text(encoding="utf-8")
    digest = _sha384(src)
    assert digest == SOCKETIO_SHA384, f"vendored socket.io integrity mismatch: {digest}"
    assert "</script" not in src, "vendored socket.io contains </script which would break the inline tag"
    src = re.sub(r"//# sourceMappingURL=\S*", "", src).strip()

    module = (
        HEADER
        + f'SOCKETIO_UMD_SHA384 = "{_sha384(src)}"\n\n'
        + f"SOCKETIO_UMD = {src!r}\n"
    )
    assert "'''" not in module, "generated module contains a triple-quote sequence"
    assert "</script" not in module

    TARGET_PY.write_text(module, encoding="utf-8")
    print(f"wrote {TARGET_PY} ({len(module)} chars)")


if __name__ == "__main__":
    main()
