#!/usr/bin/env bash
set -euo pipefail

# Reproduce this repo's Python virtualenv for other users.
# Uses latest available versions (no pins).

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

# Long network timeouts/retries (Open WebUI installs can take a long time).
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-600}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

mkdir -p "$(dirname "$VENV_DIR")" 2>/dev/null || true

if [[ -d "$VENV_DIR" ]]; then
  if [[ "$FORCE_RECREATE" == "1" ]]; then
    rm -rf "$VENV_DIR"
  else
    echo "Venv already exists at: $VENV_DIR" >&2
    echo "Re-run with FORCE_RECREATE=1 to recreate it from scratch." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# --- installs (append as we go) ---
"$VENV_DIR/bin/python" -m pip install --upgrade --prefer-binary open-webui
"$VENV_DIR/bin/python" -m pip install --upgrade -e .
"$VENV_DIR/bin/python" -m pip install --upgrade pytest pytest-asyncio pytest-cov
"$VENV_DIR/bin/python" -m pip install --upgrade ruff pyright vulture pyflakes
"$VENV_DIR/bin/python" -m pip install --upgrade git-filter-repo
"$VENV_DIR/bin/python" -m pip install --upgrade cairosvg
