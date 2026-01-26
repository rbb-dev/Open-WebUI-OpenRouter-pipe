#!/usr/bin/env bash
set -euo pipefail

# Reproduce this repo's Python virtualenv using uv (10-100x faster than pip).
# Uses latest available versions (no pins).
#
# uv is a drop-in replacement for pip, written in Rust by Astral.
# https://github.com/astral-sh/uv

# Get the directory where this script lives (the project's scripts/ folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"
CPU_ONLY="${CPU_ONLY:-1}"  # Default to CPU-only (saves ~5GB, faster installs)

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing..." >&2
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv version: $(uv --version)"

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

# Create venv with uv (faster than python -m venv)
uv venv "$VENV_DIR" --python "$PYTHON_BIN"

# Install PyTorch first (CPU-only by default to save ~5GB and speed up installs)
if [[ "$CPU_ONLY" == "1" ]]; then
    echo "Installing CPU-only PyTorch (set CPU_ONLY=0 for CUDA support)..."
    uv pip install --python "$VENV_DIR/bin/python" \
        torch --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch with CUDA support..."
    uv pip install --python "$VENV_DIR/bin/python" torch
fi

# Install everything else (uv handles this efficiently)
# No need for separate pip/setuptools/wheel upgrade - uv doesn't need them
uv pip install --python "$VENV_DIR/bin/python" \
    open-webui \
    aiohttp cairosvg cryptography fastapi httpx lz4 Pillow pydantic pydantic-core pyzipper sqlalchemy tenacity \
    pytest pytest-asyncio pytest-cov aioresponses \
    ruff pyright vulture pyflakes \
    git-filter-repo

echo ""
echo "Done! Activate with: source $VENV_DIR/bin/activate"
