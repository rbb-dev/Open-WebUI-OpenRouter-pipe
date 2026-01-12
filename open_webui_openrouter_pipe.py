"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: Stub loader that installs and imports the full pipe from GitHub
required_open_webui_version: 0.6.28
version: 1.1.1
requirements: git+https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe.git@auto-update
license: MIT

This is a lightweight stub for Open WebUI.

===============================================================================
███████╗████████╗ ██████╗ ██████╗
██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
███████╗   ██║   ██║   ██║██████╔╝
╚════██║   ██║   ██║   ██║██╔═══╝
███████║   ██║   ╚██████╔╝██║
╚══════╝   ╚═╝    ╚═════╝ ╚═╝
===============================================================================

WARNING: This file is intentionally a *stub loader* and is configured to auto-update.

- This stub tracks the moving `auto-update` branch (not a fixed version).
- This means Open WebUI can pull a newer version of the real pipe implementation over time
  (for example when dependencies are reinstalled / the function environment is refreshed).

If you need a fixed, reproducible deployment:
- Change `requirements:` to pin to a tag (e.g. `@v1.2.3`) or an exact commit SHA (e.g. `@abc123...`).

If you fork this repo:
- Update the `requirements:` URL to point to *your fork*, otherwise Open WebUI will keep installing *this* upstream repo.

If you do not want auto-updates:
- Install the full pipe directly instead of this stub (upload `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py` into Open WebUI).
"""

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe as BasePipe

_MODULE_PREFIX = "function_"
_runtime_id = __name__[len(_MODULE_PREFIX) :] if __name__.startswith(_MODULE_PREFIX) else BasePipe.id


class Pipe(BasePipe):
    id = _runtime_id

__all__ = ["Pipe"]
