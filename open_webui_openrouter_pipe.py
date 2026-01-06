"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: Stub loader that installs and imports the full pipe from GitHub
required_open_webui_version: 0.6.28
version: 1.0.19
requirements: git+https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe.git@auto-update
license: MIT

This is a lightweight stub for Open WebUI.
Open WebUI installs the full implementation from GitHub (via the `requirements` line above), then this file imports `Pipe` from the installed package.
"""

# Open WebUI installs the dependency (via frontmatter) then imports `Pipe` from it.
#
# Important: users may set a different function id in the Open WebUI UI. When this stub is
# executed, Open WebUI loads it as a module named "function_<id>", so we can derive the
# runtime id from `__name__` and expose a wrapper `Pipe` with a matching `.id`.
from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe as BasePipe

_MODULE_PREFIX = "function_"
_runtime_id = __name__[len(_MODULE_PREFIX) :] if __name__.startswith(_MODULE_PREFIX) else BasePipe.id


class Pipe(BasePipe):
    id = _runtime_id

__all__ = ["Pipe"]
