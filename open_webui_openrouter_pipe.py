"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: Stub loader that installs and imports the full pipe from GitHub
required_open_webui_version: 0.7.0
version: 2.1.0
requirements: git+https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe.git@v2.1.0
license: MIT

This is a lightweight stub for Open WebUI.

If you fork this repo:
- Update the `requirements:` URL to point to *your fork*, otherwise Open WebUI will keep installing from *this* upstream repo.

"""

from open_webui_openrouter_pipe import Pipe as BasePipe

_MODULE_PREFIX = "function_"
_runtime_id = __name__[len(_MODULE_PREFIX) :] if __name__.startswith(_MODULE_PREFIX) else BasePipe.id


class Pipe(BasePipe):
    id = _runtime_id

__all__ = ["Pipe"]
