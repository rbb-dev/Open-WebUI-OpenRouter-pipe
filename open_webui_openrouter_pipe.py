"""
title: Open WebUI OpenRouter Responses Pipe
author: rbb-dev
author_url: https://github.com/rbb-dev
git_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: open_webui_openrouter_pipe
description: Stub loader that installs and imports the full pipe from GitHub
required_open_webui_version: 0.12.0
version: 2.8.0
requirements: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/archive/refs/tags/v2.8.0.zip
license: MIT
"""

from open_webui_openrouter_pipe import Pipe as BasePipe


class Pipe(BasePipe):
    pass


__all__ = ["Pipe"]
