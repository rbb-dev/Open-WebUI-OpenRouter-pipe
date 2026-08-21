# filters/

This folder contains reference Open WebUI filter functions used alongside the OpenRouter pipe.

## OpenRouter Web Tools

- `openrouter_web_tools.py` is the companion *toggleable filter* used to configure OpenRouter server tools (web search, web fetch, datetime) for the OpenRouter pipe.
- The pipe embeds a canonical copy of this filter and can **auto-install/auto-update** it into Open WebUI’s Functions DB when `AUTO_INSTALL_WEB_TOOLS_FILTER` is enabled.
- The filter is included here for review and manual installation, but **the live behavior is driven by the embedded copy inside the pipe** when auto-install is enabled.

If you edit `openrouter_web_tools.py` in this repo, it will not automatically update your running Open WebUI unless you paste/install it manually (or you update the embedded filter source in the pipe).

## OpenRouter Image Generation

- `openrouter_image_gen.py` is the companion *toggleable filter* for OpenRouter's image generation server tool.
- **This file is generated, not hand-written.** It is `FilterManager.render_openrouter_image_gen_filter_source(dedicated_image_api=True)` with no model and no contract, and `tests/test_web_tools_filter.py` fails if the two diverge. Edit the renderer in `open_webui_openrouter_pipe/filters/image_filter_renderer.py`, then regenerate.
- The settings it offers come from the selected model's own published list of what it accepts. With no arguments the renderer has no such list to read, so the copy here shows the unread case: the six controls are all there, but each offers what OpenRouter's image API accepts in general instead of that model's own choices, and the `IMAGE_GENERATION_MODEL` valve's description says so. The pipe re-renders it for whichever model that valve names, and the installed copy narrows those controls to that model's own values.
- The pipe can **auto-install/auto-update** it into Open WebUI's Functions DB when `AUTO_INSTALL_IMAGE_GEN_FILTER` is enabled.
- The filter is included here for review and manual installation, but **the live behavior is driven by the embedded copy inside the pipe** when auto-install is enabled.

## Direct Uploads toggle

- `openrouter_direct_uploads_toggle.py` is the companion *toggleable filter* used to bypass Open WebUI RAG for chat uploads and forward them to OpenRouter as direct file/audio/video inputs.
- The pipe embeds a canonical copy of this filter and can **auto-install/auto-update** it into Open WebUI’s Functions DB when `AUTO_INSTALL_DIRECT_UPLOADS_FILTER` is enabled.
- The filter is included here for review and manual installation, but **the live behavior is driven by the embedded copy inside the pipe** when auto-install is enabled.

If you edit `openrouter_direct_uploads_toggle.py` in this repo, it will not automatically update your running Open WebUI unless you paste/install it manually (or you update the embedded filter source in the pipe).

Reference documentation:
- [OpenRouter Direct Uploads (bypass OWUI RAG)](../docs/openrouter_direct_uploads.md)
