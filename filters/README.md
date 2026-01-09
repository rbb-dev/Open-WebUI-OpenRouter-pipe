# filters/

This folder contains reference Open WebUI filter functions used alongside the OpenRouter pipe.

## OpenRouter Search toggle

- `openrouter_search_toggle.py` is the companion *toggleable filter* used to enable OpenRouter-native web search.
- The pipe embeds a canonical copy of this filter and can **auto-install/auto-update** it into Open WebUI’s Functions DB when `AUTO_INSTALL_ORS_FILTER` is enabled.
- The filter is included here for review and manual installation, but **the live behavior is driven by the embedded copy inside the pipe** when auto-install is enabled.

If you edit `openrouter_search_toggle.py` in this repo, it will not automatically update your running Open WebUI unless you paste/install it manually (or you update the embedded filter source in the pipe).

## Direct Uploads toggle

- `openrouter_direct_uploads_toggle.py` is the companion *toggleable filter* used to bypass Open WebUI RAG for chat uploads and forward them to OpenRouter as direct file/audio/video inputs.
- The pipe embeds a canonical copy of this filter and can **auto-install/auto-update** it into Open WebUI’s Functions DB when `AUTO_INSTALL_DIRECT_UPLOADS_FILTER` is enabled.
- The filter is included here for review and manual installation, but **the live behavior is driven by the embedded copy inside the pipe** when auto-install is enabled.

If you edit `openrouter_direct_uploads_toggle.py` in this repo, it will not automatically update your running Open WebUI unless you paste/install it manually (or you update the embedded filter source in the pipe).

Reference documentation:
- [OpenRouter Direct Uploads (bypass OWUI RAG)](../docs/openrouter_direct_uploads.md)
