"""Filter management module.

Provides FilterManager for managing OWUI filter functions:
- OpenRouter Web Tools filter (Web Search, Web Fetch, Datetime)
- OpenRouter Image Generation filter
- Direct Uploads filter
- Provider Routing filters
"""

from .filter_manager import FilterManager

__all__ = ["FilterManager"]
