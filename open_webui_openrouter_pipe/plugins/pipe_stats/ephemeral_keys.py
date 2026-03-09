"""Re-export of :class:`EphemeralKeyStore` from the shared plugin utilities.

The canonical implementation lives in :mod:`plugins._utils`.  This module
preserves backward-compatible import paths for both ``pipe_stats`` and
``think_streaming`` (which imports via ``..pipe_stats.ephemeral_keys``).
"""

from .._utils import EphemeralKeyStore, _EK_DEFAULT_TTL

# Backward-compatible alias used by tests
_PS_EK_DEFAULT_TTL = _EK_DEFAULT_TTL

__all__ = ["EphemeralKeyStore", "_PS_EK_DEFAULT_TTL"]
