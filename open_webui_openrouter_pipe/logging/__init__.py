"""Logging subsystem.

This package provides logging and session log management:
- SessionLogManager: Background workers for log archival and assembly
"""

from __future__ import annotations

from .session_log_manager import SessionLogManager

__all__ = [
    "SessionLogManager",
]
