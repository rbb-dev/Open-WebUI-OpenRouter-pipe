"""User database operations.

This module provides user-related database operations for the pipe,
wrapping Open WebUI's Users model with async database operations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..core.timing_logger import timed

# Optional Open WebUI Users model
try:
    from open_webui.models.users import Users
except ImportError:
    Users = None  # type: ignore


@timed
async def get_user_by_id(user_id: str, logger: logging.Logger) -> Optional[Any]:
    """Fetch user record from database for file upload operations.

    Args:
        user_id: The unique identifier for the user
        logger: Logger instance for error reporting

    Returns:
        UserModel object if found, None otherwise

    Note:
        Failures are logged but do not raise exceptions.
    """
    if Users is None:
        return None
    try:
        return await Users.get_user_by_id(user_id)
    except Exception as exc:
        logger.error(f"Failed to load user {user_id}: {exc}")
        return None
