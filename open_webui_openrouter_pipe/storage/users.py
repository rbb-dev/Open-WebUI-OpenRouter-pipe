from __future__ import annotations

import logging
from typing import Any

from ..core.timing_logger import timed

try:
    from open_webui.models.users import Users
except ImportError:
    Users = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.users failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    Users = None  # type: ignore


@timed
async def get_user_by_id(user_id: str, logger: logging.Logger) -> Any | None:
    if Users is None:
        return None
    try:
        return await Users.get_user_by_id(user_id)
    except Exception:
        logger.exception("Failed to load user %s", user_id)
        return None
