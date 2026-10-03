"""Authorization helpers for Pipe Dashboard commands."""

from __future__ import annotations

ACCESS_DENIED_MD = (
    "## Access Denied\n\n"
    "The Pipe Dashboard is restricted to administrators.\n\n"
    "If you believe this is an error, contact your Open WebUI admin."
)

UNDETERMINED_MD = (
    "## Access Not Checked\n\n"
    "The Pipe Dashboard could not read your account, so it did not decide whether you "
    "may view it. Nothing was loaded and nothing was changed — restore the database, "
    "then reload."
)
