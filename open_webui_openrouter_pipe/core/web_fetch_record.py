from __future__ import annotations

FETCH_RECORD_MAX_CHARS = 150_000


def bound_fetch_record(text: str) -> str:
    if len(text) <= FETCH_RECORD_MAX_CHARS:
        return text
    omitted = len(text) - FETCH_RECORD_MAX_CHARS
    return f"{text[:FETCH_RECORD_MAX_CHARS]}\n[... {omitted} characters of this page are not kept]"
