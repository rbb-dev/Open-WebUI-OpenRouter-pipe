import re
from typing import Any

DIRECT_UPLOAD_BLOCKLIST: frozenset[str] = frozenset({
    "liquid/lfm2-8b-a1b",
    "liquid/lfm-2.2-6b",
    "relace/relace-apply-3",
    "openai/gpt-4o-audio-preview",
    "arcee-ai/spotlight",
    "arcee-ai/coder-large",

    "meta-llama/llama-guard-2-8b",
    "meta-llama/llama-guard-3-8b",
    "meta-llama/llama-guard-4-12b",

    "openai/gpt-4-turbo",
    "openai/gpt-4-1106-preview",
    "qwen/qwen2.5-coder-7b-instruct",

    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen-2.5-vl-7b-instruct",
    "qwen/qwen-2.5-vl-7b-instruct:free",
    "meta-llama/llama-3.1-405b",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3.2-1b-instruct",

    "z-ai/glm-4-32b",
    "alfredpros/codellama-7b-instruct-solidity",
    "nousresearch/hermes-3-llama-3.1-70b",
    "mancer/weaver",
    "undi95/remm-slerp-l2-13b",
    "gryphe/mythomax-l2-13b",
    "eleutherai/llemma_7b",
    "ai21/jamba-mini-1.7",
    "cohere/command-r7b-12-2024",
    "neversleep/noromaid-20b",

    "allenai/olmo-3-32b-think"
})

_NORMALIZED_DIRECT_UPLOAD_BLOCKLIST: frozenset[str] = frozenset(
    entry.strip().lower() for entry in DIRECT_UPLOAD_BLOCKLIST
)
_DATE_SUFFIX = re.compile(
    r"-\d{4}-\d{2}-\d{2}$|-\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])$"
)


def is_direct_upload_blocklisted(model_id: str) -> bool:
    normalized = model_id.strip().lower() if model_id else ""
    candidate = normalized
    while True:
        if candidate in _NORMALIZED_DIRECT_UPLOAD_BLOCKLIST:
            return True
        undated = _DATE_SUFFIX.sub("", candidate)
        if undated != candidate and undated in _NORMALIZED_DIRECT_UPLOAD_BLOCKLIST:
            return True
        base, separator, _tag = candidate.rpartition(":")
        if not separator or not base:
            return False
        candidate = base


def is_direct_upload_blocklisted_for(model_id: str, row: Any) -> bool:
    if is_direct_upload_blocklisted(model_id):
        return True
    target = row.get("alias_target") if isinstance(row, dict) else None
    slug = target.get("slug") if isinstance(target, dict) else None
    return isinstance(slug, str) and bool(slug.strip()) and is_direct_upload_blocklisted(slug.strip())
