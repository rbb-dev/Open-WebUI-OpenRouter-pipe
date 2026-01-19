"""
Blocklists for model capabilities.

These blocklists override automatic capability detection for models that have
known issues with specific features. The blocklist approach allows enabling
features by default while excluding known-incompatible models.

Blocklists are derived from empirical testing against the OpenRouter API.
"""

# =============================================================================
# DIRECT UPLOAD BLOCKLIST
# =============================================================================
#
# Models that should NOT receive direct file uploads, even though most models
# on OpenRouter support them.
#
# Background:
#   OpenRouter's model catalog (`architecture.input_modalities`) only declares
#   ~53 models as supporting "file" input. However, extensive testing (287 models)
#   revealed that ~83% of models actually accept and process file uploads correctly.
#
#   Rather than gating on incomplete upstream metadata, we now default to enabling
#   direct uploads for all models and maintain this blocklist of known failures.
#
# Blocklist categories:
#   - Provider rejected (HTTP 400): Model/provider explicitly rejects file input
#   - Guard/classifier models: Safety models not designed for chat with files
#   - Explicit "can't do files": Models that return 200 but state they cannot process files
#   - Empty/broken responses: Models that fail to produce meaningful output with files
#
# Last updated: 2026-01-19
# Test coverage: 287 models tested with PDF file upload
# Success rate: 239/287 (83.3%)
#

DIRECT_UPLOAD_BLOCKLIST: frozenset[str] = frozenset({
    # --- Provider rejected (HTTP 400) ---
    "liquid/lfm2-8b-a1b",
    "liquid/lfm-2.2-6b",
    "relace/relace-apply-3",
    "openai/gpt-4o-audio-preview",
    "arcee-ai/spotlight",
    "arcee-ai/coder-large",

    # --- Guard/classifier models (not designed for chat) ---
    "meta-llama/llama-guard-2-8b",
    "meta-llama/llama-guard-3-8b",
    "meta-llama/llama-guard-4-12b",

    # --- Explicit "can't process files" responses ---
    "openai/gpt-4-turbo",
    "openai/gpt-4-1106-preview",
    "qwen/qwen2.5-coder-7b-instruct",

    # --- Empty or broken responses with files ---
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen-2.5-vl-7b-instruct",
    "qwen/qwen-2.5-vl-7b-instruct:free",
    "meta-llama/llama-3.1-405b",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3.2-1b-instruct",

    # --- Hallucinated/incorrect file content ---
    "z-ai/glm-4-32b",
    "alfredpros/codellama-7b-instruct-solidity",
    "nousresearch/hermes-3-llama-3.1-70b",
    "mancer/weaver",
    "undi95/remm-slerp-l2-13b",
    "gryphe/mythomax-l2-13b",
})


def is_direct_upload_blocklisted(model_id: str) -> bool:
    """Check if a model is on the direct uploads blocklist.

    Args:
        model_id: The model identifier (e.g., "openai/gpt-4-turbo")

    Returns:
        True if the model is blocklisted (should NOT receive direct file uploads).
    """
    # Normalize: strip whitespace, lowercase for comparison
    normalized = model_id.strip().lower() if model_id else ""
    return any(normalized == blocklisted.lower() for blocklisted in DIRECT_UPLOAD_BLOCKLIST)
