"""Message history translation for OpenRouter Responses Pipe.

This module handles converting Open WebUI chat messages to Responses API format,
including artifact loading, ULID marker parsing, tool output replay, and reasoning
token management.

Layer: domain (imports from core, never from adapters)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..core import markers, logging as core_logging

LOGGER = core_logging.SessionLogger.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Message Translation: Open WebUI → Responses API
# ─────────────────────────────────────────────────────────────────────────────

async def transform_messages_to_input(
    messages: List[Dict[str, Any]],
    *,
    chat_id: Optional[str] = None,
    openwebui_model_id: Optional[str] = None,
    artifact_loader: Optional[Callable[[Optional[str], Optional[str], List[str]], Awaitable[Dict[str, Dict[str, Any]]]]] = None,
    pruning_turns: int = 0,
    replayed_reasoning_refs: Optional[List[Tuple[str, str]]] = None,
    valves: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Build an OpenAI Responses-API `input` array from Open WebUI-style messages.

    This function:
    1. Converts Open WebUI message format to Responses API format
    2. Loads persisted artifacts (reasoning, tool outputs) via artifact_loader
    3. Parses ULID markers from assistant messages
    4. Reconstructs complete tool calls + outputs from markers
    5. Prunes old tool outputs to save tokens
    6. Handles reasoning token replay and cleanup

    Args:
        messages: Open WebUI message list
        chat_id: Optional chat ID for artifact loading
        openwebui_model_id: Optional model ID for scoped artifact tables
        artifact_loader: Async function to load artifacts from DB
        pruning_turns: Number of recent turns to keep full tool outputs
        replayed_reasoning_refs: List to append (chat_id, artifact_id) for cleanup
        valves: Valve configuration
        logger: Logger instance

    Returns:
        List[dict]: Responses API `input` array
    """
    logger = logger or LOGGER

    def _extract_plain_text(content: Any) -> str:
        """Collapse Open WebUI content blocks into a single string."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text_val = block.get("text") or block.get("content")
                    if isinstance(text_val, str):
                        parts.append(text_val)
            return "\n".join(parts)
        if isinstance(content, dict):
            text_val = content.get("text") or content.get("content")
            if isinstance(text_val, str):
                return text_val
        return str(content or "")

    # Build Responses API input array
    input_items: List[Dict[str, Any]] = []

    # Track logical turns for pruning
    turn_boundaries: List[int] = []
    current_turn_start = 0

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        # User messages start new turns
        if role == "user":
            if msg_idx > 0:
                turn_boundaries.append(current_turn_start)
            current_turn_start = len(input_items)

            # Add user message
            input_items.append({
                "type": "message",
                "role": "user",
                "content": _extract_plain_text(content),
            })

        elif role == "assistant":
            # Parse ULID markers from assistant message
            text = _extract_plain_text(content)
            ulids = _extract_ulids_from_text(text)

            # Load artifacts if available
            artifacts: Dict[str, Dict[str, Any]] = {}
            if chat_id and openwebui_model_id and artifact_loader and ulids:
                try:
                    artifacts = await artifact_loader(chat_id, msg.get("id"), ulids)
                except Exception as exc:
                    logger.warning(f"Failed to load artifacts for message {msg.get('id')}: {exc}")

            # Reconstruct items from artifacts
            reconstructed = _reconstruct_items_from_artifacts(
                text=text,
                ulids=ulids,
                artifacts=artifacts,
                pruning_turns=pruning_turns,
                current_turn_index=len(turn_boundaries),
                replayed_reasoning_refs=replayed_reasoning_refs,
                chat_id=chat_id,
            )

            input_items.extend(reconstructed)

    return input_items


def _extract_ulids_from_text(text: str) -> List[str]:
    """Extract ULID markers from assistant message text.

    Args:
        text: Assistant message content

    Returns:
        List[str]: List of ULID strings found in text
    """
    if not text or not markers.contains_marker(text):
        return []

    segments = markers.split_text_by_markers(text)
    ulids = []
    for seg in segments:
        if seg.get("type") == "marker":
            ulid = seg.get("marker")
            if ulid:
                ulids.append(ulid)
    return ulids


def _reconstruct_items_from_artifacts(
    text: str,
    ulids: List[str],
    artifacts: Dict[str, Dict[str, Any]],
    pruning_turns: int,
    current_turn_index: int,
    replayed_reasoning_refs: Optional[List[Tuple[str, str]]],
    chat_id: Optional[str],
) -> List[Dict[str, Any]]:
    """Reconstruct Responses API items from artifacts and markers.

    Args:
        text: Assistant message text with markers
        ulids: List of ULIDs extracted from text
        artifacts: Loaded artifacts by ULID
        pruning_turns: Number of recent turns to keep full outputs
        current_turn_index: Current turn number for pruning
        replayed_reasoning_refs: List to append reasoning refs for cleanup
        chat_id: Chat ID for reasoning cleanup tracking

    Returns:
        List[dict]: Reconstructed Responses API items
    """
    items: List[Dict[str, Any]] = []

    # Split text by markers to get segments
    segments = markers.split_text_by_markers(text)

    for seg in segments:
        if seg.get("type") == "text":
            # Plain text becomes a message item
            text_content = seg.get("text", "").strip()
            if text_content:
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": text_content,
                })

        elif seg.get("type") == "marker":
            # Marker references an artifact
            ulid = seg.get("marker")
            artifact = artifacts.get(ulid) if ulid else None

            if not artifact:
                continue

            artifact_type = artifact.get("type")

            # Handle reasoning artifacts
            if artifact_type == "reasoning":
                reasoning_content = artifact.get("content", [])
                reasoning_summary = artifact.get("summary", [])

                items.append({
                    "type": "reasoning",
                    "content": reasoning_content,
                    "summary": reasoning_summary,
                })

                # Track for cleanup if replay mode is next_reply
                if replayed_reasoning_refs is not None and chat_id and ulid:
                    replayed_reasoning_refs.append((chat_id, ulid))

            # Handle function call outputs
            elif artifact_type == "function_call_output":
                call_id = artifact.get("call_id")
                output = artifact.get("output", "")

                # Prune old outputs to save tokens
                should_prune = pruning_turns > 0 and current_turn_index >= pruning_turns

                if should_prune and ulid:
                    output = markers.prune_tool_output(output, ulid)

                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(output),
                })

            # Handle function calls
            elif artifact_type == "function_call":
                items.append({
                    "type": "function_call",
                    "id": artifact.get("call_id"),
                    "name": artifact.get("name"),
                    "arguments": artifact.get("arguments", "{}"),
                })

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Clean text from markers
# ─────────────────────────────────────────────────────────────────────────────

def strip_markers_from_text(text: str) -> str:
    """Remove ULID markers from text, keeping only visible content.

    Args:
        text: Text with embedded markers

    Returns:
        str: Text with markers removed
    """
    if not text or not markers.contains_marker(text):
        return text

    segments = markers.split_text_by_markers(text)
    parts = []
    for seg in segments:
        if seg.get("type") == "text":
            parts.append(seg.get("text", ""))

    return "".join(parts)
