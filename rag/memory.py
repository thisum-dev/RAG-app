"""
rag/memory.py - Simple in-memory conversation history manager.

Responsibility:
  - Store and retrieve per-session chat history.
  - Provide helper to append a new turn (user + assistant).
  - Keep memory scoped to the running process (no DB required).

Note:
  This implementation stores history in a plain Python dict.
  It is intentionally simple — suitable for development and single-process
  deployments. For multi-process or persistent memory, replace with Redis
  or a database backend.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# ─── In-memory store: { session_id: [ {"role": ..., "content": ...}, ... ] } ─
_store: dict[str, list[dict]] = defaultdict(list)


def get_history(session_id: str) -> list[dict]:
    """
    Return the full chat history for a session.

    Args:
        session_id: Unique identifier for the user's session (e.g. UUID).

    Returns:
        List of message dicts: [{"role": "user"|"assistant", "content": "..."}, ...]
    """
    history = _store[session_id]
    logger.debug(f"Session {session_id}: {len(history)} turns retrieved.")
    return history


def add_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """
    Append a completed conversation turn to the session history.

    Args:
        session_id:        Session identifier.
        user_message:      The user's question for this turn.
        assistant_message: The assistant's answer for this turn.
    """
    _store[session_id].append({"role": "user",      "content": user_message})
    _store[session_id].append({"role": "assistant", "content": assistant_message})
    logger.debug(f"Session {session_id}: turn added, total turns = {len(_store[session_id]) // 2}.")


def clear_history(session_id: str) -> None:
    """
    Clear the history for a specific session.

    Args:
        session_id: Session identifier.
    """
    _store[session_id] = []
    logger.info(f"Session {session_id}: history cleared.")
