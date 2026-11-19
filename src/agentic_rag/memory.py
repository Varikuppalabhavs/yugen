
"""Persistent conversation history for the Agentic RAG assistant."""

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERSATIONS_DIR = PROJECT_ROOT / "data" / "conversations"


def _history_path(session_id: str) -> Path:
    """Return the history file belonging to one valid session."""

    if not session_id or not re.fullmatch(
        r"[a-zA-Z0-9_-]{20,100}", session_id
    ):
        raise ValueError("Invalid session ID.")

    return CONVERSATIONS_DIR / f"{session_id}.json"


def load_chat_history(session_id: str) -> list[dict[str, Any]]:
    """Load a session's saved conversation."""

    path = _history_path(session_id)

    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            messages = json.load(file)

        if not isinstance(messages, list):
            return []

        return [
            message
            for message in messages
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ]

    except (OSError, json.JSONDecodeError):
        return []


def save_chat_history(
    session_id: str,
    messages: list[dict[str, Any]],
) -> None:
    """Persist conversation history for a session."""

    path = _history_path(session_id)
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(messages, file, ensure_ascii=False, indent=2)

    temporary_path.replace(path)


def add_message(
    messages: list[dict[str, Any]],
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """Append a message to the in-memory conversation."""

    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role.")

    messages.append({
        "role": role,
        "content": content,
        "sources": sources or [],
    })


def clear_chat_history(session_id: str) -> None:
    """Delete the saved conversation without deleting documents."""

    path = _history_path(session_id)

    if path.exists():
        path.unlink()
