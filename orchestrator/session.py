"""Session management for the POGO v2 orchestrator.

Stores conversation state in DynamoDB keyed by session_id.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

SESSIONS_TABLE = "pogo-sessions"

_dynamodb = None


def _get_table():
    """Return a cached DynamoDB Table resource."""
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    return _dynamodb.Table(SESSIONS_TABLE)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

VALID_STATES = {"initial", "awaiting_context", "review", "iterating", "accepted"}
VALID_MODELS = {"claude", "gpt", "gemini"}


@dataclass
class Session:
    """Represents a single multi-turn conversation session."""

    session_id: str
    user_id: str
    target_model: str
    state: str = "initial"
    user_intent: str = ""
    task_category: str = ""
    conversation_history: list[dict] = field(default_factory=list)
    current_draft: str = ""
    user_context: dict = field(default_factory=dict)
    clarification_answers: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        """Serialise to a plain dict suitable for DynamoDB."""
        d = asdict(self)
        # DynamoDB cannot store empty strings for non-key attributes in
        # some SDKs, so we convert empty-string fields to a sentinel.
        # We store conversation_history / user_context / scores as JSON
        # strings to avoid DynamoDB type-mapping complexity.
        d["conversation_history"] = json.dumps(d["conversation_history"])
        d["user_context"] = json.dumps(d["user_context"])
        d["clarification_answers"] = json.dumps(d["clarification_answers"])
        d["scores"] = json.dumps(d["scores"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Session:
        """Deserialise from a DynamoDB item dict."""
        d = dict(d)  # shallow copy
        d["conversation_history"] = json.loads(d.get("conversation_history", "[]"))
        d["user_context"] = json.loads(d.get("user_context", "{}"))
        d["clarification_answers"] = json.loads(d.get("clarification_answers", "{}"))
        d["scores"] = json.loads(d.get("scores", "{}"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
        })

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp."""
        self.updated_at = _now_iso()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def create_session(user_id: str, target_model: str, user_intent: str) -> Session:
    """Create a new session with a fresh UUID.

    Args:
        user_id: Caller identity (from auth context or ``"anonymous"``).
        target_model: One of ``"claude"``, ``"gpt"``, ``"gemini"``.
        user_intent: The user's original task description.

    Returns:
        A new :class:`Session` instance (not yet persisted).
    """
    now = _now_iso()
    return Session(
        session_id=str(uuid.uuid4()),
        user_id=user_id,
        target_model=target_model,
        state="initial",
        user_intent=user_intent,
        created_at=now,
        updated_at=now,
    )


def save_session(session: Session) -> None:
    """Persist *session* to DynamoDB.

    Args:
        session: The session to save.
    """
    session.touch()
    table = _get_table()
    table.put_item(Item=session.to_dict())


def load_session(session_id: str) -> Session | None:
    """Load a session from DynamoDB.

    Args:
        session_id: The UUID of the session.

    Returns:
        The :class:`Session` if found, otherwise ``None``.
    """
    table = _get_table()
    try:
        resp = table.get_item(Key={"session_id": session_id})
    except ClientError:
        return None
    item = resp.get("Item")
    if item is None:
        return None
    return Session.from_dict(item)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
