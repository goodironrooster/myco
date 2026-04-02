"""Chat history persistence for GGUF CLI."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .logging import LogConfig


@dataclass
class ChatSession:
    """Represents a chat session with metadata."""

    id: str
    model: str
    created_at: str
    messages: list[dict] = field(default_factory=list)
    system_prompt: Optional[str] = None
    title: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert session to dictionary for serialization."""
        return {
            "id": self.id,
            "model": self.model,
            "created_at": self.created_at,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSession":
        """Create session from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            model=data.get("model", "unknown"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            messages=data.get("messages", []),
            system_prompt=data.get("system_prompt"),
            title=data.get("title"),
        )


class ChatHistoryManager:
    """Manages chat history persistence."""

    def __init__(self, history_dir: Optional[Path] = None):
        """Initialize chat history manager.

        Args:
            history_dir: Directory to store chat history.
                        Defaults to ~/.gguf-cli/history/
        """
        self.logger = LogConfig.get_logger("gguf.chat_history")

        if history_dir is None:
            history_dir = Path.home() / ".gguf-cli" / "history"

        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self._current_session: Optional[ChatSession] = None

    def create_session(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        title: Optional[str] = None,
    ) -> ChatSession:
        """Create a new chat session.

        Args:
            model: Model name/ID
            system_prompt: Optional system prompt
            title: Optional session title

        Returns:
            New ChatSession object
        """
        session = ChatSession(
            id=str(uuid.uuid4()),
            model=model,
            created_at=datetime.now().isoformat(),
            system_prompt=system_prompt,
            title=title or self._generate_title(model),
        )

        self._current_session = session
        self.logger.info(f"Created new chat session: {session.id}")
        return session

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session from disk.

        Args:
            session_id: Session UUID

        Returns:
            ChatSession if found, None otherwise
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            self.logger.warning(f"Session not found: {session_id}")
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            session = ChatSession.from_dict(data)
            self._current_session = session
            self.logger.info(f"Loaded session: {session_id}")
            return session
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def save_session(self, session: Optional[ChatSession] = None) -> bool:
        """Save current session to disk.

        Args:
            session: Session to save (uses current if None)

        Returns:
            True if saved successfully
        """
        session = session or self._current_session

        if session is None:
            self.logger.warning("No session to save")
            return False

        session_file = self._get_session_file(session.id)

        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2)
            self.logger.debug(f"Saved session: {session.id}")
            return True
        except OSError as e:
            self.logger.error(f"Failed to save session: {e}")
            return False

    def list_sessions(self) -> list[ChatSession]:
        """List all saved sessions.

        Returns:
            List of ChatSession objects, newest first
        """
        sessions = []

        for session_file in self.history_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = ChatSession.from_dict(data)
                sessions.append(session)
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Skipping corrupted session {session_file}: {e}")

        # Sort by created_at, newest first
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session UUID

        Returns:
            True if deleted successfully
        """
        session_file = self._get_session_file(session_id)

        if not session_file.exists():
            return False

        try:
            session_file.unlink()
            self.logger.info(f"Deleted session: {session_id}")

            if self._current_session and self._current_session.id == session_id:
                self._current_session = None

            return True
        except OSError as e:
            self.logger.error(f"Failed to delete session: {e}")
            return False

    def add_message(
        self,
        role: str,
        content: str,
        session: Optional[ChatSession] = None,
    ) -> bool:
        """Add a message to the current session.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            session: Session to add to (uses current if None)

        Returns:
            True if added successfully
        """
        session = session or self._current_session

        if session is None:
            self.logger.warning("No session to add message to")
            return False

        session.messages.append({"role": role, "content": content})
        return True

    def clear_session(self, session: Optional[ChatSession] = None) -> None:
        """Clear messages in current session (keeps metadata).

        Args:
            session: Session to clear (uses current if None)
        """
        session = session or self._current_session

        if session is None:
            return

        # Keep system prompt if exists
        system_prompt = session.system_prompt
        session.messages = []
        if system_prompt:
            session.messages.insert(0, {"role": "system", "content": system_prompt})

    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current active session."""
        return self._current_session

    def set_current_session(self, session: Optional[ChatSession]) -> None:
        """Set the current active session."""
        self._current_session = session

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session."""
        # Sanitize session_id to prevent path traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return self.history_dir / f"{safe_id}.json"

    def _generate_title(self, model: str) -> str:
        """Generate a default title for a session."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        model_short = model.split("/")[-1].replace(".gguf", "")[:20]
        return f"{model_short} - {timestamp}"
