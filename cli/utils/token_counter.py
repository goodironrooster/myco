"""Token counting utilities for GGUF CLI."""

import re
from dataclasses import dataclass
from typing import Optional

from .logging import LogConfig


@dataclass
class TokenCount:
    """Token count statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def human_readable(self) -> str:
        """Return human-readable token count."""
        parts = []
        if self.prompt_tokens > 0:
            parts.append(f"prompt: {self.prompt_tokens:,}")
        if self.completion_tokens > 0:
            parts.append(f"completion: {self.completion_tokens:,}")
        if self.total_tokens > 0:
            parts.append(f"total: {self.total_tokens:,}")
        return ", ".join(parts) if parts else "0 tokens"


class TokenCounter:
    """Tracks token usage across sessions."""

    def __init__(self):
        """Initialize token counter."""
        self.logger = LogConfig.get_logger("gguf.token_counter")
        self._session_tokens: dict[str, TokenCount] = {}
        self._current_session: Optional[str] = None
        self._total_lifetime: TokenCount = TokenCount()

    def start_session(self, session_id: str) -> None:
        """Start tracking a new session.

        Args:
            session_id: Unique session identifier
        """
        self._current_session = session_id
        if session_id not in self._session_tokens:
            self._session_tokens[session_id] = TokenCount()
        self.logger.debug(f"Started token tracking for session: {session_id}")

    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: Optional[int] = None,
    ) -> TokenCount:
        """Record token usage for current session.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (calculated if not provided)

        Returns:
            Updated TokenCount for current session
        """
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        session_id = self._current_session or "default"

        if session_id not in self._session_tokens:
            self._session_tokens[session_id] = TokenCount()

        session_count = self._session_tokens[session_id]
        session_count.prompt_tokens += prompt_tokens
        session_count.completion_tokens += completion_tokens
        session_count.total_tokens += total_tokens

        # Update lifetime total
        self._total_lifetime.prompt_tokens += prompt_tokens
        self._total_lifetime.completion_tokens += completion_tokens
        self._total_lifetime.total_tokens += total_tokens

        self.logger.debug(
            f"Recorded tokens for {session_id}: "
            f"prompt={prompt_tokens}, completion={completion_tokens}, "
            f"total={total_tokens}"
        )

        return session_count

    def get_session_tokens(self, session_id: str) -> Optional[TokenCount]:
        """Get token count for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            TokenCount or None if session not found
        """
        return self._session_tokens.get(session_id)

    def get_current_tokens(self) -> TokenCount:
        """Get token count for current session.

        Returns:
            TokenCount for current session (empty if no session)
        """
        session_id = self._current_session or "default"
        return self._session_tokens.get(session_id, TokenCount())

    def get_lifetime_tokens(self) -> TokenCount:
        """Get lifetime token count.

        Returns:
            Total TokenCount across all sessions
        """
        return self._total_lifetime

    def reset_session(self, session_id: Optional[str] = None) -> None:
        """Reset token count for a session.

        Args:
            session_id: Session to reset (current if None)
        """
        session_id = session_id or self._current_session

        if session_id and session_id in self._session_tokens:
            self._session_tokens[session_id] = TokenCount()
            self.logger.debug(f"Reset token count for session: {session_id}")

    def reset_all(self) -> None:
        """Reset all token counts."""
        self._session_tokens.clear()
        self._total_lifetime = TokenCount()
        self._current_session = None
        self.logger.info("Reset all token counts")


# Simple token estimation (when server doesn't provide counts)
def estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    This is a rough estimation based on character count.
    English text averages ~4 characters per token.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Rough estimation: 1 token ≈ 4 characters for English
    # This is approximate and varies by language/model
    char_count = len(text)
    estimated = max(1, char_count // 4)

    return estimated


def count_words(text: str) -> int:
    """Count words in text.

    Args:
        text: Text to count words in

    Returns:
        Word count
    """
    if not text or not text.strip():
        return 0

    return len(re.findall(r"\b\w+\b", text))
