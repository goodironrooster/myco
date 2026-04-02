"""Tests for chat history persistence."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from cli.utils.chat_history import ChatHistoryManager, ChatSession


class TestChatSession:
    """Test ChatSession dataclass."""

    def test_create_session(self):
        """Test creating a new session."""
        session = ChatSession(
            id="test-123",
            model="qwen3.5-9b",
            created_at="2026-03-19T10:00:00",
            system_prompt="You are helpful.",
            title="Test Session",
        )

        assert session.id == "test-123"
        assert session.model == "qwen3.5-9b"
        assert session.system_prompt == "You are helpful."
        assert session.title == "Test Session"
        assert session.messages == []

    def test_session_to_dict(self):
        """Test converting session to dictionary."""
        session = ChatSession(
            id="test-456",
            model="llama-3",
            created_at="2026-03-19T11:00:00",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        )

        data = session.to_dict()

        assert data["id"] == "test-456"
        assert data["model"] == "llama-3"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"

    def test_session_from_dict(self):
        """Test creating session from dictionary."""
        data = {
            "id": "test-789",
            "model": "mistral-7b",
            "created_at": "2026-03-19T12:00:00",
            "messages": [{"role": "system", "content": "Be concise"}],
            "system_prompt": "Be concise",
            "title": "Dict Session",
        }

        session = ChatSession.from_dict(data)

        assert session.id == "test-789"
        assert session.model == "mistral-7b"
        assert session.system_prompt == "Be concise"
        assert len(session.messages) == 1

    def test_session_from_dict_missing_fields(self):
        """Test creating session with missing fields."""
        data = {"model": "test-model"}

        session = ChatSession.from_dict(data)

        assert session.id is not None  # Auto-generated UUID
        assert session.model == "test-model"
        assert session.messages == []
        assert session.system_prompt is None


class TestChatHistoryManager:
    """Test ChatHistoryManager functionality."""

    @pytest.fixture
    def temp_history_dir(self):
        """Create temporary directory for history files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def history_manager(self, temp_history_dir):
        """Create ChatHistoryManager with temp directory."""
        return ChatHistoryManager(history_dir=temp_history_dir)

    def test_create_session(self, history_manager):
        """Test creating a new session."""
        session = history_manager.create_session(
            model="qwen3.5-9b",
            system_prompt="Test prompt",
            title="My Session",
        )

        assert session.model == "qwen3.5-9b"
        assert session.system_prompt == "Test prompt"
        assert session.title == "My Session"
        assert session.id is not None
        assert history_manager.get_current_session() == session

    def test_save_session(self, history_manager, temp_history_dir):
        """Test saving a session to disk."""
        session = history_manager.create_session(model="test-model")
        session.messages.append({"role": "user", "content": "Hello"})

        result = history_manager.save_session(session)

        assert result is True
        session_file = temp_history_dir / f"{session.id}.json"
        assert session_file.exists()

        # Verify content
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["model"] == "test-model"
        assert len(data["messages"]) == 1

    def test_load_session(self, history_manager, temp_history_dir):
        """Test loading a session from disk."""
        # Create and save session
        session = history_manager.create_session(model="load-test")
        session.messages.append({"role": "assistant", "content": "Response"})
        history_manager.save_session(session)

        # Clear current session and reload
        history_manager.set_current_session(None)
        loaded = history_manager.load_session(session.id)

        assert loaded is not None
        assert loaded.model == "load-test"
        assert len(loaded.messages) == 1

    def test_load_session_not_found(self, history_manager):
        """Test loading non-existent session."""
        result = history_manager.load_session("nonexistent-id")
        assert result is None

    def test_list_sessions(self, history_manager):
        """Test listing all sessions."""
        # Create multiple sessions
        for i in range(3):
            session = history_manager.create_session(model=f"model-{i}")
            session.title = f"Session {i}"
            history_manager.save_session(session)

        sessions = history_manager.list_sessions()

        assert len(sessions) == 3
        # Should be sorted newest first
        assert sessions[0].title == "Session 2"

    def test_list_sessions_empty(self, history_manager):
        """Test listing when no sessions exist."""
        sessions = history_manager.list_sessions()
        assert sessions == []

    def test_delete_session(self, history_manager, temp_history_dir):
        """Test deleting a session."""
        session = history_manager.create_session(model="delete-me")
        history_manager.save_session(session)

        result = history_manager.delete_session(session.id)

        assert result is True
        session_file = temp_history_dir / f"{session.id}.json"
        assert not session_file.exists()
        assert history_manager.get_current_session() is None

    def test_delete_session_not_found(self, history_manager):
        """Test deleting non-existent session."""
        result = history_manager.delete_session("nonexistent")
        assert result is False

    def test_add_message(self, history_manager):
        """Test adding messages to session."""
        session = history_manager.create_session(model="msg-test")

        result = history_manager.add_message("user", "Hello", session)

        assert result is True
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"

    def test_add_message_no_session(self, history_manager):
        """Test adding message without session."""
        history_manager.set_current_session(None)
        result = history_manager.add_message("user", "Hello")
        assert result is False

    def test_clear_session(self, history_manager):
        """Test clearing session messages."""
        session = history_manager.create_session(
            model="clear-test",
            system_prompt="Keep this",
        )
        session.messages.append({"role": "user", "content": "Message 1"})
        session.messages.append({"role": "assistant", "content": "Message 2"})

        history_manager.clear_session(session)

        # Should only keep system prompt
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"

    def test_clear_session_no_system_prompt(self, history_manager):
        """Test clearing session without system prompt."""
        session = history_manager.create_session(model="clear-test")
        session.messages.append({"role": "user", "content": "Message"})

        history_manager.clear_session(session)

        assert len(session.messages) == 0

    def test_session_file_sanitization(self, history_manager):
        """Test session ID sanitization for file paths."""
        # Create session with potentially unsafe characters
        session = ChatSession(
            id="test/../evil-id",
            model="test",
            created_at=datetime.now().isoformat(),
        )
        history_manager.set_current_session(session)
        history_manager.save_session(session)

        # Should create file with sanitized name (removes / but keeps alphanumeric)
        session_file = history_manager._get_session_file(session.id)
        # Path traversal should be prevented (no directory separators in filename)
        assert "/" not in session_file.name
        assert "\\" not in session_file.name
        # File should be in the history directory
        assert session_file.parent == history_manager.history_dir

    def test_generate_title(self, history_manager):
        """Test automatic title generation."""
        session = history_manager.create_session(model="long-model-name-gguf")

        assert "long-model-name" in session.title
        assert datetime.now().strftime("%Y-%m-%d") in session.title


class TestChatHistoryIntegration:
    """Integration tests for chat history."""

    @pytest.fixture
    def temp_history_dir(self):
        """Create temporary directory for history files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_full_session_lifecycle(self, temp_history_dir):
        """Test complete session lifecycle."""
        manager = ChatHistoryManager(history_dir=temp_history_dir)

        # Create
        session = manager.create_session(
            model="qwen3.5",
            system_prompt="You are helpful",
            title="Integration Test",
        )
        session_id = session.id

        # Add messages
        manager.add_message("user", "Hello", session)
        manager.add_message("assistant", "Hi! How can I help?", session)
        manager.save_session(session)

        # Load in new manager instance
        manager2 = ChatHistoryManager(history_dir=temp_history_dir)
        loaded = manager2.load_session(session_id)

        assert loaded is not None
        assert loaded.model == "qwen3.5"
        assert len(loaded.messages) == 2
        assert loaded.system_prompt == "You are helpful"

        # Delete
        manager2.delete_session(session_id)
        assert manager2.load_session(session_id) is None
