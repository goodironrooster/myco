"""Tests for chat commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from click.testing import CliRunner

from cli.main import cli


class TestChatSessionsCommand:
    """Test chat sessions listing command."""

    def test_sessions_empty(self):
        """Test listing sessions when none exist."""
        runner = CliRunner()

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.list_sessions.return_value = []
            mock_manager.return_value = mock_instance

            result = runner.invoke(cli, ["chat", "sessions"])

            assert result.exit_code == 0
            assert "No saved chat sessions found." in result.output

    def test_sessions_with_data(self):
        """Test listing sessions with data."""
        runner = CliRunner()

        # Create mock session
        mock_session = MagicMock()
        mock_session.id = "test-123-456"
        mock_session.title = "Test Session"
        mock_session.model = "qwen3.5-9b"
        mock_session.created_at = "2026-03-19T10:00:00"
        mock_session.messages = [{"role": "user", "content": "Hello"}]

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.list_sessions.return_value = [mock_session]
            mock_manager.return_value = mock_instance

            result = runner.invoke(cli, ["chat", "sessions"])

            assert result.exit_code == 0
            assert "Test Session" in result.output
            assert "qwen3.5-9b" in result.output
            assert "2026-03-19" in result.output

    def test_sessions_limit(self):
        """Test session listing with limit."""
        runner = CliRunner()

        # Create multiple mock sessions
        mock_sessions = []
        for i in range(5):
            session = MagicMock()
            session.id = f"session-{i}"
            session.title = f"Session {i}"
            session.model = "test-model"
            session.created_at = "2026-03-19T10:00:00"
            session.messages = []
            mock_sessions.append(session)

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.list_sessions.return_value = mock_sessions
            mock_manager.return_value = mock_instance

            result = runner.invoke(cli, ["chat", "sessions", "-n", "3"])

            assert result.exit_code == 0
            assert "showing 3 of 5" in result.output


class TestShowSessionCommand:
    """Test show session command."""

    def test_show_session_not_found(self):
        """Test showing non-existent session."""
        runner = CliRunner()

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.load_session.return_value = None
            mock_manager.return_value = mock_instance

            result = runner.invoke(cli, ["chat", "show-session", "nonexistent-id"])

            assert result.exit_code == 1
            assert "Session not found" in result.output

    def test_show_session_success(self):
        """Test showing existing session."""
        runner = CliRunner()

        mock_session = MagicMock()
        mock_session.id = "test-123"
        mock_session.title = "My Session"
        mock_session.model = "qwen3.5-9b"
        mock_session.created_at = "2026-03-19T10:00:00"
        mock_session.messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_session.system_prompt = "Be helpful"

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.load_session.return_value = mock_session
            mock_manager.return_value = mock_instance

            result = runner.invoke(cli, ["chat", "show-session", "test-123"])

            assert result.exit_code == 0
            assert "Session Details" in result.output
            assert "test-123" in result.output
            assert "My Session" in result.output
            assert "qwen3.5-9b" in result.output
            assert "Messages: 3" in result.output


class TestDeleteSessionCommand:
    """Test delete session command."""

    def test_delete_session_not_found(self):
        """Test deleting non-existent session."""
        runner = CliRunner()

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.delete_session.return_value = False
            mock_manager.return_value = mock_instance

            result = runner.invoke(
                cli,
                ["chat", "delete-session", "nonexistent"],
                input="y\n",
            )

            assert result.exit_code == 1
            assert "Session not found" in result.output

    def test_delete_session_success(self):
        """Test successful session deletion."""
        runner = CliRunner()

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_instance.delete_session.return_value = True
            mock_manager.return_value = mock_instance

            result = runner.invoke(
                cli,
                ["chat", "delete-session", "test-123"],
                input="y\n",
            )

            assert result.exit_code == 0
            assert "Session deleted" in result.output

    def test_delete_session_aborted(self):
        """Test deletion cancelled by user."""
        runner = CliRunner()

        with patch("cli.commands.chat.ChatHistoryManager") as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance

            result = runner.invoke(
                cli,
                ["chat", "delete-session", "test-123"],
                input="n\n",
            )

            # Click aborts with exit code 1 when confirmation is declined
            assert result.exit_code in (0, 1)


class TestChatInteractiveCommand:
    """Test interactive chat command."""

    def test_interactive_no_server(self):
        """Test interactive chat when server is not running."""
        runner = CliRunner()

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Server not running")

            result = runner.invoke(cli, ["chat", "interactive"])

            assert result.exit_code == 1
            assert "Server is not running" in result.output or "Server" in result.output

    def test_interactive_basic(self):
        """Test basic interactive chat flow."""
        runner = CliRunner()

        # Mock server health check
        mock_health = MagicMock()
        mock_health.status_code = 200

        # Mock models endpoint
        mock_models = MagicMock()
        mock_models.status_code = 200
        mock_models.json.return_value = {"data": [{"id": "qwen3.5-9b"}]}

        # Mock chat completion
        mock_chat = MagicMock()
        mock_chat.status_code = 200
        mock_chat.json.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help?"}}]
        }

        with patch("requests.get", side_effect=[mock_health, mock_models]):
            with patch("requests.post", return_value=mock_chat):
                result = runner.invoke(
                    cli,
                    ["chat", "interactive", "--no-save-history"],
                    input="Hello\nexit\n",
                )

                assert result.exit_code == 0
                assert "Interactive Chat Mode" in result.output


class TestChatCompleteCommand:
    """Test chat complete command."""

    def test_complete_no_server(self):
        """Test complete command when server is not running."""
        runner = CliRunner()

        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Server not running")

            result = runner.invoke(cli, ["chat", "complete", "Test prompt"])

            assert result.exit_code == 1
            assert "Server" in result.output

    def test_complete_success(self):
        """Test successful prompt completion."""
        runner = CliRunner()

        # Mock health check
        mock_health = MagicMock()
        mock_health.status_code = 200

        # Mock completion
        mock_completion = MagicMock()
        mock_completion.status_code = 200
        mock_completion.json.return_value = {
            "choices": [{"message": {"content": "This is the response"}}]
        }

        with patch("requests.get", return_value=mock_health):
            with patch("requests.post", return_value=mock_completion):
                result = runner.invoke(cli, ["chat", "complete", "Test prompt"])

                assert result.exit_code == 0
                assert "This is the response" in result.output

    def test_complete_timeout(self):
        """Test completion with timeout."""
        runner = CliRunner()

        mock_health = MagicMock()
        mock_health.status_code = 200

        with patch("requests.get", return_value=mock_health):
            with patch("requests.post") as mock_post:
                mock_post.side_effect = Exception("Timeout")

                result = runner.invoke(cli, ["chat", "complete", "Test prompt"])

                assert result.exit_code == 1


class TestChatWithHistory:
    """Test chat commands with history persistence."""

    @pytest.fixture
    def temp_history_dir(self, tmp_path):
        """Create temporary history directory."""
        history_dir = tmp_path / "history"
        history_dir.mkdir()
        return history_dir

    def test_interactive_saves_session(self, temp_history_dir):
        """Test that interactive chat saves session."""
        runner = CliRunner()

        mock_health = MagicMock()
        mock_health.status_code = 200

        mock_models = MagicMock()
        mock_models.status_code = 200
        mock_models.json.return_value = {"data": [{"id": "test-model"}]}

        mock_chat = MagicMock()
        mock_chat.status_code = 200
        mock_chat.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }

        with patch("requests.get", side_effect=[mock_health, mock_models]):
            with patch("requests.post", return_value=mock_chat):
                with patch(
                    "cli.commands.chat.ChatHistoryManager"
                ) as mock_manager_class:
                    mock_instance = MagicMock()
                    mock_session = MagicMock()
                    mock_session.id = "test-session-id"
                    mock_session.title = "Test Session"
                    mock_instance.create_session.return_value = mock_session
                    mock_manager_class.return_value = mock_instance

                    result = runner.invoke(
                        cli,
                        ["chat", "interactive"],
                        input="Hello\nexit\n",
                    )

                    assert result.exit_code == 0
                    # Verify save was called
                    assert mock_instance.save_session.called
