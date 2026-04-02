"""Tests for CLI commands."""

from click.testing import CliRunner

from cli.main import cli


class TestCliBase:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test CLI help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "GGUF CLI" in result.output
        assert "model" in result.output
        assert "server" in result.output
        assert "chat" in result.output

    def test_cli_version(self):
        """Test CLI version output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_cli_verbose_flag(self):
        """Test verbose flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--verbose", "--help"])

        assert result.exit_code == 0


class TestModelCommands:
    """Test model management commands."""

    def test_model_list(self):
        """Test model list command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "list"])

        assert result.exit_code == 0
        # Should not crash even if no models found

    def test_model_info_missing(self):
        """Test model info with missing file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "info", "nonexistent.gguf"])

        # Should report error but not crash
        assert result.exit_code != 0 or "not found" in result.output.lower()

    def test_model_validate_missing(self):
        """Test model validate with missing file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["model", "validate", "nonexistent.gguf"])

        assert result.exit_code != 0


class TestServerCommands:
    """Test server control commands."""

    def test_server_status(self):
        """Test server status command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["server", "status"])

        assert result.exit_code == 0
        assert "running" in result.output.lower() or "not running" in result.output.lower()

    def test_server_status_json(self):
        """Test server status with JSON output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["server", "status", "--json"])

        assert result.exit_code == 0
        assert "{" in result.output  # Basic JSON check

    def test_server_stop_no_server(self):
        """Test stopping when no server is running."""
        runner = CliRunner()
        result = runner.invoke(cli, ["server", "stop"])

        # Should handle gracefully
        assert result.exit_code == 0 or "not running" in result.output.lower()


class TestChatCommands:
    """Test chat commands."""

    def test_chat_complete_no_server(self):
        """Test chat complete when server is not running."""
        runner = CliRunner()
        result = runner.invoke(cli, ["chat", "complete", "Hello"])

        # Should report server not running
        assert result.exit_code != 0 or "not running" in result.output.lower()


class TestInitCommand:
    """Test init command."""

    def test_init_creates_config(self):
        """Test init command creates config file."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])

            assert result.exit_code == 0
            assert "Configuration saved" in result.output
