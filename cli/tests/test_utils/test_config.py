"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from cli.utils.config import Config, DEFAULT_CONFIG


class TestConfig:
    """Test configuration management."""

    def test_default_config(self, tmp_path):
        """Test loading default configuration."""
        config_file = tmp_path / "config.yaml"
        config = Config(config_file)

        assert config.get("server", "host") == "127.0.0.1"
        assert config.get("server", "port") == 1234
        assert config.get("chat", "temperature") == 0.7

    def test_load_existing_config(self, tmp_path):
        """Test loading configuration from file."""
        config_file = tmp_path / "config.yaml"

        custom_config = {
            "server": {
                "port": 8080,
                "host": "localhost",
            },
            "chat": {
                "temperature": 0.9,
            },
        }

        with open(config_file, "w") as f:
            yaml.safe_dump(custom_config, f)

        config = Config(config_file)

        assert config.get("server", "port") == 8080
        assert config.get("server", "host") == "localhost"
        assert config.get("chat", "temperature") == 0.9
        # Default value not overridden
        assert config.get("server", "context_length") == 8192

    def test_get_nested_value(self, tmp_path):
        """Test getting nested configuration values."""
        # Use fresh temp file, not shared config
        config_file = tmp_path / "test_config.yaml"
        config = Config(config_file)

        # Valid nested access - should use defaults
        value = config.get("server", "host")
        assert value == "127.0.0.1"

        # Missing key returns default
        value = config.get("nonexistent", "key", default="fallback")
        assert value == "fallback"

    def test_set_nested_value(self, tmp_path):
        """Test setting nested configuration values."""
        config_file = tmp_path / "config.yaml"
        config = Config(config_file)

        config.set(9999, "server", "port")
        assert config.get("server", "port") == 9999

        config.set("custom_value", "custom", "section", "key")
        assert config.get("custom", "section", "key") == "custom_value"

    def test_save_config(self, tmp_path):
        """Test saving configuration to file."""
        config_file = tmp_path / "config.yaml"
        config = Config(config_file)

        config.set(8888, "server", "port")
        config.save()

        assert config_file.exists()

        with open(config_file, "r") as f:
            saved = yaml.safe_load(f)

        assert saved["server"]["port"] == 8888

    def test_config_path_property(self, tmp_path):
        """Test config path property."""
        config_file = tmp_path / "config.yaml"
        config = Config(config_file)

        assert config.config_path == config_file

    def test_deep_update_preserves_structure(self, tmp_path):
        """Test that deep update preserves nested structure."""
        # Use unique file per test
        config_file = tmp_path / "partial_config.yaml"

        # Partial config - only port
        partial = {
            "server": {
                "port": 5000,
            },
        }

        with open(config_file, "w") as f:
            yaml.safe_dump(partial, f)

        config = Config(config_file)

        # Should have custom port but default host
        assert config.get("server", "port") == 5000
        assert config.get("server", "host") == "127.0.0.1"

    def test_invalid_config_file(self, tmp_path):
        """Test handling of invalid config file."""
        config_file = tmp_path / "invalid_config.yaml"

        # Write invalid YAML
        with open(config_file, "w") as f:
            f.write("invalid: yaml: content: [")

        # Should fall back to defaults without crashing
        config = Config(config_file)
        # Defaults should be used when config is invalid
        assert config.get("server", "port") == 1234
        assert config.get("server", "host") == "127.0.0.1"

    def test_empty_config_file(self, tmp_path):
        """Test handling of empty config file."""
        config_file = tmp_path / "empty_config.yaml"
        config_file.write_text("")

        config = Config(config_file)
        # Empty file should use defaults
        assert config.get("server", "port") == 1234
        assert config.get("server", "host") == "127.0.0.1"
