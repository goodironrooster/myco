"""Configuration management for GGUF CLI."""

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml


DEFAULT_CONFIG = {
    "server": {
        "host": "127.0.0.1",
        "port": 1234,
        "context_length": 131072,  # 128K context for large conversations (Qwen3.5-9B supports 256K max)
        "threads": None,  # Auto-detect
    },
    "model": {
        "default": None,  # Auto-detect first GGUF file
    },
    "chat": {
        "system_prompt": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_tokens": 8192,  # Increased for very long responses
    },
}


class Config:
    """Manages CLI configuration with YAML persistence."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_path: Path to config file. Defaults to ~/.gguf-cli/config.yaml
        """
        if config_path is None:
            config_dir = Path.home() / ".gguf-cli"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.yaml"

        self._config_path = config_path
        self._config: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load configuration from file or return defaults."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                # Merge with defaults (use deep copy to avoid mutation)
                config = copy.deepcopy(DEFAULT_CONFIG)
                self._deep_update(config, loaded)
                return config
            except (yaml.YAMLError, OSError) as e:
                # Log warning and return defaults (logging not available yet)
                print(f"Warning: Could not load config: {e}")
                return copy.deepcopy(DEFAULT_CONFIG)
        return copy.deepcopy(DEFAULT_CONFIG)

    def _deep_update(self, base: dict, update: dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def save(self) -> None:
        """Save current configuration to file."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value.

        Args:
            keys: Path to config value (e.g., 'server', 'port')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, value: Any, *keys: str) -> None:
        """Set nested configuration value.

        Args:
            value: Value to set
            keys: Path to config value (e.g., 'server', 'port')
        """
        config = self._config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

    @property
    def config_path(self) -> Path:
        """Return the configuration file path."""
        return self._config_path
