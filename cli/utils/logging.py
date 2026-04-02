"""Logging configuration for GGUF CLI."""

import logging
import sys
from pathlib import Path
from typing import Optional


class LogConfig:
    """Configures and manages logging for the CLI."""

    _initialized = False
    _logger: Optional[logging.Logger] = None

    @classmethod
    def setup(
        cls,
        level: str = "INFO",
        log_file: Optional[Path] = None,
        verbose: bool = False,
    ) -> logging.Logger:
        """Set up logging configuration.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
            verbose: If True, show more detailed output

        Returns:
            Configured logger instance
        """
        if cls._initialized:
            return cls._logger or logging.getLogger("gguf")

        # Determine log level
        if verbose:
            level = "DEBUG"

        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Create logger
        logger = logging.getLogger("gguf")
        logger.setLevel(numeric_level)
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            "%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        cls._initialized = True
        cls._logger = logger

        return logger

    @classmethod
    def get_logger(cls, name: str = "gguf") -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name (usually module name)

        Returns:
            Logger instance
        """
        if cls._logger:
            return cls._logger
        return logging.getLogger(name)
