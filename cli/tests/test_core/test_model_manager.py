"""Tests for model manager module."""

import tempfile
from pathlib import Path

import pytest

from cli.core.model_manager import ModelManager, ModelInfo


class TestModelManager:
    """Test model management functionality."""

    def test_list_models_empty(self, tmp_path):
        """Test listing models in empty directory."""
        manager = ModelManager(tmp_path)
        models = manager.list_models()

        assert len(models) == 0

    def test_list_models_single(self, tmp_path):
        """Test listing models with one GGUF file."""
        # Create a fake GGUF file (minimal valid header)
        gguf_file = tmp_path / "test-model.gguf"
        self._create_minimal_gguf(gguf_file)

        manager = ModelManager(tmp_path)
        models = manager.list_models()

        assert len(models) == 1
        assert models[0].name == "test-model.gguf"
        assert models[0].path == gguf_file.absolute()

    def test_list_models_multiple(self, tmp_path):
        """Test listing multiple GGUF files."""
        # Create multiple fake GGUF files
        for name in ["model1.gguf", "model2.gguf", "model3.gguf"]:
            self._create_minimal_gguf(tmp_path / name)

        manager = ModelManager(tmp_path)
        models = manager.list_models()

        assert len(models) == 3
        # Should be sorted by name
        assert models[0].name == "model1.gguf"
        assert models[1].name == "model2.gguf"
        assert models[2].name == "model3.gguf"

    def test_get_model_info(self, tmp_path):
        """Test getting model information."""
        gguf_file = tmp_path / "Qwen3.5-9B-Q4_0.gguf"
        self._create_minimal_gguf(gguf_file)

        manager = ModelManager(tmp_path)
        info = manager.get_model_info(gguf_file)

        assert info.name == "Qwen3.5-9B-Q4_0.gguf"
        assert info.size_bytes > 0
        assert info.sha256 is not None
        assert len(info.sha256) == 16  # Short hash

    def test_get_model_info_quantization_extraction(self, tmp_path):
        """Test quantization extraction from filename."""
        test_cases = [
            ("model-Q4_0.gguf", "Q4_0"),
            ("model-Q5_K.gguf", "Q5_K"),
            ("model-Q8_0.gguf", "Q8_0"),
            ("model-fp16.gguf", None),
        ]

        for filename, expected_quant in test_cases:
            gguf_file = tmp_path / filename
            self._create_minimal_gguf(gguf_file)

            manager = ModelManager(tmp_path)
            info = manager.get_model_info(gguf_file)

            assert info.quantization == expected_quant

    def test_get_model_info_not_found(self, tmp_path):
        """Test getting info for non-existent file."""
        manager = ModelManager(tmp_path)
        non_existent = tmp_path / "nonexistent.gguf"

        with pytest.raises(FileNotFoundError):
            manager.get_model_info(non_existent)

    def test_validate_model_valid(self, tmp_path):
        """Test validating a valid model."""
        gguf_file = tmp_path / "valid.gguf"
        self._create_minimal_gguf(gguf_file)

        manager = ModelManager(tmp_path)
        is_valid, message = manager.validate_model(gguf_file)

        # Minimal GGUF without architecture info is still valid as a file
        # The validation checks file integrity, not metadata completeness
        assert is_valid is True
        assert "valid.gguf" in message or "valid" in message.lower()

    def test_validate_model_not_found(self, tmp_path):
        """Test validating non-existent model."""
        manager = ModelManager(tmp_path)
        non_existent = tmp_path / "nonexistent.gguf"

        is_valid, message = manager.validate_model(non_existent)

        assert is_valid is False
        assert "not found" in message.lower()

    def test_validate_model_invalid(self, tmp_path):
        """Test validating invalid GGUF file."""
        # Create a file with invalid GGUF magic
        invalid_file = tmp_path / "invalid.gguf"
        with open(invalid_file, "wb") as f:
            f.write(b"NOT_GGUF")

        manager = ModelManager(tmp_path)
        is_valid, message = manager.validate_model(invalid_file)

        assert is_valid is False

    def test_model_info_size_human(self, tmp_path):
        """Test human-readable size formatting."""
        gguf_file = tmp_path / "test.gguf"
        self._create_minimal_gguf(gguf_file, size=1024 * 1024)  # 1 MB

        manager = ModelManager(tmp_path)
        info = manager.get_model_info(gguf_file)

        assert "MB" in info.size_human

    def test_list_models_nonexistent_directory(self):
        """Test listing models in non-existent directory."""
        manager = ModelManager(Path("/nonexistent/path"))
        models = manager.list_models()

        assert len(models) == 0

    def _create_minimal_gguf(self, path: Path, size: int = 1024):
        """Create a minimal valid GGUF file.

        GGUF format:
        - Magic: "GGUF" (4 bytes)
        - Version: uint32 (4 bytes)
        - Tensor count: uint64 (8 bytes)
        - KV count: uint64 (8 bytes)
        """
        import struct

        with open(path, "wb") as f:
            # Magic number
            f.write(b"GGUF")
            # Version (3)
            f.write(struct.pack("<I", 3))
            # Tensor count (0)
            f.write(struct.pack("<Q", 0))
            # KV count (0)
            f.write(struct.pack("<Q", 0))

            # Pad to desired size
            remaining = size - f.tell()
            if remaining > 0:
                f.write(b"\x00" * remaining)
