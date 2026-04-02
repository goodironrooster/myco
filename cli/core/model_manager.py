"""Model management core logic."""

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelInfo:
    """Information about a GGUF model file."""

    path: Path
    name: str
    size_bytes: int
    sha256: str
    architecture: Optional[str] = None
    parameter_count: Optional[str] = None
    quantization: Optional[str] = None

    @property
    def size_human(self) -> str:
        """Return human-readable file size."""
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"


class ModelManager:
    """Manages GGUF model files."""

    GGUF_MAGIC = b"GGUF"

    def __init__(self, models_dir: Path):
        """Initialize model manager.

        Args:
            models_dir: Directory to search for GGUF models
        """
        self.models_dir = models_dir

    def list_models(self) -> list[ModelInfo]:
        """List all GGUF model files in the models directory.

        Returns:
            List of ModelInfo objects
        """
        models = []
        if not self.models_dir.exists():
            return models

        for path in self.models_dir.glob("*.gguf"):
            try:
                info = self.get_model_info(path)
                models.append(info)
            except (OSError, ValueError) as e:
                # Skip invalid files
                continue

        return sorted(models, key=lambda m: m.name)

    def get_model_info(self, path: Path) -> ModelInfo:
        """Get detailed information about a model file.

        Args:
            path: Path to the GGUF file

        Returns:
            ModelInfo object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid GGUF file
        """
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        # Calculate file hash
        sha256 = self._calculate_hash(path)

        # Get file size
        size_bytes = path.stat().st_size

        # Try to parse GGUF header
        architecture = None
        parameter_count = None
        quantization = None

        try:
            with open(path, "rb") as f:
                # Read magic number
                magic = f.read(4)
                if magic != self.GGUF_MAGIC:
                    raise ValueError(f"Not a valid GGUF file: {path}")

                # Read version (uint32)
                version = struct.unpack("<I", f.read(4))[0]

                # Read tensor count (uint64)
                tensor_count = struct.unpack("<Q", f.read(8))[0]

                # Read KV count (uint64)
                kv_count = struct.unpack("<Q", f.read(8))[0]

                # Parse key-value pairs (simplified - just get architecture)
                for _ in range(kv_count):
                    key_type = struct.unpack("<I", f.read(4))[0]
                    key_len = struct.unpack("<I", f.read(4))[0]
                    key = f.read(key_len).decode("utf-8", errors="ignore")

                    # Read value type
                    value_type = struct.unpack("<I", f.read(4))[0]

                    if key == "general.architecture":
                        # String value
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        architecture = f.read(str_len).decode("utf-8", errors="ignore")
                    else:
                        # Skip value
                        self._skip_value(f, value_type)

        except (struct.error, UnicodeDecodeError) as e:
            # File might be corrupted or in use
            pass

        # Extract quantization from filename
        quantization = self._extract_quantization(path.name)

        return ModelInfo(
            path=path.absolute(),
            name=path.name,
            size_bytes=size_bytes,
            sha256=sha256,
            architecture=architecture,
            parameter_count=parameter_count,
            quantization=quantization,
        )

    def validate_model(self, path: Path) -> tuple[bool, str]:
        """Validate a model file.

        Args:
            path: Path to the GGUF file

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            info = self.get_model_info(path)

            # Check if file has valid GGUF magic (already verified in get_model_info)
            # Architecture being None means metadata couldn't be read, but file may still be valid
            if info.architecture is None:
                # File is valid GGUF but we couldn't read metadata
                return True, f"Valid GGUF file: {info.name} ({info.size_human}) [metadata incomplete]"

            return True, f"Valid GGUF model: {info.name} ({info.size_human})"

        except FileNotFoundError as e:
            return False, str(e)
        except ValueError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Validation error: {e}"

    def _calculate_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        chunk_size = 8192

        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)

        return sha256.hexdigest()[:16]  # Short hash for display

    def _skip_value(self, f, value_type: int) -> None:
        """Skip a value in the GGUF file based on its type."""
        # Type constants from GGUF spec
        TYPE_F32 = 0
        TYPE_U32 = 2
        TYPE_I32 = 4
        TYPE_F64 = 5
        TYPE_U64 = 6
        TYPE_I64 = 7
        TYPE_STRING = 9

        if value_type == TYPE_STRING:
            str_len = struct.unpack("<Q", f.read(8))[0]
            f.read(str_len)
        elif value_type in (TYPE_F32, TYPE_U32, TYPE_I32):
            f.read(4)
        elif value_type in (TYPE_F64, TYPE_U64, TYPE_I64):
            f.read(8)

    def _extract_quantization(self, filename: str) -> Optional[str]:
        """Extract quantization type from filename."""
        # Common quantization patterns
        patterns = [
            "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
            "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
            "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
        ]

        filename_upper = filename.upper()
        for pattern in patterns:
            if pattern.upper() in filename_upper:
                return pattern

        return None
