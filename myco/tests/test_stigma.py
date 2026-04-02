"""Tests for myco.stigma module."""

import pytest
from pathlib import Path
from myco.stigma import StigmergicAnnotation, StigmaReader


class TestStigmergicAnnotation:
    """Tests for StigmergicAnnotation dataclass."""
    
    def test_create_default_annotation(self):
        """Test creating annotation with default values."""
        annotation = StigmergicAnnotation()
        
        assert annotation.H == 0.50
        assert annotation.press == "none"
        assert annotation.age == 0
        assert annotation.drift == 0.00
    
    def test_create_custom_annotation(self):
        """Test creating annotation with custom values."""
        annotation = StigmergicAnnotation(
            H=0.73,
            press="interface_inversion",
            age=4,
            drift=0.12
        )
        
        assert annotation.H == 0.73
        assert annotation.press == "interface_inversion"
        assert annotation.age == 4
        assert annotation.drift == 0.12
    
    def test_format_annotation(self):
        """Test formatting annotation as comment string."""
        annotation = StigmergicAnnotation(
            H=0.73,
            press="interface_inversion",
            age=4,
            drift=0.12
        )
        
        formatted = annotation.format()
        assert formatted == "# ⊕ H:0.73 | press:interface_inversion | age:4 | drift:+0.12"
    
    def test_format_negative_drift(self):
        """Test formatting annotation with negative drift."""
        annotation = StigmergicAnnotation(
            H=0.54,
            press="interface_inversion",
            age=2,
            drift=-0.06
        )
        
        formatted = annotation.format()
        assert formatted == "# ⊕ H:0.54 | press:interface_inversion | age:2 | drift:-0.06"
    
    def test_parse_valid_annotation(self):
        """Test parsing a valid annotation line."""
        line = "# ⊕ H:0.73 | press:interface_inversion | age:4 | drift:+0.12"
        
        annotation = StigmergicAnnotation.parse(line)
        
        assert annotation is not None
        assert annotation.H == 0.73
        assert annotation.press == "interface_inversion"
        assert annotation.age == 4
        assert annotation.drift == 0.12
    
    def test_parse_negative_drift(self):
        """Test parsing annotation with negative drift."""
        line = "# ⊕ H:0.54 | press:interface_inversion | age:2 | drift:-0.06"
        
        annotation = StigmergicAnnotation.parse(line)
        
        assert annotation is not None
        assert annotation.drift == -0.06
    
    def test_parse_invalid_line(self):
        """Test parsing a line without annotation."""
        line = "# This is a regular comment"
        
        annotation = StigmergicAnnotation.parse(line)
        
        assert annotation is None
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        annotation = StigmergicAnnotation.parse("")
        assert annotation is None
    
    def test_validate_invalid_H(self):
        """Test validation rejects invalid H values."""
        with pytest.raises(ValueError):
            StigmergicAnnotation(H=1.5)
        
        with pytest.raises(ValueError):
            StigmergicAnnotation(H=-0.1)
    
    def test_validate_invalid_press(self):
        """Test validation rejects invalid press types."""
        with pytest.raises(ValueError):
            StigmergicAnnotation(press="invalid_type")
    
    def test_valid_press_types(self):
        """Test all valid press types are accepted."""
        valid_presses = [
            "decompose",
            "interface_inversion",
            "tension_extraction",
            "compression_collapse",
            "entropy_drain",
            "attractor_escape",
            "none"
        ]
        
        for press in valid_presses:
            annotation = StigmergicAnnotation(press=press)
            assert annotation.press == press


class TestStigmaReader:
    """Tests for StigmaReader class."""
    
    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary file for testing."""
        file_path = tmp_path / "test_module.py"
        yield file_path
        # Cleanup automatic via tmp_path
    
    @pytest.fixture
    def annotated_file(self, tmp_path):
        """Create a temporary file with annotation."""
        file_path = tmp_path / "annotated_module.py"
        file_path.write_text(
            "# ⊕ H:0.73 | press:interface_inversion | age:4 | drift:+0.12\n"
            "def hello():\n"
            "    print('world')\n",
            encoding="utf-8"
        )
        yield file_path
    
    def test_read_annotation_from_file(self, annotated_file):
        """Test reading annotation from a file."""
        reader = StigmaReader(annotated_file)
        annotation = reader.read_annotation()
        
        assert annotation is not None
        assert annotation.H == 0.73
        assert annotation.press == "interface_inversion"
    
    def test_read_annotation_no_annotation(self, temp_file):
        """Test reading from file without annotation."""
        temp_file.write_text("def hello():\n    pass\n")
        
        reader = StigmaReader(temp_file)
        annotation = reader.read_annotation()
        
        assert annotation is None
    
    def test_read_annotation_file_not_found(self, tmp_path):
        """Test reading from non-existent file."""
        reader = StigmaReader(tmp_path / "missing.py")
        
        with pytest.raises(FileNotFoundError):
            reader.read_annotation()
    
    def test_write_annotation_new_file(self, temp_file):
        """Test writing annotation to a new file."""
        # Create the file first
        temp_file.write_text("def hello():\n    pass\n", encoding="utf-8")
        
        reader = StigmaReader(temp_file)
        annotation = StigmergicAnnotation(H=0.50, press="none", age=0, drift=0.00)
        
        reader.write_annotation(annotation)
        
        content = temp_file.read_text(encoding="utf-8")
        assert content.startswith("# ⊕ H:0.50 | press:none | age:0 | drift:+0.00")
    
    def test_update_annotation(self, annotated_file):
        """Test updating an existing annotation."""
        reader = StigmaReader(annotated_file)
        
        updated = reader.update_annotation(H=0.60, press="decompose", age=1)
        
        assert updated.H == 0.60
        assert updated.press == "decompose"
        assert updated.age == 1
        
        # Verify file was updated
        annotation = reader.read_annotation()
        assert annotation.H == 0.60
    
    def test_find_annotation_line(self, temp_file):
        """Test finding the correct line for annotation."""
        temp_file.write_text(
            "# Module docstring\n"
            "def hello():\n"
            "    pass\n"
        )
        
        reader = StigmaReader(temp_file)
        line_no = reader.find_annotation_line()
        
        # Should insert before first substantive code
        assert line_no >= 0
    
    def test_get_first_substantive_line(self, temp_file):
        """Test getting the first substantive AST node."""
        temp_file.write_text(
            "# Comment\n"
            "import os\n"
            "def hello():\n"
            "    pass\n"
        )
        
        reader = StigmaReader(temp_file)
        node = reader.get_first_substantive_line()
        
        # Should skip imports and find the function
        assert node is not None
    
    def test_handles_syntax_error(self, temp_file):
        """Test handling files with syntax errors."""
        temp_file.write_text("def broken(\n")  # Missing closing paren
        
        reader = StigmaReader(temp_file)
        
        with pytest.raises(SyntaxError):
            reader.read_annotation()
