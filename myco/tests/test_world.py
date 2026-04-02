"""Tests for myco.world module."""

import json
import pytest
from pathlib import Path
from datetime import datetime
from myco.world import WorldModel


class TestWorldModel:
    """Tests for WorldModel class."""
    
    @pytest.fixture
    def world_path(self, tmp_path):
        """Create a temporary .myco directory."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        yield tmp_path
    
    def test_create_default_world(self):
        """Test creating WorldModel with default values."""
        world = WorldModel()
        
        assert world.schema_version == 1
        assert world.session_count == 0
        assert world.entropy_baseline == 0.50
        assert world.entropy_trend == 0.00
        assert world.crystallized_modules == []
        assert world.diffuse_modules == []
        assert world.open_intentions == []
    
    def test_load_nonexistent_creates_bootstrap(self, world_path):
        """Test loading non-existent file creates bootstrap world."""
        world = WorldModel.load(world_path)
        
        assert world.session_count == 0
        assert world.entropy_baseline == 0.50
        
        # File should be created
        world_file = world_path / ".myco" / "world.json"
        assert world_file.exists()
    
    def test_load_existing_file(self, world_path):
        """Test loading existing world file."""
        # Create world file
        world_file = world_path / ".myco" / "world.json"
        world_file.write_text(json.dumps({
            "schema_version": 1,
            "last_session": "2026-03-19T00:00:00Z",
            "session_count": 5,
            "entropy_baseline": 0.45,
            "entropy_trend": -0.05,
            "crystallized_modules": ["module_a"],
            "diffuse_modules": [],
            "active_attractors": [],
            "last_press_type": "decompose",
            "tensegrity_violations": 0,
            "open_intentions": ["fix module_a"]
        }))
        
        world = WorldModel.load(world_path)
        
        assert world.session_count == 5
        assert world.entropy_baseline == 0.45
        assert world.entropy_trend == -0.05
        assert "module_a" in world.crystallized_modules
        assert "fix module_a" in world.open_intentions
    
    def test_load_corrupted_file_bootstrap(self, world_path):
        """Test loading corrupted file creates bootstrap."""
        world_file = world_path / ".myco" / "world.json"
        world_file.write_text("not valid json")
        
        world = WorldModel.load(world_path)
        
        # Should bootstrap with defaults
        assert world.session_count == 0
    
    def test_save_world(self, world_path):
        """Test saving world model."""
        world = WorldModel(
            session_count=3,
            entropy_baseline=0.55,
            _path=world_path / ".myco" / "world.json"
        )
        
        world.save()
        
        # Verify file was written
        world_file = world_path / ".myco" / "world.json"
        assert world_file.exists()
        
        # Verify content
        data = json.loads(world_file.read_text())
        assert data["session_count"] == 3
        assert data["entropy_baseline"] == 0.55
    
    def test_start_session(self, world_path):
        """Test starting a new session."""
        world = WorldModel.load(world_path)
        initial_count = world.session_count
        
        world.start_session()
        
        assert world.session_count == initial_count + 1
        assert world.last_session is not None
    
    def test_end_session(self, world_path):
        """Test ending a session with metrics."""
        world = WorldModel.load(world_path)
        world.start_session()
        
        world.end_session(
            entropy_baseline=0.45,
            crystallized=["module_a"],
            diffuse=["module_b"],
            tensegrity_violations=1
        )
        
        assert world.entropy_baseline == 0.45
        assert "module_a" in world.crystallized_modules
        assert "module_b" in world.diffuse_modules
        assert world.tensegrity_violations == 1
    
    def test_end_session_calculates_trend(self, world_path):
        """Test that end_session calculates entropy trend."""
        world = WorldModel.load(world_path)
        world.entropy_baseline = 0.50
        
        world.end_session(
            entropy_baseline=0.45,
            crystallized=[],
            diffuse=[]
        )
        
        # Use approximate comparison for floating point
        assert abs(world.entropy_trend - (-0.05)) < 0.001
    
    def test_record_press(self, world_path):
        """Test recording a press type."""
        world = WorldModel.load(world_path)
        
        world.record_press("interface_inversion")
        
        assert world.last_press_type == "interface_inversion"
    
    def test_add_intention(self, world_path):
        """Test adding an open intention."""
        world = WorldModel.load(world_path)
        
        world.add_intention("refactor module_a")
        
        assert "refactor module_a" in world.open_intentions
    
    def test_add_intention_no_duplicates(self, world_path):
        """Test adding intention doesn't create duplicates."""
        world = WorldModel.load(world_path)
        
        world.add_intention("refactor module_a")
        world.add_intention("refactor module_a")
        
        count = world.open_intentions.count("refactor module_a")
        assert count == 1
    
    def test_resolve_intention_found(self, world_path):
        """Test resolving an existing intention."""
        world = WorldModel.load(world_path)
        world.add_intention("refactor module_a")
        
        result = world.resolve_intention("refactor module_a")
        
        assert result is True
        assert "refactor module_a" not in world.open_intentions
    
    def test_resolve_intention_not_found(self, world_path):
        """Test resolving non-existent intention."""
        world = WorldModel.load(world_path)
        
        result = world.resolve_intention("nonexistent")
        
        assert result is False
    
    def test_resolve_intention_partial_match(self, world_path):
        """Test resolving intention with partial match."""
        world = WorldModel.load(world_path)
        world.add_intention("refactor module_a to improve coupling")
        
        result = world.resolve_intention("refactor module_a")
        
        assert result is True
    
    def test_add_attractor(self, world_path):
        """Test adding an attractor."""
        world = WorldModel.load(world_path)
        
        world.add_attractor("import_restructure_loop")
        
        assert "import_restructure_loop" in world.active_attractors
    
    def test_clear_attractors(self, world_path):
        """Test clearing all attractors."""
        world = WorldModel.load(world_path)
        world.add_attractor("attractor_1")
        world.add_attractor("attractor_2")
        
        world.clear_attractors()
        
        assert world.active_attractors == []
    
    def test_to_context_dict(self, world_path):
        """Test converting to context dictionary."""
        world = WorldModel.load(world_path)
        world.start_session()
        
        context = world.to_context_dict()
        
        assert "schema_version" in context
        assert "session_count" in context
        assert "entropy_baseline" in context
        assert "open_intentions" in context
    
    def test_str_representation(self, world_path):
        """Test string representation."""
        world = WorldModel.load(world_path)
        world.add_intention("test intention")
        
        str_repr = str(world)
        
        assert "World Model" in str_repr
        assert "test intention" in str_repr
    
    def test_save_without_path_raises(self):
        """Test saving without path raises error."""
        world = WorldModel()
        
        with pytest.raises(ValueError):
            world.save()
