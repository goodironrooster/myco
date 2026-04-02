"""Tests for myco.gate module."""

import pytest
from pathlib import Path
from myco.gate import AutopoieticGate, GateResult, gate_action
from myco.world import WorldModel


class TestGateResult:
    """Tests for GateResult dataclass."""
    
    def test_create_permit_result(self):
        """Test creating a permit result."""
        result = GateResult(
            permitted=True,
            reason="",
            entropy_before=0.5,
            entropy_after=0.52
        )
        
        assert result.permitted is True
        assert result.entropy_before == 0.5
        assert result.entropy_after == 0.52
    
    def test_create_block_result(self):
        """Test creating a block result."""
        result = GateResult(
            permitted=False,
            reason="Entropy increase too high",
            violation_type="entropy_increase"
        )
        
        assert result.permitted is False
        assert result.reason == "Entropy increase too high"
        assert result.violation_type == "entropy_increase"
    
    def test_str_permit(self):
        """Test string representation for permit."""
        result = GateResult(
            permitted=True,
            entropy_before=0.5,
            entropy_after=0.52
        )
        
        str_repr = str(result)
        
        assert "PERMIT" in str_repr
        assert "0.5" in str_repr
    
    def test_str_block(self):
        """Test string representation for block."""
        result = GateResult(
            permitted=False,
            reason="Test reason"
        )
        
        str_repr = str(result)
        
        assert "BLOCK" in str_repr
        assert "Test reason" in str_repr


class TestAutopoieticGate:
    """Tests for AutopoieticGate class."""
    
    @pytest.fixture
    def gate(self, tmp_path):
        """Create an autopoietic gate."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        world = WorldModel.load(tmp_path)
        return AutopoieticGate(tmp_path, world)
    
    @pytest.fixture
    def simple_project(self, tmp_path):
        """Create a simple project for testing."""
        module_a = tmp_path / "module_a.py"
        module_a.write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
            "from module_b import helper\n"
            "def func_a(): pass\n",
            encoding="utf-8"
        )
        
        module_b = tmp_path / "module_b.py"
        module_b.write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
            "def helper(): pass\n",
            encoding="utf-8"
        )
        
        yield tmp_path
    
    def test_create_gate(self, tmp_path):
        """Test creating a gate."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        world = WorldModel.load(tmp_path)
        gate = AutopoieticGate(tmp_path, world)
        
        assert gate.project_root == tmp_path
        assert gate.world_model == world
        assert gate.get_consecutive_blocks() == 0
    
    def test_check_entropy_delta_permits_small_change(self, gate):
        """Test entropy check permits small changes."""
        result = gate.check_entropy_delta(
            "test.py",
            "add_function"
        )
        
        assert result.permitted is True
    
    def test_check_annotation_preservation_with_annotation(self, gate, tmp_path):
        """Test annotation preservation check with existing annotation."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
            "def test(): pass\n",
            encoding="utf-8"
        )
        
        new_content = (
            "# ⊕ H:0.52 | press:edit | age:0 | drift:+0.02\n"
            "def test(): pass\n"
        )
        
        result = gate.check_annotation_preservation(test_file, new_content)
        
        assert result.permitted is True
    
    def test_check_annotation_preservation_without_annotation(self, gate, tmp_path):
        """Test annotation preservation check removes annotation."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
            "def test(): pass\n",
            encoding="utf-8"
        )
        
        # New content without annotation
        new_content = "def test(): pass\n"
        
        result = gate.check_annotation_preservation(test_file, new_content)
        
        assert result.permitted is False
        assert "annotation" in result.reason.lower()
    
    def test_check_tensegrity_violation_same_type(self, gate):
        """Test tensegrity check detects same-type imports."""
        tensegrity_map = {
            "module_a": "tension",
            "module_b": "tension"
        }
        
        result = gate.check_tensegrity_violation(
            ("module_a", "module_b"),
            tensegrity_map
        )
        
        assert result.permitted is False
        assert "tension" in result.reason
    
    def test_check_tensegrity_valid_cross_boundary(self, gate):
        """Test tensegrity check allows cross-boundary imports."""
        tensegrity_map = {
            "module_a": "compression",
            "module_b": "tension"
        }
        
        result = gate.check_tensegrity_violation(
            ("module_a", "module_b"),
            tensegrity_map
        )
        
        assert result.permitted is True
    
    def test_check_tensegrity_unknown_modules(self, gate):
        """Test tensegrity check allows unknown modules."""
        tensegrity_map = {}
        
        result = gate.check_tensegrity_violation(
            ("unknown_a", "unknown_b"),
            tensegrity_map
        )
        
        assert result.permitted is True
    
    def test_check_action_write_permits(self, gate, tmp_path):
        """Test check_action permits valid write."""
        test_file = tmp_path / "test.py"
        
        result = gate.check_action(
            test_file,
            "write",
            proposed_content="# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef test(): pass\n"
        )
        
        assert result.permitted is True
    
    def test_check_action_resets_block_counter(self, gate):
        """Test check_action resets block counter on permit."""
        # First, manually set block counter
        gate._consecutive_blocks = 3
        
        result = gate.check_action(
            "test.py",
            "write",
            proposed_content="# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef test(): pass\n"
        )
        
        # Should reset on successful check
        assert gate.get_consecutive_blocks() == 0
    
    def test_estimate_entropy_delta_add_import(self, gate):
        """Test entropy delta estimation for add_import."""
        delta = gate._estimate_entropy_delta("add_import")
        
        assert delta > 0  # Adding imports increases coupling
    
    def test_estimate_entropy_delta_decompose(self, gate):
        """Test entropy delta estimation for decompose."""
        delta = gate._estimate_entropy_delta("decompose")
        
        assert delta < 0  # Decomposition reduces entropy
    
    def test_estimate_entropy_delta_default(self, gate):
        """Test entropy delta estimation default."""
        delta = gate._estimate_entropy_delta("unknown_change")
        
        assert delta == 0.02  # Default small increase
    
    def test_path_to_module_name(self, gate, tmp_path):
        """Test path to module name conversion."""
        test_file = tmp_path / "test_module.py"
        
        name = gate._path_to_module_name(test_file)
        
        assert name == "test_module"
    
    def test_reset_block_counter(self, gate):
        """Test resetting block counter."""
        gate._consecutive_blocks = 5
        gate.reset_block_counter()
        
        assert gate.get_consecutive_blocks() == 0


class TestGateAction:
    """Tests for gate_action convenience function."""
    
    def test_gate_action_permits_valid_action(self, tmp_path):
        """Test gate_action permits valid action."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        world = WorldModel.load(tmp_path)
        
        result = gate_action(
            tmp_path,
            world,
            "test.py",
            "write",
            proposed_content="# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef test(): pass\n"
        )
        
        assert result.permitted is True
    
    def test_gate_action_with_import_edge(self, tmp_path):
        """Test gate_action with import edge."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        world = WorldModel.load(tmp_path)
        
        # Create simple modules
        (tmp_path / "module_a.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_a(): pass\n",
            encoding="utf-8"
        )
        (tmp_path / "module_b.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_b(): pass\n",
            encoding="utf-8"
        )
        
        result = gate_action(
            tmp_path,
            world,
            "module_a.py",
            "add_import",
            import_edge=("module_a", "module_b")
        )
        
        # Should complete without error
        assert result is not None
