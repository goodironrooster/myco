"""Tests for myco.delta module."""

import pytest
from pathlib import Path
from myco.delta import (
    DeltaAnalysis,
    EntropyDeltaCalculator,
    analyze_change
)


class TestDeltaAnalysis:
    """Tests for DeltaAnalysis dataclass."""
    
    def test_create_analysis(self):
        """Test creating a delta analysis."""
        analysis = DeltaAnalysis(
            change_type="add_import",
            entropy_before=0.5,
            entropy_after=0.58,
            delta=0.08,
            is_inflection_point=False,
            affected_modules=["module_a", "module_b"],
            recommendation="Proceed"
        )
        
        assert analysis.change_type == "add_import"
        assert analysis.entropy_before == 0.5
        assert analysis.entropy_after == 0.58
        assert analysis.delta == 0.08
        assert analysis.is_inflection_point is False
        assert len(analysis.affected_modules) == 2
    
    def test_create_inflection_point_analysis(self):
        """Test creating inflection point analysis."""
        analysis = DeltaAnalysis(
            change_type="extract_module",
            entropy_before=0.5,
            entropy_after=0.35,
            delta=-0.15,
            is_inflection_point=True,
            affected_modules=["new_module", "source_module"]
        )
        
        assert analysis.is_inflection_point is True
        assert analysis.delta == -0.15
    
    def test_to_dict(self):
        """Test converting analysis to dictionary."""
        analysis = DeltaAnalysis(
            change_type="add_import",
            entropy_before=0.5,
            entropy_after=0.58,
            delta=0.08,
            is_inflection_point=False,
            affected_modules=["a", "b"]
        )
        
        d = analysis.to_dict()
        
        assert d["change_type"] == "add_import"
        assert d["delta"] == 0.08
        assert len(d["affected_modules"]) == 2
    
    def test_to_summary(self):
        """Test generating summary string."""
        analysis = DeltaAnalysis(
            change_type="add_import",
            entropy_before=0.5,
            entropy_after=0.58,
            delta=0.08,
            is_inflection_point=False,
            affected_modules=["module_a", "module_b"],
            recommendation="Proceed"
        )
        
        summary = analysis.to_summary()
        
        assert "Entropy Delta Analysis" in summary
        assert "add_import" in summary
        assert "0.5" in summary
        assert "Proceed" in summary
    
    def test_to_summary_inflection_point(self):
        """Test summary for inflection point."""
        analysis = DeltaAnalysis(
            change_type="extract_module",
            entropy_before=0.5,
            entropy_after=0.35,
            delta=-0.15,
            is_inflection_point=True,
            affected_modules=["new_module"]
        )
        
        summary = analysis.to_summary()
        
        assert "INFLECTION POINT" in summary


class TestEntropyDeltaCalculator:
    """Tests for EntropyDeltaCalculator class."""
    
    @pytest.fixture
    def calculator(self, tmp_path):
        """Create an entropy delta calculator."""
        return EntropyDeltaCalculator(tmp_path)
    
    @pytest.fixture
    def simple_project(self, tmp_path):
        """Create a simple project for testing."""
        module_a = tmp_path / "module_a.py"
        module_a.write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
            "def func_a(): pass\n",
            encoding="utf-8"
        )
        
        module_b = tmp_path / "module_b.py"
        module_b.write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
            "def func_b(): pass\n",
            encoding="utf-8"
        )
        
        yield tmp_path
    
    def test_create_calculator(self, tmp_path):
        """Test creating a calculator."""
        calc = EntropyDeltaCalculator(tmp_path)
        
        assert calc.project_root == tmp_path
        assert calc._base_graph is None
    
    def test_load_base_graph(self, simple_project):
        """Test loading base graph."""
        calc = EntropyDeltaCalculator(simple_project)
        calc.load_base_graph()
        
        assert calc._base_graph is not None
        assert calc._base_entropy is not None
    
    def test_analyze_add_import(self, simple_project):
        """Test analyzing add import change."""
        calc = EntropyDeltaCalculator(simple_project)
        calc.load_base_graph()
        
        analysis = calc.analyze_add_import("module_a", "module_b")
        
        assert analysis.change_type == "add_import"
        assert analysis.affected_modules == ["module_a", "module_b"]
        assert isinstance(analysis.recommendation, str)
    
    def test_analyze_remove_import(self, simple_project):
        """Test analyzing remove import change."""
        calc = EntropyDeltaCalculator(simple_project)
        calc.load_base_graph()
        
        analysis = calc.analyze_remove_import("module_a", "module_b")
        
        assert analysis.change_type == "remove_import"
        assert isinstance(analysis.recommendation, str)
    
    def test_analyze_add_module(self, simple_project):
        """Test analyzing add module change."""
        calc = EntropyDeltaCalculator(simple_project)
        calc.load_base_graph()
        
        analysis = calc.analyze_add_module(
            module_name="new_module",
            imports=["module_a"],
            imported_by=["module_b"]
        )
        
        assert analysis.change_type == "add_module"
        assert "new_module" in analysis.affected_modules
    
    def test_analyze_extract_module(self, simple_project):
        """Test analyzing extract module change."""
        calc = EntropyDeltaCalculator(simple_project)
        calc.load_base_graph()
        
        analysis = calc.analyze_extract_module(
            module_name="extracted",
            source_modules=["module_a"],
            functions_to_extract=["func_a"]
        )
        
        assert analysis.change_type == "extract_module"
        assert "extracted" in analysis.affected_modules
    
    def test_calculate_graph_entropy(self, simple_project):
        """Test graph entropy calculation."""
        calc = EntropyDeltaCalculator(simple_project)
        calc.load_base_graph()
        
        # Test with empty graph
        import networkx as nx
        empty_graph = nx.DiGraph()
        entropy = calc._calculate_graph_entropy(empty_graph)
        assert entropy == 0.0
    
    def test_get_inflection_candidates(self, simple_project):
        """Test getting inflection point candidates."""
        calc = EntropyDeltaCalculator(simple_project)
        calc.load_base_graph()
        
        proposed_changes = [
            {"type": "add_import", "importer": "module_a", "imported": "module_b"},
            {"type": "extract_module", "module_name": "new", "source_modules": ["module_a"]},
        ]
        
        candidates = calc.get_inflection_candidates(proposed_changes)
        
        # Should return list (may be empty if no inflection points)
        assert isinstance(candidates, list)
    
    def test_analyze_change_without_load(self, simple_project):
        """Test analyze methods work without explicit load."""
        calc = EntropyDeltaCalculator(simple_project)
        
        # Should auto-load
        analysis = calc.analyze_add_import("module_a", "module_b")
        
        assert analysis is not None


class TestAnalyzeChange:
    """Tests for analyze_change convenience function."""
    
    def test_analyze_add_import(self, tmp_path):
        """Test analyze_change for add_import."""
        # Create modules
        (tmp_path / "module_a.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_a(): pass\n",
            encoding="utf-8"
        )
        (tmp_path / "module_b.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_b(): pass\n",
            encoding="utf-8"
        )
        
        analysis = analyze_change(
            tmp_path,
            "add_import",
            importer="module_a",
            imported="module_b"
        )
        
        assert analysis.change_type == "add_import"
    
    def test_analyze_remove_import(self, tmp_path):
        """Test analyze_change for remove_import."""
        (tmp_path / "module_a.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_a(): pass\n",
            encoding="utf-8"
        )
        (tmp_path / "module_b.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_b(): pass\n",
            encoding="utf-8"
        )
        
        analysis = analyze_change(
            tmp_path,
            "remove_import",
            importer="module_a",
            imported="module_b"
        )
        
        assert analysis.change_type == "remove_import"
    
    def test_analyze_add_module(self, tmp_path):
        """Test analyze_change for add_module."""
        (tmp_path / "module_a.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_a(): pass\n",
            encoding="utf-8"
        )
        
        analysis = analyze_change(
            tmp_path,
            "add_module",
            module_name="new_module",
            imports=["module_a"],
            imported_by=[]
        )
        
        assert analysis.change_type == "add_module"
    
    def test_analyze_extract_module(self, tmp_path):
        """Test analyze_change for extract_module."""
        (tmp_path / "module_a.py").write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\ndef func_a(): pass\n",
            encoding="utf-8"
        )
        
        analysis = analyze_change(
            tmp_path,
            "extract_module",
            module_name="extracted",
            source_modules=["module_a"],
            functions_to_extract=["func_a"]
        )
        
        assert analysis.change_type == "extract_module"
    
    def test_analyze_unknown_change_type(self, tmp_path):
        """Test analyze_change with unknown type raises error."""
        with pytest.raises(ValueError) as exc_info:
            analyze_change(
                tmp_path,
                "unknown_type",
                some_arg="value"
            )
        
        assert "Unknown change type" in str(exc_info.value)
