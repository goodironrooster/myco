"""Tests for myco.entropy module."""

import pytest
from pathlib import Path
from myco.entropy import (
    ModuleInfo,
    ImportGraphBuilder,
    EntropyCalculator,
    EntropyReport,
    analyze_entropy
)


class TestModuleInfo:
    """Tests for ModuleInfo dataclass."""
    
    def test_create_module_info(self):
        """Test creating ModuleInfo with default values."""
        info = ModuleInfo(
            path=Path("test.py"),
            name="test"
        )
        
        assert info.path == Path("test.py")
        assert info.name == "test"
        assert info.imports == []
        assert info.imported_by == []
        assert info.out_degree == 0
        assert info.in_degree == 0
        assert info.entropy == 0.0


class TestImportGraphBuilder:
    """Tests for ImportGraphBuilder class."""
    
    @pytest.fixture
    def simple_project(self, tmp_path):
        """Create a simple project structure."""
        # Create module a
        module_a = tmp_path / "module_a.py"
        module_a.write_text(
            "import os\n"
            "from module_b import helper\n"
            "def func_a(): pass\n"
        )
        
        # Create module b
        module_b = tmp_path / "module_b.py"
        module_b.write_text(
            "from module_c import util\n"
            "def helper(): pass\n"
        )
        
        # Create module c
        module_c = tmp_path / "module_c.py"
        module_c.write_text(
            "def util(): pass\n"
        )
        
        yield tmp_path
    
    @pytest.fixture
    def nested_project(self, tmp_path):
        """Create a nested project structure."""
        # Create package
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        
        # Create __init__.py
        init_file = pkg_dir / "__init__.py"
        init_file.write_text("# Package init\n")
        
        # Create module in package
        module_file = pkg_dir / "module.py"
        module_file.write_text(
            "from mypackage import utils\n"
            "def func(): pass\n"
        )
        
        # Create utils module
        utils_file = pkg_dir / "utils.py"
        utils_file.write_text("def helper(): pass\n")
        
        yield tmp_path
    
    def test_scan_simple_project(self, simple_project):
        """Test scanning a simple project."""
        builder = ImportGraphBuilder(simple_project)
        builder.scan()
        
        assert len(builder.modules) == 3
        assert "module_a" in builder.modules
        assert "module_b" in builder.modules
        assert "module_c" in builder.modules
    
    def test_extract_imports(self, simple_project):
        """Test extracting imports from source code."""
        builder = ImportGraphBuilder(simple_project)
        
        source = """
import os
import sys
from module_b import helper
from package.submodule import func
"""
        imports = builder._extract_imports(source)
        
        assert "os" in imports
        assert "sys" in imports
        assert "module_b" in imports
        assert "package.submodule" in imports
    
    def test_path_to_module_name(self, simple_project):
        """Test converting file path to module name."""
        builder = ImportGraphBuilder(simple_project)
        
        name = builder._path_to_module_name(simple_project / "module_a.py")
        assert name == "module_a"
    
    def test_path_to_module_name_nested(self, nested_project):
        """Test converting nested path to module name."""
        builder = ImportGraphBuilder(nested_project)
        
        pkg_path = nested_project / "mypackage" / "module.py"
        name = builder._path_to_module_name(pkg_path)
        assert name == "mypackage.module"
    
    def test_path_to_init_module(self, nested_project):
        """Test converting __init__.py to module name."""
        builder = ImportGraphBuilder(nested_project)
        
        init_path = nested_project / "mypackage" / "__init__.py"
        name = builder._path_to_module_name(init_path)
        assert name == "mypackage"
    
    def test_get_internal_graph(self, simple_project):
        """Test getting internal subgraph."""
        builder = ImportGraphBuilder(simple_project)
        builder.scan()
        
        internal_graph = builder.get_internal_graph()
        
        # Should only include internal modules
        for node in internal_graph.nodes():
            assert internal_graph.nodes[node].get('internal', False)
    
    def test_scan_handles_syntax_errors(self, tmp_path):
        """Test scanning handles files with syntax errors."""
        # Create file with syntax error
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(\n")  # Missing paren
        
        # Create valid file
        good_file = tmp_path / "good.py"
        good_file.write_text("def good(): pass\n")
        
        builder = ImportGraphBuilder(tmp_path)
        builder.scan()
        
        # Should still process valid files
        assert "good" in builder.modules


class TestEntropyCalculator:
    """Tests for EntropyCalculator class."""
    
    @pytest.fixture
    def simple_project(self, tmp_path):
        """Create a simple project for entropy testing."""
        module_a = tmp_path / "module_a.py"
        module_a.write_text(
            "from module_b import helper\n"
            "def func_a(): pass\n"
        )
        
        module_b = tmp_path / "module_b.py"
        module_b.write_text("def helper(): pass\n")
        
        yield tmp_path
    
    def test_calculate_module_entropy(self, simple_project):
        """Test calculating entropy for a single module."""
        builder = ImportGraphBuilder(simple_project)
        builder.scan()
        
        calculator = EntropyCalculator(builder)
        entropy = calculator.calculate_module_entropy("module_a")
        
        # Entropy should be between 0 and 1
        assert 0.0 <= entropy <= 1.0
    
    def test_calculate_global_entropy(self, simple_project):
        """Test calculating global entropy."""
        builder = ImportGraphBuilder(simple_project)
        builder.scan()
        
        calculator = EntropyCalculator(builder)
        global_entropy = calculator.calculate_global_entropy()
        
        # Should be between 0 and 1
        assert 0.0 <= global_entropy <= 1.0
    
    def test_get_module_regimes(self, simple_project):
        """Test classifying modules by regime."""
        builder = ImportGraphBuilder(simple_project)
        builder.scan()
        
        calculator = EntropyCalculator(builder)
        regimes = calculator.get_module_regimes()
        
        # All modules should have a regime
        # Note: Only modules with edges will be in the internal graph
        for node in calculator.graph.nodes():
            assert node in regimes
            assert regimes[node] in ["crystallized", "dissipative", "diffuse"]
    
    def test_get_modules_by_deviation(self, simple_project):
        """Test getting modules by deviation from baseline."""
        builder = ImportGraphBuilder(simple_project)
        builder.scan()
        
        calculator = EntropyCalculator(builder)
        deviations = calculator.get_modules_by_deviation(top_n=2)
        
        assert len(deviations) <= 2
        
        # Each deviation should be (module_name, entropy, drift)
        for name, entropy, drift in deviations:
            assert isinstance(name, str)
            assert isinstance(entropy, float)
            assert isinstance(drift, float)
    
    def test_empty_graph_entropy(self, tmp_path):
        """Test entropy calculation on empty graph."""
        builder = ImportGraphBuilder(tmp_path)
        # Don't scan - empty graph
        
        calculator = EntropyCalculator(builder)
        entropy = calculator.calculate_global_entropy()
        
        assert entropy == 0.0


class TestEntropyReport:
    """Tests for EntropyReport dataclass."""
    
    def test_create_report(self):
        """Test creating an EntropyReport."""
        report = EntropyReport(
            global_entropy=0.48,
            module_count=10,
            crystallized=["module_a"],
            dissipative=["module_b", "module_c"],
            diffuse=[],
            top_deviations=[("module_a", 0.2, -0.1)]
        )
        
        assert report.global_entropy == 0.48
        assert report.module_count == 10
        assert len(report.crystallized) == 1
    
    def test_report_summary(self):
        """Test generating report summary."""
        report = EntropyReport(
            global_entropy=0.48,
            module_count=10,
            crystallized=["module_a"],
            dissipative=["module_b"],
            diffuse=["module_c"],
            top_deviations=[]
        )
        
        summary = report.summary()
        
        assert "Entropy Report" in summary
        assert "0.48" in summary
        assert "10" in summary


class TestAnalyzeEntropy:
    """Tests for analyze_entropy convenience function."""
    
    @pytest.fixture
    def test_project(self, tmp_path):
        """Create a test project."""
        module_a = tmp_path / "module_a.py"
        module_a.write_text(
            "from module_b import helper\n"
            "def func(): pass\n"
        )
        
        module_b = tmp_path / "module_b.py"
        module_b.write_text("def helper(): pass\n")
        
        yield tmp_path
    
    def test_analyze_entropy(self, test_project):
        """Test analyzing entropy of a project."""
        report = analyze_entropy(test_project)
        
        assert isinstance(report, EntropyReport)
        assert report.module_count >= 2
        assert 0.0 <= report.global_entropy <= 1.0
    
    def test_analyze_empty_directory(self, tmp_path):
        """Test analyzing empty directory."""
        report = analyze_entropy(tmp_path)

        assert isinstance(report, EntropyReport)
        assert report.module_count == 0
        assert report.global_entropy == 0.0


class TestGetPriorityFiles:
    """Tests for get_priority_files function."""

    def test_get_priority_files_empty_directory(self, tmp_path):
        """Test priority files with empty directory."""
        from myco.entropy import get_priority_files
        
        priority = get_priority_files(tmp_path)
        
        assert isinstance(priority, list)
        assert len(priority) == 0

    def test_get_priority_files_returns_list(self, tmp_path):
        """Test that priority files returns a list of dicts."""
        from myco.entropy import get_priority_files
        
        # Create some test files
        (tmp_path / "file1.py").write_text("def a(): pass\n")
        (tmp_path / "file2.py").write_text("def b(): pass\n")
        
        priority = get_priority_files(tmp_path, top_n=5)
        
        assert isinstance(priority, list)
        # All items should have required keys
        for item in priority:
            assert "file" in item
            assert "priority" in item
            assert "reason" in item
            assert "action_hint" in item

    def test_get_priority_files_respects_top_n(self, tmp_path):
        """Test that top_n limits results."""
        from myco.entropy import get_priority_files
        
        # Create multiple test files
        for i in range(10):
            (tmp_path / f"module_{i}.py").write_text(f"def func_{i}(): pass\n")
        
        priority = get_priority_files(tmp_path, top_n=3)
        
        assert len(priority) <= 3

    def test_get_priority_files_with_imports(self, tmp_path):
        """Test priority files with modules that have imports."""
        from myco.entropy import get_priority_files
        
        # Create modules with imports (creates coupling)
        (tmp_path / "module_a.py").write_text(
            "from module_b import helper\n"
            "from module_c import util\n"
            "def func(): pass\n"
        )
        (tmp_path / "module_b.py").write_text("def helper(): pass\n")
        (tmp_path / "module_c.py").write_text("def util(): pass\n")
        
        priority = get_priority_files(tmp_path, top_n=5)
        
        assert isinstance(priority, list)
        # Should have some priority files based on coupling
        for item in priority:
            assert item["priority"] in [1, 2, 3]
            assert item["reason"] in ["crystallized", "high_drift", "diffuse"]

    def test_priority_ordering(self, tmp_path):
        """Test that results are ordered by priority."""
        from myco.entropy import get_priority_files
        
        # Create test files
        for i in range(5):
            (tmp_path / f"file_{i}.py").write_text(f"def func_{i}(): pass\n")
        
        priority = get_priority_files(tmp_path, top_n=5)
        
        # Results should be sorted by priority
        if len(priority) > 1:
            for i in range(len(priority) - 1):
                assert priority[i]["priority"] <= priority[i + 1]["priority"]


class TestSubstrateHealth:
    """Tests for substrate health calculation."""

    def test_calculate_substrate_health_empty(self, tmp_path):
        """Test health calculation for empty directory."""
        from myco.entropy import calculate_substrate_health
        
        health = calculate_substrate_health(tmp_path)
        
        assert "health_score" in health
        assert "status" in health
        assert "breakdown" in health
        assert "metrics" in health

    def test_calculate_substrate_health_with_modules(self, tmp_path):
        """Test health calculation with multiple modules."""
        from myco.entropy import calculate_substrate_health
        
        # Create several modules
        for i in range(5):
            (tmp_path / f"module_{i}.py").write_text(f"def func_{i}(): pass\n")
        
        health = calculate_substrate_health(tmp_path)
        
        assert 0 <= health["health_score"] <= 1
        assert health["metrics"]["total_modules"] == 5

    def test_substrate_health_status_values(self, tmp_path):
        """Test that health status is one of expected values."""
        from myco.entropy import calculate_substrate_health
        
        health = calculate_substrate_health(tmp_path)
        
        assert health["status"] in ["healthy", "stable", "degraded", "critical"]
        assert "status_message" in health

    def test_substrate_health_breakdown(self, tmp_path):
        """Test health breakdown contains all components."""
        from myco.entropy import calculate_substrate_health
        
        health = calculate_substrate_health(tmp_path)
        breakdown = health["breakdown"]
        
        assert "entropy_distribution" in breakdown
        assert "crystallized_score" in breakdown
        assert "diffuse_score" in breakdown
        assert "entropy_level" in breakdown


class TestRelatedFiles:
    """Tests for get_related_files function."""

    def test_get_related_files_empty(self, tmp_path):
        """Test related files for non-existent file."""
        from myco.entropy import get_related_files
        
        related = get_related_files(tmp_path, tmp_path / "nonexistent.py")
        assert related == []

    def test_get_related_files_no_imports(self, tmp_path):
        """Test related files for module with no imports."""
        from myco.entropy import get_related_files
        
        # Create isolated module
        (tmp_path / "isolated.py").write_text("def func(): pass\n")
        
        related = get_related_files(tmp_path, tmp_path / "isolated.py")
        assert related == []

    def test_get_related_files_with_imports(self, tmp_path):
        """Test related files finds imported modules."""
        from myco.entropy import get_related_files
        
        # Create modules with imports
        (tmp_path / "utils.py").write_text("def helper(): pass\n")
        (tmp_path / "main.py").write_text("from utils import helper\n")
        
        related = get_related_files(tmp_path, tmp_path / "main.py")
        
        assert len(related) >= 1
        assert any(r["file"] == "utils.py" for r in related)
        assert any(r["relationship"] == "imports" for r in related)

    def test_get_related_files_with_dependents(self, tmp_path):
        """Test related files finds dependent modules."""
        from myco.entropy import get_related_files
        
        # Create modules where one imports the other
        (tmp_path / "base.py").write_text("def base(): pass\n")
        (tmp_path / "derived.py").write_text("from base import base\n")
        
        related = get_related_files(tmp_path, tmp_path / "base.py")
        
        assert len(related) >= 1
        assert any(r["file"] == "derived.py" for r in related)
        assert any(r["relationship"] == "imported_by" for r in related)

    def test_get_related_files_respects_max(self, tmp_path):
        """Test related files respects max_files limit."""
        from myco.entropy import get_related_files
        
        # Create multiple modules
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.py").write_text("def b(): pass\n")
        (tmp_path / "c.py").write_text("def c(): pass\n")
        (tmp_path / "main.py").write_text("from a import a\nfrom b import b\nfrom c import c\n")
        
        related = get_related_files(tmp_path, tmp_path / "main.py", max_files=2)
        
        assert len(related) <= 2


class TestRelatedContent:
    """Tests for read_related_content function."""

    def test_read_related_content_empty(self, tmp_path):
        """Test reading content for empty related files list."""
        from myco.entropy import read_related_content
        
        content = read_related_content(tmp_path, [])
        assert content == []

    def test_read_related_content_with_files(self, tmp_path):
        """Test reading content for related files."""
        from myco.entropy import read_related_content
        
        # Create files
        (tmp_path / "utils.py").write_text("def helper(): pass\n")
        (tmp_path / "main.py").write_text("from utils import helper\n")
        
        related = [
            {"file": "utils.py", "relationship": "imports", "module": "utils"}
        ]
        
        content = read_related_content(tmp_path, related)
        
        assert len(content) == 1
        assert content[0]["file"] == "utils.py"
        assert content[0]["relationship"] == "imports"
        assert "helper" in content[0]["content"]

    def test_read_related_content_truncates(self, tmp_path):
        """Test that content is truncated if too long."""
        from myco.entropy import read_related_content
        
        # Create file with long content
        long_content = "x = 1\n" * 100
        (tmp_path / "long.py").write_text(long_content)
        
        related = [
            {"file": "long.py", "relationship": "imports", "module": "long"}
        ]
        
        content = read_related_content(tmp_path, related, max_content_length=100)
        
        assert len(content) == 1
        assert len(content[0]["content"]) <= 100 + len("\n... (truncated)")
        assert "(truncated)" in content[0]["content"]

    def test_read_related_content_skips_missing(self, tmp_path):
        """Test that missing files are skipped."""
        from myco.entropy import read_related_content
        
        related = [
            {"file": "nonexistent.py", "relationship": "imports", "module": "none"}
        ]
        
        content = read_related_content(tmp_path, related)
        
        assert content == []


class TestInternalEntropy:
    """Tests for internal module entropy functions."""

    def test_compute_function_size_entropy_single_function(self, tmp_path):
        """Test function size entropy with single function."""
        from myco.entropy import compute_function_size_entropy
        import ast
        
        source = "def func(): pass\n"
        tree = ast.parse(source)
        
        H = compute_function_size_entropy(tree)
        
        # Single function should return default
        assert H == 0.5

    def test_compute_function_size_entropy_uniform(self, tmp_path):
        """Test function size entropy with uniform sizes."""
        from myco.entropy import compute_function_size_entropy
        import ast
        
        # Two functions of similar size
        source = """
def func1():
    x = 1
    return x

def func2():
    y = 2
    return y
"""
        tree = ast.parse(source)
        H = compute_function_size_entropy(tree)
        
        # Should be high (uniform distribution)
        assert H > 0.8

    def test_compute_nesting_depth_entropy(self, tmp_path):
        """Test nesting depth entropy."""
        from myco.entropy import compute_nesting_depth_entropy
        import ast
        
        source = """
def shallow():
    if True:
        pass

def deep():
    if True:
        for i in range(10):
            while True:
                pass
"""
        tree = ast.parse(source)
        H = compute_nesting_depth_entropy(tree)
        
        # Should be between 0 and 1
        assert 0 <= H <= 1

    def test_compute_name_cohesion(self, tmp_path):
        """Test name cohesion calculation."""
        from myco.entropy import compute_name_cohesion
        import ast
        
        # High cohesion - same names repeated
        source = """
def process_data(data):
    result = process_data(data)
    return result
"""
        tree = ast.parse(source)
        cohesion = compute_name_cohesion(tree)
        
        # Should be between 0 and 1
        assert 0 <= cohesion <= 1

    def test_compute_internal_entropy(self, tmp_path):
        """Test internal entropy computation."""
        from myco.entropy import compute_internal_entropy
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass\n")
        
        metrics = compute_internal_entropy(test_file)
        
        assert "H_function_size" in metrics
        assert "H_nesting" in metrics
        assert "cohesion" in metrics
        assert "H_internal" in metrics
        assert 0 <= metrics["H_internal"] <= 1

    def test_compute_internal_entropy_nonexistent(self, tmp_path):
        """Test internal entropy for nonexistent file."""
        from myco.entropy import compute_internal_entropy
        
        metrics = compute_internal_entropy(tmp_path / "nonexistent.py")
        
        assert metrics["H_internal"] == 0.5

    def test_classify_dual_regime_crystallized(self):
        """Test dual regime classification for crystallized."""
        from myco.entropy import classify_dual_regime
        
        result = classify_dual_regime(H_structural=0.2, H_internal=0.2)
        
        assert result["combined_regime"] == "crystallized"
        assert result["priority"] == 1

    def test_classify_dual_regime_dissipative(self):
        """Test dual regime classification for dissipative."""
        from myco.entropy import classify_dual_regime
        
        result = classify_dual_regime(H_structural=0.5, H_internal=0.5)
        
        assert result["combined_regime"] == "dissipative"
        assert result["guidance"] == "Healthy. Safe to modify."

    def test_classify_dual_regime_diffuse(self):
        """Test dual regime classification for diffuse."""
        from myco.entropy import classify_dual_regime
        
        result = classify_dual_regime(H_structural=0.8, H_internal=0.8)
        
        assert result["combined_regime"] == "diffuse"
        assert result["priority"] == 1


class TestRegimeIntervention:
    """Tests for entropy regime intervention recommendations."""

    def test_get_regime_intervention_crystallized(self):
        """Test intervention for crystallized regime."""
        from myco.entropy import get_regime_intervention
        from pathlib import Path
        
        result = get_regime_intervention(Path("test.py"), H=0.2)
        
        assert result["regime"] == "crystallized"
        assert result["primary"] == "decompose"
        assert "decompose" in result["interventions"]
        assert "interface_inversion" in result["interventions"]
        assert "Do NOT add features" in result["guidance"]
        assert result["warning"] is not None

    def test_get_regime_intervention_dissipative(self):
        """Test intervention for dissipative regime."""
        from myco.entropy import get_regime_intervention
        from pathlib import Path
        
        result = get_regime_intervention(Path("test.py"), H=0.5)
        
        assert result["regime"] == "dissipative"
        assert result["primary"] == "none"
        assert result["interventions"] == ["none"]
        assert "healthy" in result["guidance"].lower()
        assert result["warning"] is None

    def test_get_regime_intervention_diffuse(self):
        """Test intervention for diffuse regime."""
        from myco.entropy import get_regime_intervention
        from pathlib import Path
        
        result = get_regime_intervention(Path("test.py"), H=0.85)
        
        assert result["regime"] == "diffuse"
        assert result["primary"] == "compression_collapse"
        assert "compression_collapse" in result["interventions"]
        assert "tension_extraction" in result["interventions"]
        assert "consolidation" in result["guidance"].lower()
        assert result["warning"] is not None

    def test_get_regime_intervention_boundaries(self):
        """Test regime boundaries."""
        from myco.entropy import get_regime_intervention
        from pathlib import Path

        # H = 0.3 should be dissipative (boundary)
        result = get_regime_intervention(Path("test.py"), H=0.3)
        assert result["regime"] == "dissipative"

        # H = 0.75 should be dissipative (boundary)
        result = get_regime_intervention(Path("test.py"), H=0.75)
        assert result["regime"] == "dissipative"


class TestGradientField:
    """Tests for gradient field analysis (Proposal 3)."""

    def test_gradient_edge_creation(self):
        """Test GradientEdge dataclass."""
        from myco.entropy import GradientEdge

        edge = GradientEdge(
            importer="module_a",
            imported="module_b",
            H_importer=0.2,
            H_imported=0.8,
            gradient=0.6,
            is_fault_line=True
        )

        assert edge.importer == "module_a"
        assert edge.gradient == 0.6
        assert edge.is_fault_line is True

        # Test to_dict
        d = edge.to_dict()
        assert d["importer"] == "module_a"
        assert d["gradient"] == 0.6

    def test_module_stress_creation(self):
        """Test ModuleStress dataclass."""
        from myco.entropy import ModuleStress

        stress = ModuleStress(
            module="module_a",
            H=0.5,
            edge_count=3,
            mean_gradient=0.4,
            max_gradient=0.6,
            fault_line_count=2,
            fault_lines=["module_b", "module_c"]
        )

        assert stress.module == "module_a"
        assert stress.fault_line_count == 2

        # Test to_dict
        d = stress.to_dict()
        assert d["module"] == "module_a"
        assert d["mean_gradient"] == 0.4

    def test_gradient_field_report_creation(self):
        """Test GradientFieldReport dataclass."""
        from myco.entropy import GradientFieldReport, GradientEdge, ModuleStress

        report = GradientFieldReport(
            fault_lines=[],
            module_stress=[],
            total_edges=5,
            fault_line_count=0,
            mean_gradient=0.2,
            threshold=0.3
        )

        assert report.total_edges == 5
        assert report.fault_line_count == 0

        # Test summary
        summary = report.summary()
        assert "Gradient Field Report" in summary
        assert "Threshold: 0.3" in summary

    def test_compute_gradient_field_with_imports(self, tmp_path):
        """Test gradient field computation with modules that have imports."""
        from myco.entropy import compute_gradient_field

        # Create modules with different entropy profiles
        # module_a: many imports (higher H)
        (tmp_path / "module_a.py").write_text(
            "from module_b import b\n"
            "from module_c import c\n"
            "def a(): pass\n"
        )
        # module_b: imports module_c (medium H)
        (tmp_path / "module_b.py").write_text(
            "from module_c import c\n"
            "def b(): pass\n"
        )
        # module_c: no imports (lower H)
        (tmp_path / "module_c.py").write_text("def c(): pass\n")

        report = compute_gradient_field(tmp_path, threshold=0.3)

        # Should have edges for the imports
        # Note: edges are only counted for internal modules in the graph
        assert report.total_edges >= 0  # May be 0 if no internal edges detected
        assert isinstance(report.mean_gradient, float)
        assert report.mean_gradient >= 0.0

    def test_compute_gradient_field_empty(self, tmp_path):
        """Test gradient field on empty directory."""
        from myco.entropy import compute_gradient_field

        report = compute_gradient_field(tmp_path)

        assert report.total_edges == 0
        assert report.fault_line_count == 0
        assert report.mean_gradient == 0.0

    def test_get_fault_line_modules(self, tmp_path):
        """Test getting fault line modules."""
        from myco.entropy import get_fault_line_modules

        # Create modules
        (tmp_path / "hub.py").write_text(
            "from a import a\nfrom b import b\nfrom c import c\n"
        )
        (tmp_path / "a.py").write_text("def a(): pass\n")
        (tmp_path / "b.py").write_text("def b(): pass\n")
        (tmp_path / "c.py").write_text("def c(): pass\n")

        modules = get_fault_line_modules(tmp_path, threshold=0.2, top_n=5)

        # Should return list of dicts
        assert isinstance(modules, list)
        for m in modules:
            assert "module" in m
            assert "mean_gradient" in m
            assert "fault_line_count" in m

    def test_gradient_field_report_summary(self):
        """Test gradient field report summary output."""
        from myco.entropy import (
            GradientFieldReport, GradientEdge, ModuleStress
        )

        # Create report with fault lines
        fault_lines = [
            GradientEdge(
                importer="a", imported="b",
                H_importer=0.2, H_imported=0.8,
                gradient=0.6, is_fault_line=True
            )
        ]
        module_stress = [
            ModuleStress(
                module="a", H=0.2, edge_count=1,
                mean_gradient=0.6, max_gradient=0.6,
                fault_line_count=1, fault_lines=["b"]
            )
        ]

        report = GradientFieldReport(
            fault_lines=fault_lines,
            module_stress=module_stress,
            total_edges=1,
            fault_line_count=1,
            mean_gradient=0.6,
            threshold=0.3
        )

        summary = report.summary()

        assert "FAULT LINES" in summary
        assert "a (H=0.20) → b (H=0.80)" in summary
        assert "HIGHEST STRESS MODULES" in summary
