"""Integration tests for myco CLI."""

import json
import pytest
from pathlib import Path
from click.testing import CliRunner
from myco.cli import cli
from myco.world import WorldModel


@pytest.fixture
def project_with_world(tmp_path):
    """Create a project with world model."""
    myco_dir = tmp_path / ".myco"
    myco_dir.mkdir()

    # Create a simple Python file
    test_file = tmp_path / "test_module.py"
    test_file.write_text(
        "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
        "def hello():\n"
        "    return 'world'\n",
        encoding="utf-8"
    )

    yield tmp_path


@pytest.fixture
def project_with_imports(tmp_path):
    """Create a project with module imports for dual regime testing."""
    myco_dir = tmp_path / ".myco"
    myco_dir.mkdir()

    # Create module_a (imports module_b)
    module_a = tmp_path / "module_a.py"
    module_a.write_text(
        "from module_b import helper\n\n"
        "def func_a():\n"
        "    return helper()\n"
    )

    # Create module_b (imports module_c)
    module_b = tmp_path / "module_b.py"
    module_b.write_text(
        "from module_c import util\n\n"
        "def helper():\n"
        "    return util()\n"
    )

    # Create module_c (base module)
    module_c = tmp_path / "module_c.py"
    module_c.write_text(
        "def util():\n"
        "    pass\n"
    )

    yield tmp_path


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestCliCommands:
    """Integration tests for CLI commands."""
    
    def test_cli_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "0.9" in result.output  # Updated version
    
    def test_cli_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(cli, ["--help"])
        
        assert result.exit_code == 0
        assert "myco" in result.output.lower()
        assert "run" in result.output
        assert "entropy" in result.output
        assert "tensegrity" in result.output
        assert "world" in result.output


class TestWorldCommand:
    """Tests for world command."""
    
    def test_world_command(self, runner, project_with_world):
        """Test world command displays state."""
        with runner.isolated_filesystem():
            # Change to project directory
            import os
            os.chdir(project_with_world)
            
            result = runner.invoke(cli, ["world"])
            
            assert result.exit_code == 0
            assert "World Model" in result.output
            assert "Sessions" in result.output
            assert "Entropy baseline" in result.output


class TestEntropyCommand:
    """Tests for entropy command."""
    
    def test_entropy_command(self, runner, project_with_world):
        """Test entropy command analyzes codebase."""
        with runner.isolated_filesystem():
            import os
            os.chdir(project_with_world)
            
            result = runner.invoke(cli, ["entropy"])
            
            assert result.exit_code == 0
            assert "Entropy Analysis" in result.output
            assert "Entropy Report" in result.output
            assert "Modules analyzed" in result.output


class TestTensegrityCommand:
    """Tests for tensegrity command."""
    
    def test_tensegrity_command(self, runner, project_with_world):
        """Test tensegrity command analyzes structure."""
        with runner.isolated_filesystem():
            import os
            os.chdir(project_with_world)
            
            result = runner.invoke(cli, ["tensegrity"])
            
            assert result.exit_code == 0
            assert "Tensegrity Classification Report" in result.output
            assert "Tension modules" in result.output
            assert "Compression modules" in result.output


class TestSessionLoggerIntegration:
    """Tests for session logger integration."""
    
    def test_world_file_updated(self, tmp_path):
        """Test that world.json is accessible."""
        # Create world model
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        
        # Verify world file exists after load
        world = WorldModel.load(tmp_path)
        assert world.schema_version == 1


class TestStigmergicAnnotationIntegration:
    """Tests for stigmergic annotation integration."""
    
    def test_annotation_read_write(self, tmp_path):
        """Test annotation can be read and written."""
        from myco.stigma import StigmaReader, StigmergicAnnotation
        
        # Create test file with annotation
        test_file = tmp_path / "test_module.py"
        test_file.write_text(
            "# ⊕ H:0.50 | press:none | age:0 | drift:+0.00\n"
            "def hello():\n"
            "    return 'world'\n",
            encoding="utf-8"
        )
        
        # Read existing annotation
        reader = StigmaReader(test_file)
        annotation = reader.read_annotation()
        
        assert annotation is not None
        assert annotation.H == 0.50
        assert annotation.press == "none"
        
        # Update annotation (use valid press type)
        reader.update_annotation(H=0.55, press="decompose", age=0, drift=0.05)
        
        # Verify update with fresh reader
        updated_reader = StigmaReader(test_file)
        updated = updated_reader.read_annotation()
        assert updated is not None
        assert updated.H == 0.55
        assert updated.press == "decompose"


class TestModuleIntegration:
    """Tests for module integration."""
    
    def test_all_modules_importable(self):
        """Test all myco modules can be imported."""
        from myco import stigma, entropy, world, gate, attractor, energy
        from myco import tensegrity, rank, delta, session_log, cli
        
        # Verify key classes exist
        assert hasattr(stigma, "StigmergicAnnotation")
        assert hasattr(stigma, "StigmaReader")
        assert hasattr(entropy, "ImportGraphBuilder")
        assert hasattr(entropy, "EntropyCalculator")
        assert hasattr(world, "WorldModel")
        assert hasattr(gate, "AutopoieticGate")
        assert hasattr(attractor, "AttractorDetector")
        assert hasattr(energy, "EnergyTracker")
        assert hasattr(tensegrity, "TensegrityClassifier")
        assert hasattr(rank, "RankCalculator")
        assert hasattr(delta, "EntropyDeltaCalculator")
        assert hasattr(session_log, "SessionLogger")
    
    def test_cli_has_all_commands(self):
        """Test CLI has all expected commands."""
        from myco.cli import cli
        
        # Get command names
        command_names = list(cli.commands.keys())
        
        assert "run" in command_names
        assert "entropy" in command_names
        assert "tensegrity" in command_names
        assert "world" in command_names


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_entropy_on_empty_directory(self, runner, tmp_path):
        """Test entropy command handles empty directory."""
        with runner.isolated_filesystem():
            import os
            os.chdir(tmp_path)
            
            result = runner.invoke(cli, ["entropy"])
            
            assert result.exit_code == 0
            # Should complete without error, even with 0 modules
    
    def test_tensegrity_on_empty_directory(self, runner, tmp_path):
        """Test tensegrity command handles empty directory."""
        with runner.isolated_filesystem():
            import os
            os.chdir(tmp_path)
            
            result = runner.invoke(cli, ["tensegrity"])
            
            assert result.exit_code == 0
            # Should complete without error


class TestWorldModelPersistence:
    """Tests for world model persistence."""
    
    def test_world_model_survives_round_trip(self, tmp_path):
        """Test world model can be saved and loaded."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()
        
        # Create and save
        world = WorldModel.load(tmp_path)
        world.session_count = 5
        world.entropy_baseline = 0.45
        world.add_intention("test intention")
        world.save()
        
        # Load and verify
        loaded = WorldModel.load(tmp_path)
        assert loaded.session_count == 5
        assert abs(loaded.entropy_baseline - 0.45) < 0.001
        assert "test intention" in loaded.open_intentions
    
    def test_world_model_handles_corruption(self, tmp_path):
        """Test world model handles corrupted file."""
        myco_dir = tmp_path / ".myco"
        myco_dir.mkdir()

        # Write corrupted file
        world_file = myco_dir / "world.json"
        world_file.write_text("not valid json {{{")

        # Should bootstrap new world instead of crashing
        world = WorldModel.load(tmp_path)
        assert world is not None
        assert world.schema_version == 1


class TestDualRegimeIntegration:
    """Tests for Proposal 1: Dual regime classification integration."""

    def test_context_assembly_includes_dual_regime(self, project_with_imports):
        """Test that context assembly includes dual regime data."""
        from myco.cli import assemble_context
        
        world = WorldModel.load(project_with_imports)
        context = assemble_context(project_with_imports, world)
        
        # Check regime analysis exists
        regime_analysis = context['entropy_gradient'].get('regime_analysis', [])
        
        # Should have regime data for priority files
        if regime_analysis:
            for regime in regime_analysis:
                # Check dual regime fields are present
                assert 'H_structural' in regime or 'H' in regime
                assert 'H_internal' in regime
                assert 'internal_regime' in regime
                assert 'combined_regime' in regime or 'regime' in regime
                assert 'dual_guidance' in regime or 'guidance' in regime

    def test_dual_regime_classification_values(self, project_with_imports):
        """Test dual regime classification produces valid values."""
        from myco.cli import assemble_context
        from myco.entropy import classify_dual_regime
        
        # Test various combinations
        result = classify_dual_regime(0.2, 0.8)
        assert result['combined_regime'] == 'crystallized_chaotic'
        assert result['structural_regime'] == 'crystallized'
        assert result['internal_regime'] == 'chaotic'
        assert result['priority'] == 1
        
        result = classify_dual_regime(0.5, 0.5)
        assert result['combined_regime'] == 'dissipative'
        assert result['guidance'] == 'Healthy. Safe to modify.'
        
        result = classify_dual_regime(0.8, 0.8)
        assert result['combined_regime'] == 'diffuse'
        assert result['priority'] == 1

    def test_format_context_shows_dual_regime(self, project_with_imports):
        """Test that format_context displays dual regime information."""
        from myco.cli import assemble_context, format_context
        
        world = WorldModel.load(project_with_imports)
        context = assemble_context(project_with_imports, world)
        
        formatted = format_context(context)
        
        # Should contain context sections
        assert '[WORLD MODEL]' in formatted
        assert '[ENTROPY GRADIENT]' in formatted
        
        # If regime analysis exists, should show dual classification
        if context['entropy_gradient'].get('regime_analysis'):
            assert '[REGIME ANALYSIS - DUAL CLASSIFICATION]' in formatted


class TestSubstrateReport:
    """Tests for Proposal 5: Substrate Health Report."""

    def test_report_command_exists(self, runner):
        """Test that report command is available."""
        result = runner.invoke(cli, ["report"])

        assert result.exit_code == 0
        assert "# Substrate Health Report" in result.output

    def test_report_contains_overall_health(self, runner, tmp_path):
        """Test report contains overall health section."""
        with runner.isolated_filesystem():
            import os
            os.chdir(tmp_path)

            # Create myco directory
            (tmp_path / ".myco").mkdir()

            result = runner.invoke(cli, ["report"])

            assert result.exit_code == 0
            assert "## Overall Health" in result.output
            assert "Health Score" in result.output

    def test_report_contains_modules_section(self, runner, tmp_path):
        """Test report contains modules requiring attention section."""
        with runner.isolated_filesystem():
            import os
            os.chdir(tmp_path)

            # Create myco directory
            (tmp_path / ".myco").mkdir()

            result = runner.invoke(cli, ["report"])

            assert result.exit_code == 0
            assert "## Modules Requiring Attention" in result.output

    def test_report_contains_gradient_field(self, runner, tmp_path):
        """Test report contains gradient field section."""
        with runner.isolated_filesystem():
            import os
            os.chdir(tmp_path)

            # Create myco directory
            (tmp_path / ".myco").mkdir()

            result = runner.invoke(cli, ["report"])

            assert result.exit_code == 0
            assert "## Structural Stress" in result.output or "Gradient Field" in result.output

    def test_report_contains_recommendations(self, runner, tmp_path):
        """Test report contains recommendations section."""
        with runner.isolated_filesystem():
            import os
            os.chdir(tmp_path)

            # Create myco directory
            (tmp_path / ".myco").mkdir()

            result = runner.invoke(cli, ["report"])

            assert result.exit_code == 0
            assert "## Recommendations" in result.output

    def test_generate_substrate_report_structure(self, tmp_path):
        """Test report generation function returns valid structure."""
        from myco.cli import generate_substrate_report
        from myco.entropy import analyze_entropy, calculate_substrate_health, compute_gradient_field
        from myco.world import WorldModel

        # Create myco directory
        (tmp_path / ".myco").mkdir()

        entropy_report = analyze_entropy(tmp_path)
        substrate_health = calculate_substrate_health(tmp_path)
        gradient_field = compute_gradient_field(tmp_path)
        world = WorldModel.load(tmp_path)

        lines = generate_substrate_report(
            project_root=tmp_path,
            entropy_report=entropy_report,
            substrate_health=substrate_health,
            gradient_field=gradient_field,
            world=world,
            regime_details=[],
            intervention_records=[]
        )

        assert isinstance(lines, list)
        assert len(lines) > 0
        assert "# Substrate Health Report" in lines[0]
