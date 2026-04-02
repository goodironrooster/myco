# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Autopoietic gate for myco.

The gate is the primary operating loop. It checks invariants before any action.
If an action would make the codebase worse for its future self, the gate blocks it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .entropy import EntropyCalculator, ImportGraphBuilder
from .stigma import StigmaReader, StigmergicAnnotation
from .world import WorldModel


@dataclass
class GateResult:
    """Result from the autopoietic gate check."""
    permitted: bool
    reason: str = ""
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    violation_type: Optional[str] = None
    affects_self: bool = False  # Step 2: Track if change affects myco/ itself

    def __str__(self) -> str:
        if self.permitted:
            self_note = " [SELF]" if self.affects_self else ""
            return f"PERMIT (H: {self.entropy_before:.3f} → {self.entropy_after:.3f}){self_note}"
        else:
            return f"BLOCK: {self.reason}"


class AutopoieticGate:
    """The autopoietic gate - checks invariants before any action.
    
    Invariants:
    1. Entropy must not increase by more than 0.15
    2. Stigmergic annotations must not be removed without rewriting
    3. No compression→compression import edges
    4. No tension→tension import edges
    
    The gate is not a safety feature. It is the primary operating loop.
    """
    
    ENTROPY_THRESHOLD = 0.15
    
    def __init__(self, project_root: Path | str, world_model: WorldModel):
        """Initialize the gate.

        Args:
            project_root: Root directory of the project
            world_model: World model instance
        """
        self.project_root = Path(project_root)
        self.world_model = world_model
        self._consecutive_blocks = 0
        
        # File-type threshold adjustments
        # HTML/CSS/MD files don't participate in import graph, so relaxed threshold
        self.THRESHOLD_ADJUSTMENTS = {
            ".html": 0.30,  # HTML files are naturally large
            ".htm": 0.30,
            ".css": 0.30,   # CSS can be large
            ".scss": 0.30,
            ".sass": 0.30,
            ".less": 0.30,
            ".md": 0.40,    # Markdown doesn't affect structure
            ".markdown": 0.40,
            ".txt": 0.30,   # Text files OK
            ".json": 0.20,  # Config files OK
            ".yaml": 0.20,
            ".yml": 0.20,
            ".xml": 0.20,
            ".js": 0.15,    # JavaScript moderate
            ".ts": 0.15,    # TypeScript moderate
            ".py": 0.00,    # Python strict (import graph matters)
        }
    
    def check_entropy_delta(
        self,
        file_path: Path | str,
        proposed_change: str
    ) -> GateResult:
        """Check if a proposed change would increase entropy beyond threshold.

        Args:
            file_path: Path to the file being modified
            proposed_change: Description of the proposed change

        Returns:
            GateResult with permit/block decision
        """
        file_path = Path(file_path)

        # Build import graph before change
        builder_before = ImportGraphBuilder(self.project_root)
        builder_before.scan()
        calc_before = EntropyCalculator(builder_before)

        # Get module entropy before
        module_name = self._path_to_module_name(file_path)
        try:
            entropy_before = calc_before.calculate_module_entropy(module_name)
        except Exception:
            entropy_before = 0.0

        # Estimate entropy after based on change type
        entropy_delta = self._estimate_entropy_delta(proposed_change)
        entropy_after = entropy_before + entropy_delta

        # Compute effective threshold (re-counts files each time for dynamic adjustment)
        threshold = self._get_effective_threshold(file_path)

        # Use small epsilon for floating point comparison tolerance
        # This allows values like 0.300 to pass when threshold is 0.30
        epsilon = 0.001
        
        if entropy_delta > threshold + epsilon:
            # Get regime for actionable suggestions
            regime = self._get_file_regime(file_path, entropy_before)
            suggestion = self._get_refactoring_suggestion(regime, file_path)

            reason = f"Entropy increase {entropy_delta:.3f} exceeds threshold {threshold:.2f}"
            if suggestion:
                reason += f"\n\n{suggestion}"

            return GateResult(
                permitted=False,
                reason=reason,
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                violation_type="entropy_increase"
            )
        
        # Warning: Close to threshold (within epsilon tolerance)
        if entropy_delta > threshold - epsilon:
            return GateResult(
                permitted=True,
                reason=f"WARNING: Entropy increase {entropy_delta:.3f} is at threshold limit {threshold:.2f}. Consider smaller changes.",
                entropy_before=entropy_before,
                entropy_after=entropy_after,
                violation_type="threshold_warning"
            )

        return GateResult(
            permitted=True,
            entropy_before=entropy_before,
            entropy_after=entropy_after
        )

    def _get_file_regime(self, file_path: Path, entropy: float) -> str:
        """Determine the entropy regime for a file."""
        if entropy < 0.3:
            return "crystallized"
        elif entropy > 0.75:
            return "diffuse"
        else:
            return "dissipative"

    def _get_refactoring_suggestion(self, regime: str, file_path: Path) -> str:
        """Get actionable refactoring suggestion based on regime."""
        suggestions = {
            "crystallized": """SUGGESTION: This module is crystallized (rigid, over-coupled).
Consider these refactoring approaches:
  1. Extract Protocol/ABC - Create interfaces for external dependencies
  2. Decompose - Split into 2-3 smaller, focused modules
  3. Dependency inversion - Depend on abstractions, not concretions

Example: If this is a service module, extract a Protocol for its interface.""",
            
            "diffuse": """SUGGESTION: This module is diffuse (under-coupled, boundary-less).
Consider these refactoring approaches:
  1. Consolidate - Merge related functions into coherent classes
  2. Extract common abstractions - Find shared patterns
  3. Add structure - Organize with clear boundaries

Example: If functions are scattered, group them into a class.""",
            
            "dissipative": """SUGGESTION: This module is in healthy dissipative regime.
The entropy increase may still be too large. Consider:
  1. Smaller changes - Break into multiple smaller edits
  2. Incremental refactoring - Make one change at a time
  3. Review necessity - Is this change essential now?"""
        }
        return suggestions.get(regime, "")

    def _get_effective_threshold(self, file_path: Path | str) -> float:
        """Get effective entropy threshold based on project maturity and file type.

        Project Maturity Model:
        - Embryo (0-5 files): Maximum flexibility (0.50) — no coupling exists yet
        - Growth (5-20 files): Moderate protection (0.30) — structure forming
        - Mature (20-100 files): Strict protection (0.15) — real coupling exists
        - Legacy (100+ files): Maximum protection (0.10) — any change is risky

        File Type Adjustments:
        - HTML/CSS/MD: Relaxed (don't participate in import graph)
        - Python: Base threshold (import graph matters)

        Args:
            file_path: Path to the file being modified

        Returns:
            Effective threshold value
        """
        # Ensure file_path is a Path object
        file_path = Path(file_path)
        
        # Count Python files (better metric for maturity than all code files)
        py_files = len(list(self.project_root.glob("**/*.py")))
        
        # Maturity-based base threshold
        if py_files < 5:
            base_threshold = 0.50  # Embryo: maximum flexibility
        elif py_files < 20:
            base_threshold = 0.30  # Growth: moderate protection
        elif py_files < 100:
            base_threshold = 0.15  # Mature: strict protection
        else:
            base_threshold = 0.10  # Legacy: maximum protection

        # File-type adjustment (non-Python files don't affect import graph)
        ext = file_path.suffix.lower()
        if ext not in {".py"}:
            type_adjustment = self.THRESHOLD_ADJUSTMENTS.get(ext, 0.00)
            return base_threshold + type_adjustment
        
        # Python files: use base threshold (import graph matters)
        return base_threshold
    
    def check_annotation_preservation(
        self,
        file_path: Path | str,
        new_content: str
    ) -> GateResult:
        """Check that stigmergic annotation is preserved or rewritten.
        
        Args:
            file_path: Path to the file being modified
            new_content: New file content
            
        Returns:
            GateResult with permit/block decision
        """
        file_path = Path(file_path)
        
        # Read existing annotation
        try:
            reader = StigmaReader(file_path)
            existing = reader.read_annotation()
        except (FileNotFoundError, SyntaxError):
            existing = None
        
        # Check if new content has annotation
        has_annotation = new_content.startswith("# ⊕")
        
        if existing and not has_annotation:
            return GateResult(
                permitted=False,
                reason="Stigmergic annotation removed without replacement",
                violation_type="annotation_removal"
            )
        
        return GateResult(permitted=True)
    
    def check_tensegrity_violation(
        self,
        import_edge: tuple[str, str],
        tensegrity_map: dict[str, str]
    ) -> GateResult:
        """Check that an import edge crosses the tension/compression boundary.
        
        Args:
            import_edge: (importer, imported) module names
            tensegrity_map: Dict mapping module names to 'tension' or 'compression'
            
        Returns:
            GateResult with permit/block decision
        """
        importer, imported = import_edge
        
        importer_type = tensegrity_map.get(importer)
        imported_type = tensegrity_map.get(imported)
        
        if importer_type is None or imported_type is None:
            # Unknown modules - permit with warning
            return GateResult(permitted=True)
        
        # Check for same-type imports (violations)
        if importer_type == imported_type:
            if importer_type == "tension":
                return GateResult(
                    permitted=False,
                    reason=f"Tension→tension import: {importer} → {imported}",
                    violation_type="tension_tension_edge"
                )
            else:
                return GateResult(
                    permitted=False,
                    reason=f"Compression→compression import: {importer} → {imported}",
                    violation_type="compression_compression_edge"
                )
        
        return GateResult(permitted=True)
    
    def check_action(
        self,
        file_path: Path | str,
        action_type: str,
        proposed_content: Optional[str] = None,
        import_edge: Optional[tuple[str, str]] = None
    ) -> GateResult:
        """Run the full autopoietic gate check on a proposed action.

        Args:
            file_path: Path to the file being modified
            action_type: Type of action (write, edit, add_import, etc.)
            proposed_content: New file content (for write/edit)
            import_edge: (importer, imported) for import additions

        Returns:
            GateResult with permit/block decision
        """
        file_path = Path(file_path)

        # Track consecutive blocks
        self._consecutive_blocks = 0

        # Step 2: Check if this affects myco/ itself
        affects_self = False
        try:
            rel_path = file_path.relative_to(self.project_root)
            affects_self = str(rel_path).startswith("myco" + "/") or str(rel_path).startswith("myco\\")
        except ValueError:
            pass  # file_path not relative to project_root

        # Check 1: Entropy delta
        if action_type in ("write", "edit", "add_feature"):
            result = self.check_entropy_delta(file_path, action_type)
            if not result.permitted:
                self._consecutive_blocks += 1
                result.affects_self = affects_self
                return result
            result.affects_self = affects_self

        # Check 2: Annotation preservation
        if proposed_content and action_type in ("write", "edit"):
            result = self.check_annotation_preservation(file_path, proposed_content)
            if not result.permitted:
                self._consecutive_blocks += 1
                result.affects_self = affects_self
                return result
            result.affects_self = affects_self

        # Check 3: Tensegrity (for import additions)
        if import_edge:
            # Load tensegrity map
            from .tensegrity import TensegrityClassifier
            classifier = TensegrityClassifier(self.project_root)
            classifier.scan()
            tensegrity_map = classifier.classify_all()

            result = self.check_tensegrity_violation(import_edge, tensegrity_map)
            if not result.permitted:
                self._consecutive_blocks += 1
                result.affects_self = affects_self
                return result
            result.affects_self = affects_self

        # All checks passed
        return GateResult(permitted=True, affects_self=affects_self)
    
    def _estimate_entropy_delta(self, change_type: str) -> float:
        """Estimate entropy delta based on change type.

        This is a heuristic. The actual delta would require applying
        the change and recalculating the import graph.

        MYCO Principle: Pure functions (no internal imports) should have
        zero or negative entropy delta — they add value without coupling.

        Args:
            change_type: Description of the change

        Returns:
            Estimated entropy delta (negative = good, positive = caution)
        """
        # Heuristics based on change type
        # Positive changes (reduce coupling) — REWARD
        if "decompose" in change_type.lower():
            return -0.15  # Decomposition reduces entropy
        elif "extract" in change_type.lower():
            return -0.10  # Extracting modules typically reduces coupling
        elif "simplify" in change_type.lower():
            return -0.10  # Simplification reduces entropy
        elif "remove_import" in change_type.lower():
            return -0.05  # Removing imports decreases coupling
        
        # Neutral changes (maintain structure) — MINIMAL IMPACT
        elif "add_function" in change_type.lower():
            return 0.00  # Pure functions don't increase coupling (good!)
        elif "add_constant" in change_type.lower():
            return 0.00  # Constants are free
        elif "fix_bug" in change_type.lower():
            return 0.00  # Bug fixes are neutral
        elif "add_test" in change_type.lower():
            return -0.02  # Tests reduce future entropy
        
        # Negative changes (increase coupling) — CAUTION
        elif "add_import" in change_type.lower():
            return 0.10  # Adding imports increases coupling
        elif "add_class" in change_type.lower():
            return 0.08  # Classes can increase coupling potential
        elif "add_global" in change_type.lower():
            return 0.15  # Global state increases coupling
        else:
            return 0.02  # Default: small increase (conservative)
    
    def _path_to_module_name(self, path: Path) -> str:
        """Convert a file path to a module name."""
        try:
            rel_path = path.relative_to(self.project_root)
            parts = list(rel_path.parts)
            
            if parts and parts[-1].endswith('.py'):
                parts[-1] = parts[-1][:-3]
            
            if parts and parts[-1] == '__init__':
                parts = parts[:-1]
            
            return '.'.join(parts) if parts else self.project_root.name
        except ValueError:
            return path.stem
    
    def get_consecutive_blocks(self) -> int:
        """Get the number of consecutive blocks."""
        return self._consecutive_blocks
    
    def reset_block_counter(self) -> None:
        """Reset the consecutive block counter."""
        self._consecutive_blocks = 0


def gate_action(
    project_root: Path | str,
    world_model: WorldModel,
    file_path: Path | str,
    action_type: str,
    **kwargs
) -> GateResult:
    """Convenience function to gate a single action.
    
    Args:
        project_root: Root directory of the project
        world_model: World model instance
        file_path: Path to the file being modified
        action_type: Type of action
        **kwargs: Additional arguments for the gate check
        
    Returns:
        GateResult with permit/block decision
    """
    gate = AutopoieticGate(project_root, world_model)
    return gate.check_action(file_path, action_type, **kwargs)
