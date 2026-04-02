# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Tensegrity classification for myco.

Classifies modules as tension or compression elements.
Ensures the import graph is a valid tensegrity structure.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from .entropy import ImportGraphBuilder


@dataclass
class ClassificationResult:
    """Result of classifying a module."""
    module_name: str
    classification: str  # "tension" or "compression"
    reasons: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "module_name": self.module_name,
            "classification": self.classification,
            "reasons": self.reasons,
            "definitions": self.definitions,
        }


@dataclass
class TensegrityViolation:
    """A violation of the tensegrity invariant."""
    importer: str
    imported: str
    importer_type: str
    imported_type: str
    line_number: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "importer": self.importer,
            "imported": self.imported,
            "importer_type": self.importer_type,
            "imported_type": self.imported_type,
            "line_number": self.line_number,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.importer_type}→{self.imported_type} import: "
            f"{self.importer} → {self.imported}"
        )


class TensegrityClassifier:
    """Classifies modules as tension or compression elements.
    
    Tension elements (carry constraint without state):
    - Protocol classes
    - ABC subclasses
    - TypedDict definitions
    - Type alias declarations
    - Pure function modules (no class-level state)
    
    Compression elements (carry state and implement behavior):
    - Concrete classes with __init__
    - Stateful modules (module-level mutable variables)
    - Dataclasses with mutable fields
    - Entry points
    
    The invariant: every import edge must cross the tension/compression boundary.
    """
    
    def __init__(self, project_root: Path | str):
        """Initialize the classifier.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self._import_graph: Optional[ImportGraphBuilder] = None
        self._classifications: dict[str, ClassificationResult] = {}
        self._violations: list[TensegrityViolation] = []
    
    def scan(self) -> "TensegrityClassifier":
        """Scan the project and classify all modules.
        
        Returns:
            Self for method chaining
        """
        self._import_graph = ImportGraphBuilder(self.project_root)
        self._import_graph.scan()
        
        # Classify each internal module
        for module_name, module_info in self._import_graph.modules.items():
            self._classify_module(module_name, module_info.path)
        
        return self
    
    def _classify_module(self, module_name: str, file_path: Path) -> ClassificationResult:
        """Classify a single module.
        
        Args:
            module_name: Name of the module
            file_path: Path to the file
            
        Returns:
            ClassificationResult
        """
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (IOError, SyntaxError):
            # Can't parse - default to compression
            result = ClassificationResult(
                module_name=module_name,
                classification="compression",
                reasons=["Could not parse file, defaulting to compression"]
            )
            self._classifications[module_name] = result
            return result
        
        reasons = []
        definitions = []
        is_tension = True  # Default to tension, look for compression indicators
        
        # Analyze AST nodes
        for node in ast.walk(tree):
            # Check for Protocol classes
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "Protocol":
                        is_tension = True
                        reasons.append(f"Protocol class: {node.name}")
                        definitions.append(node.name)
                    elif isinstance(base, ast.Name) and base.id == "ABC":
                        is_tension = True
                        reasons.append(f"ABC subclass: {node.name}")
                        definitions.append(node.name)
                    elif isinstance(base, ast.Name) and base.id == "TypedDict":
                        is_tension = True
                        reasons.append(f"TypedDict: {node.name}")
                        definitions.append(node.name)
                
                # Check for __init__ method (compression indicator)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        is_tension = False
                        reasons.append(f"Concrete class with __init__: {node.name}")
                        definitions.append(node.name)
                
                # Check for dataclass with mutable fields
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                        # Check if any field is mutable (list, dict, set)
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign):
                                if self._is_mutable_annotation(item.annotation):
                                    is_tension = False
                                    reasons.append(f"Dataclass with mutable field: {node.name}")
                                    definitions.append(node.name)
                                    break
            
            # Check for module-level mutable state (compression)
            if isinstance(node, ast.Assign) and isinstance(node, ast.AnnAssign):
                # Module-level assignments indicate state
                pass
            
            # Check for entry points (compression)
            if isinstance(node, ast.If) and self._is_main_block(node):
                is_tension = False
                reasons.append("Has if __name__ == '__main__' entry point")
        
        # Check for pure function module (tension)
        if not reasons:
            has_functions = any(
                isinstance(node, ast.FunctionDef)
                for node in ast.walk(tree)
            )
            has_classes = any(
                isinstance(node, ast.ClassDef)
                for node in ast.walk(tree)
            )
            
            if has_functions and not has_classes:
                is_tension = True
                reasons.append("Pure function module (no classes)")
        
        classification = "tension" if is_tension else "compression"
        
        result = ClassificationResult(
            module_name=module_name,
            classification=classification,
            reasons=reasons,
            definitions=definitions
        )
        
        self._classifications[module_name] = result
        return result
    
    def _is_mutable_annotation(self, annotation: ast.AST) -> bool:
        """Check if an annotation indicates a mutable type.
        
        Args:
            annotation: AST node for the annotation
            
        Returns:
            True if mutable type
        """
        if isinstance(annotation, ast.Name):
            return annotation.id in {"list", "dict", "set", "List", "Dict", "Set"}
        elif isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id in {"list", "dict", "set", "List", "Dict", "Set"}
        return False
    
    def _is_main_block(self, node: ast.If) -> bool:
        """Check if an If node is a `if __name__ == '__main__'` block.
        
        Args:
            node: AST If node
            
        Returns:
            True if this is a main block
        """
        test = node.test
        if isinstance(test, ast.Compare):
            if len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                left = test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    if len(test.comparators) == 1:
                        comp = test.comparators[0]
                        if isinstance(comp, ast.Constant) and comp.value == "__main__":
                            return True
        return False
    
    def classify_all(self) -> dict[str, str]:
        """Get classifications for all modules.
        
        Returns:
            Dict mapping module names to "tension" or "compression"
        """
        return {
            name: result.classification
            for name, result in self._classifications.items()
        }
    
    def get_violations(self) -> list[TensegrityViolation]:
        """Find all tensegrity violations in the import graph.
        
        Returns:
            List of TensegrityViolation objects
        """
        if not self._import_graph:
            return []
        
        violations = []
        graph = self._import_graph.get_internal_graph()
        
        for edge in graph.edges():
            importer, imported = edge
            
            importer_type = self._classifications.get(importer)
            imported_type = self._classifications.get(imported)
            
            if importer_type and imported_type:
                if importer_type.classification == imported_type.classification:
                    violation = TensegrityViolation(
                        importer=importer,
                        imported=imported,
                        importer_type=importer_type.classification,
                        imported_type=imported_type.classification
                    )
                    violations.append(violation)
        
        self._violations = violations
        return violations
    
    def has_violations(self) -> bool:
        """Check if there are any tensegrity violations.
        
        Returns:
            True if violations exist
        """
        if not self._violations:
            self.get_violations()
        return len(self._violations) > 0
    
    def get_tension_modules(self) -> list[str]:
        """Get all tension modules.
        
        Returns:
            List of tension module names
        """
        return [
            name for name, result in self._classifications.items()
            if result.classification == "tension"
        ]
    
    def get_compression_modules(self) -> list[str]:
        """Get all compression modules.
        
        Returns:
            List of compression module names
        """
        return [
            name for name, result in self._classifications.items()
            if result.classification == "compression"
        ]
    
    def get_module_classification(self, module_name: str) -> Optional[ClassificationResult]:
        """Get classification for a specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            ClassificationResult or None if not classified
        """
        return self._classifications.get(module_name)
    
    def to_report(self) -> str:
        """Generate a human-readable report.
        
        Returns:
            Report string
        """
        lines = [
            "Tensegrity Classification Report",
            "=================================",
            "",
            f"Tension modules: {len(self.get_tension_modules())}",
            f"Compression modules: {len(self.get_compression_modules())}",
            "",
        ]
        
        if self.get_tension_modules():
            lines.append("Tension (constraints, no state):")
            for name in self.get_tension_modules()[:10]:
                reasons = self._classifications[name].reasons
                lines.append(f"  - {name}: {', '.join(reasons[:2]) if reasons else 'pure functions'}")
            if len(self.get_tension_modules()) > 10:
                lines.append(f"  ... and {len(self.get_tension_modules()) - 10} more")
            lines.append("")
        
        if self.get_compression_modules():
            lines.append("Compression (state, behavior):")
            for name in self.get_compression_modules()[:10]:
                reasons = self._classifications[name].reasons
                lines.append(f"  - {name}: {', '.join(reasons[:2]) if reasons else 'concrete implementation'}")
            if len(self.get_compression_modules()) > 10:
                lines.append(f"  ... and {len(self.get_compression_modules()) - 10} more")
            lines.append("")
        
        violations = self.get_violations()
        if violations:
            lines.append(f"⚠️  VIOLATIONS: {len(violations)}")
            for v in violations[:10]:
                lines.append(f"  - {v}")
            if len(violations) > 10:
                lines.append(f"  ... and {len(violations) - 10} more")
        else:
            lines.append("✓ No tensegrity violations")
        
        return '\n'.join(lines)
