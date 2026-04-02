# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Entropy delta calculator for myco.

Computes the entropy delta that would result from a proposed structural change.
Flags changes where |ΔH| > 0.2 as inflection point candidates.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import networkx as nx

from .entropy import EntropyCalculator, ImportGraphBuilder, ModuleInfo


@dataclass
class DeltaAnalysis:
    """Result of entropy delta analysis."""
    change_type: str
    entropy_before: float
    entropy_after: float
    delta: float
    is_inflection_point: bool
    affected_modules: list[str]
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "change_type": self.change_type,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "delta": self.delta,
            "is_inflection_point": self.is_inflection_point,
            "affected_modules": self.affected_modules,
            "recommendation": self.recommendation,
        }
    
    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        status = "⚠️  INFLECTION POINT" if self.is_inflection_point else "✓ Normal change"
        return (
            f"Entropy Delta Analysis: {status}\n"
            f"  Change type: {self.change_type}\n"
            f"  Entropy: {self.entropy_before:.3f} → {self.entropy_after:.3f} (Δ={self.delta:+.3f})\n"
            f"  Affected modules: {', '.join(self.affected_modules[:5])}\n"
            f"  Recommendation: {self.recommendation or 'Proceed with caution'}"
        )


class EntropyDeltaCalculator:
    """Calculates entropy delta for proposed structural changes.
    
    Change types:
    - add_import: Adding a new import edge
    - remove_import: Removing an import edge
    - add_module: Adding a new module
    - remove_module: Removing a module
    - move_function: Moving a function between modules
    - extract_module: Extracting code into a new module
    - merge_modules: Merging two modules
    """
    
    INFLECTION_THRESHOLD = 0.2
    
    def __init__(self, project_root: Path | str):
        """Initialize the calculator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self._base_graph: Optional[ImportGraphBuilder] = None
        self._base_entropy: Optional[float] = None
    
    def load_base_graph(self) -> "EntropyDeltaCalculator":
        """Load and cache the base import graph.
        
        Returns:
            Self for method chaining
        """
        self._base_graph = ImportGraphBuilder(self.project_root)
        self._base_graph.scan()
        
        calculator = EntropyCalculator(self._base_graph)
        self._base_entropy = calculator.calculate_global_entropy()
        
        return self
    
    def analyze_add_import(
        self,
        importer: str,
        imported: str
    ) -> DeltaAnalysis:
        """Analyze the entropy delta of adding an import edge.
        
        Args:
            importer: Module that will import
            imported: Module that will be imported
            
        Returns:
            DeltaAnalysis with results
        """
        if self._base_graph is None:
            self.load_base_graph()
        
        # Create a copy of the graph for simulation
        sim_graph = self._base_graph.graph.copy()
        
        # Add the proposed edge
        sim_graph.add_edge(importer, imported)
        
        # Calculate entropy after
        entropy_after = self._calculate_graph_entropy(sim_graph)
        
        # Calculate delta
        delta = entropy_after - self._base_entropy
        
        # Determine if inflection point
        is_inflection = abs(delta) > self.INFLECTION_THRESHOLD
        
        # Generate recommendation
        if delta > 0.2:
            recommendation = (
                "⚠️  Large entropy increase. Consider decomposing the imported module "
                "first, or extracting a shared interface."
            )
        elif delta < -0.2:
            recommendation = (
                "✓ Large entropy decrease. This change improves structural health. "
                "Consider this as part of a larger refactoring."
            )
        else:
            recommendation = "Proceed - entropy change is within normal range."
        
        return DeltaAnalysis(
            change_type="add_import",
            entropy_before=self._base_entropy,
            entropy_after=entropy_after,
            delta=delta,
            is_inflection_point=is_inflection,
            affected_modules=[importer, imported],
            recommendation=recommendation
        )
    
    def analyze_remove_import(
        self,
        importer: str,
        imported: str
    ) -> DeltaAnalysis:
        """Analyze the entropy delta of removing an import edge.
        
        Args:
            importer: Module that currently imports
            imported: Module that is currently imported
            
        Returns:
            DeltaAnalysis with results
        """
        if self._base_graph is None:
            self.load_base_graph()
        
        # Create a copy of the graph for simulation
        sim_graph = self._base_graph.graph.copy()
        
        # Remove the edge if it exists
        if sim_graph.has_edge(importer, imported):
            sim_graph.remove_edge(importer, imported)
        
        # Calculate entropy after
        entropy_after = self._calculate_graph_entropy(sim_graph)
        
        # Calculate delta
        delta = entropy_after - self._base_entropy
        
        # Determine if inflection point
        is_inflection = abs(delta) > self.INFLECTION_THRESHOLD
        
        # Generate recommendation
        if delta < -0.15:
            recommendation = (
                "✓ Removing this coupling improves modularity. "
                "Ensure no functionality is broken."
            )
        elif delta > 0.15:
            recommendation = (
                "⚠️  Unexpected entropy increase. This may indicate the import "
                "was providing important structure. Review dependencies."
            )
        else:
            recommendation = "Proceed - entropy change is within normal range."
        
        return DeltaAnalysis(
            change_type="remove_import",
            entropy_before=self._base_entropy,
            entropy_after=entropy_after,
            delta=delta,
            is_inflection_point=is_inflection,
            affected_modules=[importer, imported],
            recommendation=recommendation
        )
    
    def analyze_add_module(
        self,
        module_name: str,
        imports: list[str],
        imported_by: list[str]
    ) -> DeltaAnalysis:
        """Analyze the entropy delta of adding a new module.
        
        Args:
            module_name: Name of the new module
            imports: Modules this module will import
            imported_by: Modules that will import this module
            
        Returns:
            DeltaAnalysis with results
        """
        if self._base_graph is None:
            self.load_base_graph()
        
        # Create a copy of the graph for simulation
        sim_graph = self._base_graph.graph.copy()
        
        # Add the new module node
        sim_graph.add_node(module_name, internal=True)
        
        # Add import edges
        for imported in imports:
            sim_graph.add_edge(module_name, imported)
        
        for importer in imported_by:
            sim_graph.add_edge(importer, module_name)
        
        # Calculate entropy after
        entropy_after = self._calculate_graph_entropy(sim_graph)
        
        # Calculate delta
        delta = entropy_after - self._base_entropy
        
        # Determine if inflection point
        is_inflection = abs(delta) > self.INFLECTION_THRESHOLD
        
        # Generate recommendation
        if delta > 0.2:
            recommendation = (
                "⚠️  New module adds significant coupling. Consider if this module "
                "should be split further or if imports can be reduced."
            )
        elif delta < -0.1:
            recommendation = (
                "✓ New module improves overall structure. Good candidate for "
                "extracting shared functionality."
            )
        else:
            recommendation = "Proceed - module addition has neutral entropy impact."
        
        return DeltaAnalysis(
            change_type="add_module",
            entropy_before=self._base_entropy,
            entropy_after=entropy_after,
            delta=delta,
            is_inflection_point=is_inflection,
            affected_modules=[module_name] + imports + imported_by,
            recommendation=recommendation
        )
    
    def analyze_extract_module(
        self,
        module_name: str,
        source_modules: list[str],
        functions_to_extract: list[str]
    ) -> DeltaAnalysis:
        """Analyze the entropy delta of extracting a new module.
        
        Args:
            module_name: Name of the new module
            source_modules: Modules to extract code from
            functions_to_extract: Functions/classes to move
            
        Returns:
            DeltaAnalysis with results
        """
        if self._base_graph is None:
            self.load_base_graph()
        
        # Create a copy of the graph for simulation
        sim_graph = self._base_graph.graph.copy()
        
        # Add the new module
        sim_graph.add_node(module_name, internal=True)
        
        # Add edges from source modules to the new module
        for source in source_modules:
            if sim_graph.has_node(source):
                sim_graph.add_edge(source, module_name)
        
        # Calculate entropy after
        entropy_after = self._calculate_graph_entropy(sim_graph)
        
        # Calculate delta
        delta = entropy_after - self._base_entropy
        
        # Determine if inflection point
        is_inflection = abs(delta) > self.INFLECTION_THRESHOLD
        
        # Generate recommendation
        if delta < -0.15:
            recommendation = (
                "✓ Extraction reduces coupling in source modules. "
                "Good refactoring candidate."
            )
        elif delta > 0.15:
            recommendation = (
                "⚠️  Extraction may add too many new dependencies. "
                "Consider smaller extractions or different boundaries."
            )
        else:
            recommendation = "Proceed - extraction has neutral entropy impact."
        
        return DeltaAnalysis(
            change_type="extract_module",
            entropy_before=self._base_entropy,
            entropy_after=entropy_after,
            delta=delta,
            is_inflection_point=is_inflection,
            affected_modules=[module_name] + source_modules,
            recommendation=recommendation
        )
    
    def _calculate_graph_entropy(self, graph: nx.DiGraph) -> float:
        """Calculate global entropy for a graph.
        
        Args:
            graph: NetworkX DiGraph
            
        Returns:
            Global entropy value
        """
        import math
        
        internal_nodes = [
            node for node, data in graph.nodes(data=True)
            if data.get('internal', False)
        ]
        
        if not internal_nodes:
            return 0.0
        
        entropies = []
        for node in internal_nodes:
            out_degree = graph.out_degree(node)
            in_degree = graph.in_degree(node)
            total_degree = out_degree + in_degree
            
            if total_degree == 0:
                entropies.append(0.0)
                continue
            
            # Calculate probabilities
            probabilities = []
            if out_degree > 0:
                probabilities.append(out_degree / total_degree)
            if in_degree > 0:
                probabilities.append(in_degree / total_degree)
            
            # Calculate Shannon entropy
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * math.log2(p)
            
            entropies.append(min(entropy, 1.0))
        
        return sum(entropies) / len(entropies)
    
    def get_inflection_candidates(
        self,
        proposed_changes: list[dict]
    ) -> list[DeltaAnalysis]:
        """Analyze multiple proposed changes and return inflection point candidates.
        
        Args:
            proposed_changes: List of change descriptions, e.g.:
                [{"type": "add_import", "importer": "a", "imported": "b"}, ...]
                
        Returns:
            List of DeltaAnalysis for inflection point changes only
        """
        candidates = []
        
        for change in proposed_changes:
            change_type = change.get("type")
            
            if change_type == "add_import":
                analysis = self.analyze_add_import(
                    change.get("importer"),
                    change.get("imported")
                )
            elif change_type == "remove_import":
                analysis = self.analyze_remove_import(
                    change.get("importer"),
                    change.get("imported")
                )
            elif change_type == "extract_module":
                analysis = self.analyze_extract_module(
                    change.get("module_name"),
                    change.get("source_modules", []),
                    change.get("functions_to_extract", [])
                )
            else:
                continue
            
            if analysis.is_inflection_point:
                candidates.append(analysis)
        
        return candidates


def analyze_change(
    project_root: Path | str,
    change_type: str,
    **kwargs
) -> DeltaAnalysis:
    """Convenience function to analyze a single change.
    
    Args:
        project_root: Root directory of the project
        change_type: Type of change
        **kwargs: Arguments specific to the change type
        
    Returns:
        DeltaAnalysis with results
    """
    calculator = EntropyDeltaCalculator(project_root)
    calculator.load_base_graph()
    
    if change_type == "add_import":
        return calculator.analyze_add_import(kwargs["importer"], kwargs["imported"])
    elif change_type == "remove_import":
        return calculator.analyze_remove_import(kwargs["importer"], kwargs["imported"])
    elif change_type == "add_module":
        return calculator.analyze_add_module(
            kwargs["module_name"],
            kwargs.get("imports", []),
            kwargs.get("imported_by", [])
        )
    elif change_type == "extract_module":
        return calculator.analyze_extract_module(
            kwargs["module_name"],
            kwargs.get("source_modules", []),
            kwargs.get("functions_to_extract", [])
        )
    else:
        raise ValueError(f"Unknown change type: {change_type}")
