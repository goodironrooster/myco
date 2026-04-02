# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Entropy calculation for myco.

Builds the import graph using AST walking (static analysis only).
Computes Shannon entropy per module and a global baseline.
"""

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx


@dataclass
class ModuleInfo:
    """Information about a module in the import graph."""
    path: Path
    name: str
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)
    out_degree: int = 0
    in_degree: int = 0
    entropy: float = 0.0


class ImportGraphBuilder:
    """Builds an import graph from Python source files using AST."""
    
    def __init__(self, root_path: Path | str):
        """Initialize with a root path.
        
        Args:
            root_path: Root directory to scan for Python files
        """
        self.root_path = Path(root_path)
        self.modules: dict[str, ModuleInfo] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
    
    def _path_to_module_name(self, path: Path) -> str:
        """Convert a file path to a module name."""
        try:
            rel_path = path.relative_to(self.root_path)
            parts = list(rel_path.parts)
            
            # Remove .py extension from last part
            if parts and parts[-1].endswith('.py'):
                parts[-1] = parts[-1][:-3]
            
            # Handle __init__.py
            if parts and parts[-1] == '__init__':
                parts = parts[:-1]
            
            return '.'.join(parts) if parts else self.root_path.name
        except ValueError:
            return path.stem
    
    def _extract_imports(self, source: str) -> list[str]:
        """Extract import statements from source code using AST.
        
        Args:
            source: Python source code
            
        Returns:
            List of imported module names
        """
        imports = []
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return imports
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    
    def scan(self) -> "ImportGraphBuilder":
        """Scan the root directory for Python files and build the import graph.
        
        Returns:
            Self for method chaining
        """
        python_files = list(self.root_path.rglob("*.py"))
        
        # First pass: collect all modules
        for py_file in python_files:
            # Skip hidden directories and common non-source directories
            if any(part.startswith('.') for part in py_file.parts):
                continue
            if any(part in {'__pycache__', 'venv', '.venv', 'env', '.env'} for part in py_file.parts):
                continue
            
            module_name = self._path_to_module_name(py_file)
            self.modules[module_name] = ModuleInfo(path=py_file, name=module_name)
        
        # Create set of internal module names for quick lookup
        internal_modules = set(self.modules.keys())
        
        # Second pass: extract imports and build edges
        for module_name, module_info in self.modules.items():
            try:
                source = module_info.path.read_text(encoding="utf-8")
                imports = self._extract_imports(source)
                module_info.imports = imports
                
                # Add edges to graph
                for imp in imports:
                    # Check if this is an internal import (starts with any internal module)
                    is_internal = imp in internal_modules or any(
                        imp.startswith(m + '.') or m.startswith(imp + '.')
                        for m in internal_modules
                    )
                    
                    # Add node for imports
                    if imp not in self.graph.nodes:
                        self.graph.add_node(imp, internal=is_internal)
                    
                    self.graph.add_edge(module_name, imp)
                    module_info.out_degree += 1
                    
                    # Track reverse dependencies for internal modules only
                    if is_internal and imp in self.modules:
                        self.modules[imp].imported_by.append(module_name)
                        self.modules[imp].in_degree += 1
                        
            except (IOError, UnicodeDecodeError):
                continue
        
        return self
    
    def get_internal_graph(self) -> nx.DiGraph:
        """Get the subgraph of only internal modules.
        
        Returns:
            NetworkX DiGraph with only internal modules
        """
        internal_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('internal', False)
        ]
        return self.graph.subgraph(internal_nodes).copy()


class EntropyCalculator:
    """Calculates Shannon entropy for modules in the import graph."""
    
    def __init__(self, import_graph: ImportGraphBuilder):
        """Initialize with an import graph builder.
        
        Args:
            import_graph: ImportGraphBuilder instance
        """
        self.import_graph = import_graph
        self.graph = import_graph.get_internal_graph()
    
    def calculate_module_entropy(self, module_name: str) -> float:
        """Calculate Shannon entropy for a single module.
        
        H(m) = -Σ p(e) · log₂(p(e)) for all edges e incident to module m
        
        Where p(e) is the normalized out-degree distribution.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Shannon entropy value (0.0 to 1.0 normalized)
        """
        if module_name not in self.graph:
            return 0.0
        
        # Get out-degree and in-degree
        out_degree = self.graph.out_degree(module_name)
        in_degree = self.graph.in_degree(module_name)
        total_degree = out_degree + in_degree
        
        if total_degree == 0:
            return 0.0
        
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
        
        # Normalize to 0-1 range (max entropy for binary distribution is 1.0)
        return min(entropy, 1.0)
    
    def calculate_global_entropy(self) -> float:
        """Calculate the global entropy baseline for the entire codebase.
        
        Returns:
            Average entropy across all internal modules
        """
        if not self.graph.nodes():
            return 0.0
        
        entropies = [
            self.calculate_module_entropy(node)
            for node in self.graph.nodes()
        ]
        
        return sum(entropies) / len(entropies) if entropies else 0.0
    
    def get_module_regimes(self) -> dict[str, str]:
        """Classify modules by their entropy regime.
        
        Returns:
            Dict mapping module names to regime strings:
            - 'crystallized' (H < 0.3): rigid, over-coupled, brittle
            - 'dissipative' (0.3 ≤ H ≤ 0.75): absorbs pressure gracefully
            - 'diffuse' (H > 0.75): under-coupled, boundary-less
        """
        regimes = {}
        
        for node in self.graph.nodes():
            entropy = self.calculate_module_entropy(node)
            
            if entropy < 0.3:
                regimes[node] = "crystallized"
            elif entropy <= 0.75:
                regimes[node] = "dissipative"
            else:
                regimes[node] = "diffuse"
        
        return regimes
    
    def get_modules_by_deviation(
        self,
        baseline: Optional[float] = None,
        top_n: int = 5
    ) -> list[tuple[str, float, float]]:
        """Get modules with highest deviation from baseline.
        
        Args:
            baseline: Baseline entropy (uses global if not provided)
            top_n: Number of modules to return
            
        Returns:
            List of (module_name, entropy, drift) tuples sorted by |drift|
        """
        if baseline is None:
            baseline = self.calculate_global_entropy()
        
        deviations = []
        for node in self.graph.nodes():
            entropy = self.calculate_module_entropy(node)
            drift = entropy - baseline
            deviations.append((node, entropy, drift))
        
        # Sort by absolute drift descending
        deviations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return deviations[:top_n]


@dataclass
class EntropyReport:
    """Report on the entropy state of the codebase."""
    global_entropy: float
    module_count: int
    crystallized: list[str]
    dissipative: list[str]
    diffuse: list[str]
    top_deviations: list[tuple[str, float, float]]

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Entropy Report",
            f"==============",
            f"Global entropy: {self.global_entropy:.3f}",
            f"Modules analyzed: {self.module_count}",
            f"",
            f"Regime distribution:",
            f"  Crystallized (H < 0.3): {len(self.crystallized)}",
            f"  Dissipative (0.3-0.75): {len(self.dissipative)}",
            f"  Diffuse (H > 0.75): {len(self.diffuse)}",
        ]

        if self.crystallized:
            lines.append(f"  Crystallized modules: {', '.join(self.crystallized[:5])}")
        if self.diffuse:
            lines.append(f"  Diffuse modules: {', '.join(self.diffuse[:5])}")

        if self.top_deviations:
            lines.append("")
            lines.append("Top deviations from baseline:")
            for name, entropy, drift in self.top_deviations[:5]:
                lines.append(f"  {name}: H={entropy:.3f}, drift={drift:+.3f}")

        return '\n'.join(lines)


@dataclass
class GradientEdge:
    """Represents an import edge with entropy gradient."""
    importer: str
    imported: str
    H_importer: float
    H_imported: float
    gradient: float
    is_fault_line: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "importer": self.importer,
            "imported": self.imported,
            "H_importer": self.H_importer,
            "H_imported": self.H_imported,
            "gradient": self.gradient,
            "is_fault_line": self.is_fault_line,
        }


@dataclass
class ModuleStress:
    """Structural stress on a module from gradient field."""
    module: str
    H: float
    edge_count: int
    mean_gradient: float
    max_gradient: float
    fault_line_count: int
    fault_lines: list[str]  # List of connected module names via fault lines

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "module": self.module,
            "H": self.H,
            "edge_count": self.edge_count,
            "mean_gradient": self.mean_gradient,
            "max_gradient": self.max_gradient,
            "fault_line_count": self.fault_line_count,
            "fault_lines": self.fault_lines,
        }


@dataclass
class GradientFieldReport:
    """Report on the gradient field of a codebase."""
    fault_lines: list[GradientEdge]
    module_stress: list[ModuleStress]
    total_edges: int
    fault_line_count: int
    mean_gradient: float
    threshold: float = 0.3

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Gradient Field Report",
            f"=====================",
            f"Threshold: {self.threshold}",
            f"Total edges: {self.total_edges}",
            f"Fault lines: {self.fault_line_count}",
            f"Mean gradient: {self.mean_gradient:.3f}",
            f"",
        ]

        if self.fault_lines:
            lines.append("FAULT LINES (structural stress concentrations):")
            for edge in self.fault_lines[:10]:  # Show top 10
                lines.append(
                    f"  {edge.importer} (H={edge.H_importer:.2f}) → "
                    f"{edge.imported} (H={edge.H_imported:.2f}) | "
                    f"gradient={edge.gradient:.2f}"
                )
            lines.append("")

        if self.module_stress:
            lines.append("HIGHEST STRESS MODULES:")
            stressed = sorted(self.module_stress, key=lambda m: m.mean_gradient, reverse=True)[:5]
            for module in stressed:
                if module.fault_line_count > 0:
                    lines.append(
                        f"  {module.module}: stress={module.mean_gradient:.2f} "
                        f"({module.fault_line_count} fault lines, {module.edge_count} edges)"
                    )

        return '\n'.join(lines)


def analyze_entropy(root_path: Path | str) -> EntropyReport:
    """Analyze the entropy of a codebase.

    Args:
        root_path: Root directory to analyze

    Returns:
        EntropyReport with analysis results
    """
    root_path = Path(root_path)

    # Build import graph
    builder = ImportGraphBuilder(root_path)
    builder.scan()

    # Calculate entropy
    calculator = EntropyCalculator(builder)
    global_entropy = calculator.calculate_global_entropy()
    regimes = calculator.get_module_regimes()
    deviations = calculator.get_modules_by_deviation(global_entropy)
    
    # Categorize modules
    crystallized = [m for m, r in regimes.items() if r == "crystallized"]
    dissipative = [m for m, r in regimes.items() if r == "dissipative"]
    diffuse = [m for m, r in regimes.items() if r == "diffuse"]

    return EntropyReport(
        global_entropy=global_entropy,
        module_count=len(builder.modules),
        crystallized=crystallized,
        dissipative=dissipative,
        diffuse=diffuse,
        top_deviations=deviations
    )


def get_priority_files(root_path: Path | str, top_n: int = 5) -> list[dict]:
    """Get files prioritized by entropy regime for agent attention.
    
    Priority order:
    1. Crystallized modules (H < 0.3) - need decomposition
    2. High drift modules - changing rapidly
    3. Diffuse modules (H > 0.75) - may need consolidation
    
    Args:
        root_path: Root directory to analyze
        top_n: Number of priority files to return
        
    Returns:
        List of dicts with file info and priority reason
    """
    root_path = Path(root_path)
    report = analyze_entropy(root_path)
    
    priority_files = []
    
    # Priority 1: Crystallized modules (need decomposition)
    for module in report.crystallized[:top_n]:
        priority_files.append({
            "file": module,
            "priority": 1,
            "reason": "crystallized",
            "action_hint": "Consider decompose or interface_inversion"
        })
    
    # Priority 2: High drift modules (changing rapidly)
    for name, entropy, drift in report.top_deviations[:top_n]:
        if abs(drift) > 0.1:  # Significant drift
            # Avoid duplicates
            if not any(f["file"] == name for f in priority_files):
                priority_files.append({
                    "file": name,
                    "priority": 2,
                    "reason": "high_drift",
                    "drift": drift,
                    "action_hint": "Monitor for instability"
                })
    
    # Priority 3: Diffuse modules (may need consolidation)
    for module in report.diffuse[:top_n]:
        if not any(f["file"] == module for f in priority_files):
            priority_files.append({
                "file": module,
                "priority": 3,
                "reason": "diffuse",
                "action_hint": "Consider compression_collapse"
            })
    
    # Sort by priority
    priority_files.sort(key=lambda x: x["priority"])
    
    return priority_files[:top_n]


def get_regime_intervention(file_path: Path | str, H: float) -> dict:
    """Get the recommended intervention for a file based on its entropy regime.
    
    From the myco principles:
    - Crystallized (H < 0.3): Apply decompose or interface_inversion
    - Dissipative (0.3 ≤ H ≤ 0.75): Safe to make changes, preserve this range
    - Diffuse (H > 0.75): Apply compression_collapse or tension_extraction
    
    Args:
        file_path: Path to the file
        H: Current entropy value
        
    Returns:
        Dict with regime, recommended interventions, and guidance
    """
    file_path = Path(file_path)
    
    if H < 0.3:
        return {
            "file": str(file_path),
            "regime": "crystallized",
            "H": H,
            "interventions": ["decompose", "interface_inversion"],
            "primary": "decompose",
            "guidance": (
                "This module is rigid and over-coupled. "
                "Do NOT add features directly. First apply decompose to break it into smaller units, "
                "or interface_inversion to redesign from the implementer's perspective. "
                "Only after restructuring should you consider the original task."
            ),
            "warning": "Adding features to crystallized modules increases brittleness."
        }
    elif H <= 0.75:
        return {
            "file": str(file_path),
            "regime": "dissipative",
            "H": H,
            "interventions": ["none"],
            "primary": "none",
            "guidance": (
                "This module is in a healthy dissipative regime. "
                "It absorbs pressure gracefully. Proceed with the task while monitoring entropy. "
                "Keep changes minimal to preserve this regime."
            ),
            "warning": None
        }
    else:
        return {
            "file": str(file_path),
            "regime": "diffuse",
            "H": H,
            "interventions": ["compression_collapse", "tension_extraction"],
            "primary": "compression_collapse",
            "guidance": (
                "This module is under-coupled and lacks boundaries. "
                "Consider consolidation before adding new functionality. "
                "Apply compression_collapse to consolidate related functionality, "
                "or tension_extraction to extract shared interfaces."
            ),
            "warning": "Module lacks structural boundaries - consolidation recommended."
        }


def analyze_file_regime(root_path: Path | str, file_path: Path | str) -> dict:
    """Analyze the entropy regime of a specific file.
    
    Args:
        root_path: Project root directory
        file_path: File to analyze

    Returns:
        Dict with regime analysis and intervention recommendations
    """
    root_path = Path(root_path)
    file_path = Path(file_path)

    # Build import graph
    builder = ImportGraphBuilder(root_path)
    builder.scan()

    # Get module name
    module_name = builder._path_to_module_name(file_path)

    # Calculate entropy
    calculator = EntropyCalculator(builder)
    H = calculator.calculate_module_entropy(module_name)

    # Get regime intervention
    intervention = get_regime_intervention(file_path, H)

    return intervention


def calculate_substrate_health(root_path: Path | str) -> dict:
    """Calculate overall substrate health score.
    
    Health is based on:
    - Entropy distribution (more dissipative = healthier)
    - Crystallized module count (fewer = healthier)
    - Diffuse module count (fewer = healthier)
    - Entropy trend (negative/improving = healthier)
    
    Args:
        root_path: Project root directory
        
    Returns:
        Dict with health score and breakdown
    """
    root_path = Path(root_path)
    report = analyze_entropy(root_path)
    
    # Calculate component scores (0-1 scale)
    
    # 1. Entropy distribution score
    # Perfect: all modules in dissipative regime (0.3-0.75)
    total_modules = report.module_count or 1  # Avoid division by zero
    dissipative_ratio = len(report.dissipative) / total_modules
    entropy_score = dissipative_ratio
    
    # 2. Crystallized module penalty
    # More than 20% crystallized is concerning
    crystallized_ratio = len(report.crystallized) / total_modules
    crystallized_score = max(0, 1 - (crystallized_ratio * 5))  # 20% = 0 score
    
    # 3. Diffuse module penalty
    # More than 20% diffuse is concerning
    diffuse_ratio = len(report.diffuse) / total_modules
    diffuse_score = max(0, 1 - (diffuse_ratio * 5))  # 20% = 0 score
    
    # 4. Overall entropy level (0.5 is ideal)
    ideal_entropy = 0.5
    entropy_deviation = abs(report.global_entropy - ideal_entropy)
    entropy_level_score = 1 - entropy_deviation  # 0.5 = 1.0 score
    
    # Calculate weighted average
    health_score = (
        entropy_score * 0.3 +
        crystallized_score * 0.25 +
        diffuse_score * 0.25 +
        entropy_level_score * 0.2
    )
    
    # Determine health status
    if health_score >= 0.8:
        status = "healthy"
        status_message = "Substrate is in good health"
    elif health_score >= 0.6:
        status = "stable"
        status_message = "Substrate is stable with minor issues"
    elif health_score >= 0.4:
        status = "degraded"
        status_message = "Substrate needs attention"
    else:
        status = "critical"
        status_message = "Substrate requires immediate restructuring"
    
    return {
        "health_score": round(health_score, 2),
        "status": status,
        "status_message": status_message,
        "breakdown": {
            "entropy_distribution": round(entropy_score, 2),
            "crystallized_score": round(crystallized_score, 2),
            "diffuse_score": round(diffuse_score, 2),
            "entropy_level": round(entropy_level_score, 2),
        },
        "metrics": {
            "total_modules": total_modules,
            "crystallized_count": len(report.crystallized),
            "dissipative_count": len(report.dissipative),
            "diffuse_count": len(report.diffuse),
            "global_entropy": round(report.global_entropy, 3),
        }
    }


def get_related_files(root_path: Path | str, target_file: Path | str, max_files: int = 5) -> list[dict]:
    """Get files related to a target file via the import graph.
    
    Uses the import graph to find:
    1. Files that the target imports (dependencies)
    2. Files that import the target (dependents)
    
    Args:
        root_path: Project root directory
        target_file: Target file to find relations for
        max_files: Maximum number of related files to return
        
    Returns:
        List of dicts with file path and relationship type
    """
    root_path = Path(root_path)
    target_file = Path(target_file)
    
    if not target_file.exists():
        return []
    
    # Build import graph
    builder = ImportGraphBuilder(root_path)
    builder.scan()
    
    # Get target module name
    target_module = builder._path_to_module_name(target_file)
    
    related = []
    
    # Find files this module imports (dependencies)
    if target_module in builder.modules:
        module_info = builder.modules[target_module]
        for imp in module_info.imports:
            # Check if it's an internal import
            if imp in builder.modules:
                imp_path = builder.modules[imp].path
                if imp_path.exists() and imp_path != target_file:
                    related.append({
                        "file": str(imp_path.relative_to(root_path)),
                        "relationship": "imports",
                        "module": imp
                    })
        
        # Find files that import this module (dependents)
        for dep in module_info.imported_by:
            if dep in builder.modules:
                dep_path = builder.modules[dep].path
                if dep_path.exists() and dep_path != target_file:
                    related.append({
                        "file": str(dep_path.relative_to(root_path)),
                        "relationship": "imported_by",
                        "module": dep
                    })
    
    # Sort and limit results
    related = related[:max_files]
    
    return related


def read_related_content(root_path: Path | str, related_files: list[dict], max_content_length: int = 1000) -> list[dict]:
    """Read content of related files for multi-file context.
    
    Args:
        root_path: Project root directory
        related_files: List of related file dicts from get_related_files()
        max_content_length: Maximum content length per file
        
    Returns:
        List of dicts with file path, relationship, and content preview
    """
    root_path = Path(root_path)
    content_list = []

    for rel in related_files:
        file_path = root_path / rel["file"]

        if not file_path.exists():
            continue

        try:
            content = file_path.read_text(encoding="utf-8")

            # Truncate if too long
            if len(content) > max_content_length:
                content = content[:max_content_length] + "\n... (truncated)"

            content_list.append({
                "file": rel["file"],
                "relationship": rel["relationship"],
                "module": rel.get("module", ""),
                "content": content
            })
        except (IOError, UnicodeDecodeError):
            continue

    return content_list


def compute_function_size_entropy(tree: ast.AST) -> float:
    """Compute Shannon entropy over function sizes in a module.
    
    Args:
        tree: AST of the module
        
    Returns:
        Shannon entropy of function size distribution (0-1 normalized)
    """
    # Collect function sizes (in AST node count)
    sizes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Count nodes in function body
            size = sum(1 for _ in ast.walk(node))
            sizes.append(size)
    
    if not sizes or len(sizes) < 2:
        return 0.5  # Default for modules with 0-1 functions
    
    # Compute distribution
    total = sum(sizes)
    probabilities = [s / total for s in sizes]
    
    # Compute Shannon entropy
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by max possible entropy (uniform distribution)
    max_entropy = math.log2(len(sizes))
    
    return entropy / max_entropy if max_entropy > 0 else 0.5


def compute_nesting_depth_entropy(tree: ast.AST) -> float:
    """Compute entropy over maximum nesting depths of functions.
    
    Args:
        tree: AST of the module
        
    Returns:
        Shannon entropy of nesting depth distribution (0-1 normalized)
    """
    def get_max_depth(node: ast.AST, current_depth: int = 0) -> int:
        """Get maximum nesting depth in a node."""
        max_depth = current_depth
        nesting_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.With, 
                        ast.AsyncFor, ast.AsyncWith)
        
        for child in ast.iter_child_nodes(node):
            if isinstance(node, nesting_nodes):
                child_depth = get_max_depth(child, current_depth + 1)
            else:
                child_depth = get_max_depth(child, current_depth)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    # Collect max depths for each function
    depths = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            max_depth = get_max_depth(node)
            depths.append(max_depth)
    
    if not depths or len(depths) < 2:
        return 0.5  # Default
    
    # Compute distribution
    total = sum(depths) + 1  # +1 to avoid division by zero
    probabilities = [d / total for d in depths]
    
    # Compute Shannon entropy
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize
    max_entropy = math.log2(len(depths))
    
    return entropy / max_entropy if max_entropy > 0 else 0.5


def compute_name_cohesion(tree: ast.AST) -> float:
    """Compute name cohesion score based on identifier frequency.
    
    High cohesion = low entropy (reuses same vocabulary)
    Low cohesion = high entropy (many unrelated concepts)
    
    Args:
        tree: AST of the module
        
    Returns:
        Cohesion score (0-1, where 1 = high cohesion, 0 = low cohesion)
    """
    # Collect all identifiers
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
    
    if not names:
        return 0.5  # Default
    
    # Compute frequency distribution
    freq = {}
    for name in names:
        freq[name] = freq.get(name, 0) + 1
    
    total = len(names)
    probabilities = [count / total for count in freq.values()]
    
    # Compute Shannon entropy
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    # Normalize by max possible entropy
    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
    
    # Convert entropy to cohesion (inverse relationship)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    cohesion = 1 - normalized_entropy  # High entropy = low cohesion
    
    return cohesion


def compute_internal_entropy(filepath: Path | str) -> dict:
    """Compute internal entropy metrics for a module.
    
    Measures:
    - Function size entropy
    - Nesting depth entropy
    - Name cohesion
    
    Args:
        filepath: Path to Python file
        
    Returns:
        Dict with H_function_size, H_nesting, cohesion, and H_internal
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return {
            "H_function_size": 0.5,
            "H_nesting": 0.5,
            "cohesion": 0.5,
            "H_internal": 0.5
        }
    
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (IOError, SyntaxError):
        return {
            "H_function_size": 0.5,
            "H_nesting": 0.5,
            "cohesion": 0.5,
            "H_internal": 0.5
        }
    
    # Compute individual metrics
    H_function_size = compute_function_size_entropy(tree)
    H_nesting = compute_nesting_depth_entropy(tree)
    cohesion = compute_name_cohesion(tree)
    
    # Combine into single H_internal score
    # Weight: 40% function size, 30% nesting, 30% cohesion (inverted)
    H_internal = (
        H_function_size * 0.4 +
        H_nesting * 0.3 +
        (1 - cohesion) * 0.3
    )
    
    return {
        "H_function_size": round(H_function_size, 3),
        "H_nesting": round(H_nesting, 3),
        "cohesion": round(cohesion, 3),
        "H_internal": round(H_internal, 3)
    }


def classify_dual_regime(H_structural: float, H_internal: float) -> dict:
    """Classify module regime using both structural and internal entropy.
    
    Args:
        H_structural: Import graph entropy (0-1)
        H_internal: Internal structure entropy (0-1)
        
    Returns:
        Dict with regime classification and guidance
    """
    # Define thresholds
    LOW = 0.3
    HIGH = 0.75
    
    # Classify each dimension
    if H_structural < LOW:
        structural_regime = "crystallized"
    elif H_structural <= HIGH:
        structural_regime = "dissipative"
    else:
        structural_regime = "diffuse"

    if H_internal < LOW:
        internal_regime = "ordered"
    elif H_internal <= HIGH:
        internal_regime = "balanced"
    else:
        internal_regime = "chaotic"

    # Combined interpretation
    if structural_regime == "crystallized" and internal_regime == "ordered":
        combined = "crystallized"
        guidance = "Rigid externally and internally. Decompose into smaller units."
        priority = 1
    elif structural_regime == "crystallized" and internal_regime == "chaotic":
        combined = "crystallized_chaotic"
        guidance = "Coupled externally, chaotic internally. Extract and reorganize."
        priority = 1
    elif structural_regime == "diffuse" and internal_regime == "chaotic":
        combined = "diffuse"
        guidance = "Diffuse everywhere. Consolidate related functionality."
        priority = 1
    elif structural_regime == "diffuse" and internal_regime == "ordered":
        combined = "diffuse_ordered"
        guidance = "Loosely coupled but internally clean. May need better integration."
        priority = 2
    elif structural_regime == "dissipative" and internal_regime == "balanced":
        combined = "dissipative"
        guidance = "Healthy. Safe to modify."
        priority = 3
    else:
        combined = "mixed"
        guidance = "Mixed signals. Review manually."
        priority = 2

    return {
        "combined_regime": combined,
        "structural_regime": structural_regime,
        "internal_regime": internal_regime,
        "H_structural": round(H_structural, 3),
        "H_internal": round(H_internal, 3),
        "guidance": guidance,
        "priority": priority
    }


def compute_gradient_field(
    root_path: Path | str,
    threshold: float = 0.3
) -> GradientFieldReport:
    """Compute gradient field across import graph.

    Structural stress concentrates at boundaries between modules in different regimes.
    A gradient > threshold indicates a "fault line" where structural pressure builds.

    Args:
        root_path: Project root directory
        threshold: Gradient threshold for fault line detection (default 0.3)

    Returns:
        GradientFieldReport with fault lines and module stress data

    Example:
        >>> report = compute_gradient_field(".")
        >>> print(report.summary())
    """
    root_path = Path(root_path)

    # Build import graph and compute entropy
    builder = ImportGraphBuilder(root_path)
    builder.scan()
    calculator = EntropyCalculator(builder)

    # Get internal graph (only internal modules)
    graph = builder.get_internal_graph()

    # Compute entropy for all modules
    module_entropy = {
        node: calculator.calculate_module_entropy(node)
        for node in graph.nodes()
    }

    # Compute gradient for all edges
    edges: list[GradientEdge] = []
    for importer, imported in graph.edges():
        H_importer = module_entropy.get(importer, 0.0)
        H_imported = module_entropy.get(imported, 0.0)
        gradient = abs(H_importer - H_imported)
        is_fault_line = gradient > threshold

        edges.append(GradientEdge(
            importer=importer,
            imported=imported,
            H_importer=H_importer,
            H_imported=H_imported,
            gradient=gradient,
            is_fault_line=is_fault_line
        ))

    # Identify fault lines
    fault_lines = [e for e in edges if e.is_fault_line]

    # Compute module stress for each module
    module_stress: list[ModuleStress] = []
    for module in graph.nodes():
        # Get all edges connected to this module
        module_edges = [
            e for e in edges
            if e.importer == module or e.imported == module
        ]

        if not module_edges:
            continue

        gradients = [e.gradient for e in module_edges]
        fault_line_edges = [e for e in module_edges if e.is_fault_line]

        # Get connected modules via fault lines
        fault_line_modules = []
        for e in fault_line_edges:
            if e.importer == module:
                fault_line_modules.append(e.imported)
            else:
                fault_line_modules.append(e.importer)

        module_stress.append(ModuleStress(
            module=module,
            H=module_entropy.get(module, 0.0),
            edge_count=len(module_edges),
            mean_gradient=sum(gradients) / len(gradients),
            max_gradient=max(gradients),
            fault_line_count=len(fault_line_edges),
            fault_lines=fault_line_modules
        ))

    # Compute overall statistics
    total_edges = len(edges)
    all_gradients = [e.gradient for e in edges]
    mean_gradient = sum(all_gradients) / len(all_gradients) if all_gradients else 0.0

    return GradientFieldReport(
        fault_lines=fault_lines,
        module_stress=module_stress,
        total_edges=total_edges,
        fault_line_count=len(fault_lines),
        mean_gradient=mean_gradient,
        threshold=threshold
    )


def get_fault_line_modules(
    root_path: Path | str,
    threshold: float = 0.3,
    top_n: int = 5
) -> list[dict]:
    """Get modules involved in fault lines, sorted by stress.

    Args:
        root_path: Project root directory
        threshold: Gradient threshold for fault line detection
        top_n: Number of modules to return

    Returns:
        List of module dicts with stress data
    """
    report = compute_gradient_field(root_path, threshold)

    # Filter to modules with fault lines
    stressed = [m for m in report.module_stress if m.fault_line_count > 0]

    # Sort by mean gradient (stress)
    stressed.sort(key=lambda m: m.mean_gradient, reverse=True)

    # Return top N as dicts
    return [m.to_dict() for m in stressed[:top_n]]


def calculate_entropy_from_content(source_code: str) -> float:
    """Calculate entropy from source code content (without file path).
    
    This is useful for checking entropy of proposed code before writing it.
    
    Uses simplified heuristics based on:
    - Number of classes
    - Number of functions
    - Lines of code
    - Import count
    - Cyclomatic complexity estimate
    
    Args:
        source_code: Python source code as string
        
    Returns:
        Entropy value (0.0 to 1.0)
        - < 0.3: Crystallized (too simple/rigid)
        - 0.3-0.75: Dissipative (healthy)
        - > 0.75: Diffuse (too complex/scattered)
    """
    if not source_code or not source_code.strip():
        return 0.0
    
    lines = source_code.split('\n')
    total_lines = len(lines)
    
    # Skip empty files
    if total_lines < 5:
        return 0.1
    
    # Count code elements
    num_classes = source_code.count('class ')
    num_functions = source_code.count('def ')
    num_imports = source_code.count('import ')
    num_methods = source_code.count('    def ')  # Indented def = methods
    
    # Estimate complexity
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    non_empty_lines = len(code_lines)
    
    # Calculate metrics
    if num_classes > 0:
        methods_per_class = num_methods / num_classes
    else:
        methods_per_class = 0
    
    if num_functions > 0:
        lines_per_function = non_empty_lines / num_functions
    else:
        lines_per_function = non_empty_lines
    
    # Entropy calculation based on multiple factors
    entropy_components = []
    
    # 1. Size factor (larger files tend to have higher entropy)
    size_factor = min(non_empty_lines / 200, 1.0)  # Cap at 200 lines
    entropy_components.append(size_factor * 0.3)
    
    # 2. Complexity factor (many classes/functions = higher entropy)
    complexity_score = (num_classes * 3 + num_functions * 1.5 + num_methods * 0.5) / max(non_empty_lines / 15, 1)
    complexity_factor = min(complexity_score, 1.0)
    entropy_components.append(complexity_factor * 0.35)
    
    # 3. Coupling factor (many imports = higher entropy)
    coupling_factor = min(num_imports / 10, 1.0)  # Cap at 10 imports
    entropy_components.append(coupling_factor * 0.2)
    
    # 4. Organization factor (methods per class, lines per function)
    if methods_per_class > 10 or lines_per_function > 50:
        organization_factor = 0.8  # Poor organization
    elif methods_per_class > 5 or lines_per_function > 25:
        organization_factor = 0.5  # Moderate organization
    else:
        organization_factor = 0.2  # Good organization
    entropy_components.append(organization_factor * 0.2)
    
    # Combine components
    entropy = sum(entropy_components)
    
    # Normalize to 0-1 range
    entropy = min(max(entropy, 0.0), 1.0)
    
    return entropy


def get_refactoring_suggestion(file_path: str, H: float = None, source_code: str = None) -> dict:
    """Get refactoring suggestion based on entropy regime.
    
    Args:
        file_path: Path to file
        H: Entropy value (calculated if not provided)
        source_code: Source code (used to calculate H if not provided)
        
    Returns:
        Dict with refactoring suggestion:
        {
            "regime": "crystallized" | "dissipative" | "diffuse",
            "action": "refactor" | "monitor" | "none",
            "priority": "high" | "medium" | "low",
            "pattern": "decompose" | "interface_inversion" | "compression_collapse" | "extract_module",
            "reason": "...",
            "example": "..."
        }
    """
    # Calculate H if not provided
    if H is None:
        if source_code:
            H = calculate_entropy_from_content(source_code)
        elif Path(file_path).exists():
            try:
                source = Path(file_path).read_text(encoding='utf-8')
                H = calculate_entropy_from_content(source)
            except Exception:
                H = 0.5
        else:
            H = 0.5
    
    # Determine regime and suggestion
    if H < 0.3:
        return {
            "regime": "crystallized",
            "action": "refactor",
            "priority": "medium",
            "pattern": "decompose",
            "reason": f"File has low entropy (H={H:.2f}). Code is rigid and hard to extend.",
            "example": """
# Before: Single monolithic function
def process_order(order):
    # 100 lines of validation, calculation, database, email...
    pass

# After: Decomposed into focused functions
def process_order(order):
    validate_order(order)
    calculate_total(order)
    save_order(order)
    send_confirmation(order)
"""
        }
    
    elif H <= 0.75:
        return {
            "regime": "dissipative",
            "action": "none",
            "priority": "low",
            "pattern": "maintain",
            "reason": f"File has healthy entropy (H={H:.2f}). Code is well-organized.",
            "example": "# No refactoring needed - maintain current structure"
        }
    
    else:  # H > 0.75
        return {
            "regime": "diffuse",
            "action": "refactor",
            "priority": "high",
            "pattern": "compression_collapse" if H > 0.85 else "extract_module",
            "reason": f"File has high entropy (H={H:.2f}). Code lacks boundaries and is hard to maintain.",
            "example": """
# Before: God class with 20+ methods
class OrderService:
    def method1... method20...

# After: Extracted into focused services
class OrderCreationService: ...
class OrderValidationService: ...
class OrderNotificationService: ...
"""
        }


def get_refactoring_patterns() -> dict:
    """Get all available refactoring patterns with descriptions.
    
    Returns:
        Dict of pattern name → description
    """
    return {
        "decompose": {
            "name": "Decompose God Function",
            "when": "Single function does too much (H < 0.3)",
            "action": "Split into smaller, focused functions",
            "example": """
# Before
def process_order(order):
    # 100 lines doing everything
    pass

# After
def process_order(order):
    validate(order)
    calculate(order)
    save(order)
    notify(order)
"""
        },
        "interface_inversion": {
            "name": "Interface Inversion",
            "when": "Tight coupling to concrete implementations",
            "action": "Depend on abstractions, not concretions",
            "example": """
# Before
def process(db: SQLiteDatabase):
    db.query(...)

# After
class IDatabase(Protocol):
    def query(self, sql: str): ...

def process(db: IDatabase):
    db.query(...)
"""
        },
        "compression_collapse": {
            "name": "Compression Collapse",
            "when": "Too many scattered functions (H > 0.85)",
            "action": "Consolidate related functions into classes",
            "example": """
# Before: Scattered functions
def validate_email(email): ...
def validate_password(pw): ...
def validate_user(user): ...
def create_user(email, pw): ...
def delete_user(user_id): ...

# After: Consolidated service
class UserService:
    def validate_email(self, email): ...
    def validate_password(self, pw): ...
    def create(self, email, pw): ...
    def delete(self, user_id): ...
"""
        },
        "extract_module": {
            "name": "Extract Module",
            "when": "File has too many responsibilities (H > 0.75)",
            "action": "Split file into focused modules",
            "example": """
# Before: orders.py does everything
class Order: ...
class OrderService: ...
class OrderValidator: ...
class OrderNotifier: ...

# After: Extracted modules
# orders/models.py
class Order: ...

# orders/services.py
class OrderService: ...

# orders/validation.py
class OrderValidator: ...

# orders/notifications.py
class OrderNotifier: ...
"""
        },
        "maintain": {
            "name": "Maintain",
            "when": "Code is healthy (0.3 ≤ H ≤ 0.75)",
            "action": "Continue current practices",
            "example": "# No refactoring needed"
        }
    }


def get_regime_from_content(source_code: str) -> str:
    """Get entropy regime from source code content.
    
    Args:
        source_code: Python source code as string
        
    Returns:
        Regime string: "crystallized", "dissipative", or "diffuse"
    """
    h = calculate_entropy_from_content(source_code)
    
    if h < 0.3:
        return "crystallized"
    elif h <= 0.75:
        return "dissipative"
    else:
        return "diffuse"


def check_entropy_budget(
    current_source: str,
    proposed_source: str,
    max_delta: float = 0.15,
    max_new_file_H: float = 0.50
) -> tuple[bool, float, float, str]:
    """Check if a code change is within entropy budget.
    
    Args:
        current_source: Current source code (empty if new file)
        proposed_source: Proposed new source code
        max_delta: Maximum allowed entropy increase (default 0.15)
        max_new_file_H: Maximum entropy for new files (default 0.50)
        
    Returns:
        Tuple of (within_budget, current_H, proposed_H, message)
    """
    is_new_file = not current_source or not current_source.strip()
    current_H = calculate_entropy_from_content(current_source) if current_source else 0.0
    proposed_H = calculate_entropy_from_content(proposed_source)
    delta_H = proposed_H - current_H
    
    if is_new_file:
        # New files have a higher threshold (can start at moderate complexity)
        if proposed_H > max_new_file_H:
            return (
                False,
                current_H,
                proposed_H,
                f"New file entropy (H={proposed_H:.2f}) exceeds maximum ({max_new_file_H}). Consider splitting into smaller modules."
            )
        return (
            True,
            current_H,
            proposed_H,
            f"New file entropy (H={proposed_H:.2f}) acceptable. Regime: {get_regime_from_content(proposed_source)}"
        )
    else:
        # Modifications have stricter delta limit
        if delta_H > max_delta:
            return (
                False,
                current_H,
                proposed_H,
                f"Entropy increase ({delta_H:.2f}) exceeds budget ({max_delta}). Consider splitting into smaller modules."
            )
        
        return (
            True,
            current_H,
            proposed_H,
            f"Entropy change ({delta_H:+.2f}) within budget. New H={proposed_H:.2f} ({get_regime_from_content(proposed_source)})"
        )
