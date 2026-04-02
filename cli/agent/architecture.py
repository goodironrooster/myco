# ⊕ H:0.25 | press:architecture | age:0 | drift:+0.00
"""MYCO Architecture Maps - Lightweight architectural awareness for agents.

MYCO Vision:
- Stigmergic architecture traces (in code, not separate DB)
- Auto-updated manifests (self-maintaining)
- Lazy-loaded for scalability
- Entropy-aware (track entropy per module)

Architecture:
- Module Manifests (.myco_manifest.json) - Per-directory
- Architecture Map (.myco_architecture.json) - Project-wide
- In-code annotations (# ⊕ comments)
"""

import ast
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any


@dataclass
class ComponentInfo:
    """Information about a component (class/function) in a module."""
    name: str
    type: str  # "class", "function", "service", "model", etc.
    file: str
    line_number: int = 0
    dependencies: List[str] = field(default_factory=list)
    responsibility: str = ""
    external_calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModuleManifest:
    """Manifest for a single module/directory.
    
    This is the core data structure for architectural awareness.
    Lightweight (~2-5 KB) and auto-updated.
    """
    module: str  # Module name (e.g., "services", "api")
    path: str  # Relative path from project root
    description: str = ""
    layer: str = ""  # "api", "services", "models", "utils"
    
    # Entropy tracking
    entropy_avg: float = 0.0
    entropy_regime: str = "unknown"  # "crystallized", "dissipative", "diffuse"
    
    # Components in this module
    components: List[ComponentInfo] = field(default_factory=list)
    
    # Dependencies (other modules this one depends on)
    dependencies: List[str] = field(default_factory=list)
    
    # Dependents (modules that depend on this one)
    dependents: List[str] = field(default_factory=list)
    
    # External dependencies (third-party libraries)
    external_dependencies: List[Dict[str, str]] = field(default_factory=list)
    
    # Metadata
    file_count: int = 0
    last_updated: str = ""
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "module": self.module,
            "path": self.path,
            "description": self.description,
            "layer": self.layer,
            "entropy_avg": self.entropy_avg,
            "entropy_regime": self.entropy_regime,
            "components": [c.to_dict() for c in self.components],
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "external_dependencies": self.external_dependencies,
            "file_count": self.file_count,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModuleManifest":
        """Create from dictionary."""
        components = [
            ComponentInfo(**c) if isinstance(c, dict) else c
            for c in data.get("components", [])
        ]
        return cls(
            module=data.get("module", ""),
            path=data.get("path", ""),
            description=data.get("description", ""),
            layer=data.get("layer", ""),
            entropy_avg=data.get("entropy_avg", 0.0),
            entropy_regime=data.get("entropy_regime", "unknown"),
            components=components,
            dependencies=data.get("dependencies", []),
            dependents=data.get("dependents", []),
            external_dependencies=data.get("external_dependencies", []),
            file_count=data.get("file_count", 0),
            last_updated=data.get("last_updated", ""),
        )


class ModuleManifestManager:
    """Manage module manifests - create, load, update, save.
    
    This is the WORKHORSE class that agents will use actively.
    """
    
    MANIFEST_FILENAME = ".myco_manifest.json"
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self._cache: Dict[str, ModuleManifest] = {}
    
    def get_manifest(self, module_path: str) -> Optional[ModuleManifest]:
        """Get manifest for a module (loads from cache or disk).
        
        Args:
            module_path: Path to module directory (relative or absolute)
            
        Returns:
            ModuleManifest or None if not found
        """
        # Check cache first
        if module_path in self._cache:
            return self._cache[module_path]
        
        # Try to load from disk
        manifest_path = self._get_manifest_path(module_path)
        if manifest_path.exists():
            manifest = self._load_manifest(manifest_path)
            self._cache[module_path] = manifest
            return manifest
        
        return None
    
    def create_or_update_manifest(self, module_path: str) -> ModuleManifest:
        """Create new manifest or update existing one.
        
        This is called by the agent AFTER creating/modifying files.
        
        Args:
            module_path: Path to module directory
            
        Returns:
            Updated ModuleManifest
        """
        module_path_obj = self._resolve_module_path(module_path)
        
        # Analyze the module
        manifest = self._analyze_module(module_path_obj)
        
        # Save to disk
        self._save_manifest(manifest, module_path_obj)
        
        # Update cache
        self._cache[str(module_path_obj)] = manifest
        
        return manifest
    
    def get_component(self, module_path: str, component_name: str) -> Optional[ComponentInfo]:
        """Get info about a specific component in a module.
        
        Args:
            module_path: Path to module
            component_name: Name of class/function
            
        Returns:
            ComponentInfo or None
        """
        manifest = self.get_manifest(module_path)
        if not manifest:
            return None
        
        for component in manifest.components:
            if component.name == component_name:
                return component
        
        return None
    
    def get_dependencies(self, module_path: str) -> List[str]:
        """Get list of modules this one depends on.
        
        Args:
            module_path: Path to module
            
        Returns:
            List of module paths
        """
        manifest = self.get_manifest(module_path)
        return manifest.dependencies if manifest else []
    
    def get_dependents(self, module_path: str) -> List[str]:
        """Get list of modules that depend on this one.
        
        This is CRITICAL for understanding impact of changes.
        
        Args:
            module_path: Path to module
            
        Returns:
            List of module paths
        """
        manifest = self.get_manifest(module_path)
        return manifest.dependents if manifest else []
    
    def _get_manifest_path(self, module_path: str) -> Path:
        """Get path to manifest file."""
        module_path_obj = self._resolve_module_path(module_path)
        return module_path_obj / self.MANIFEST_FILENAME
    
    def _resolve_module_path(self, module_path: str) -> Path:
        """Resolve module path to absolute Path."""
        p = Path(module_path)
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()
    
    def _load_manifest(self, manifest_path: Path) -> ModuleManifest:
        """Load manifest from JSON file."""
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return ModuleManifest.from_dict(data)
    
    def _save_manifest(self, manifest: ModuleManifest, module_path: Path):
        """Save manifest to JSON file."""
        manifest_path = module_path / self.MANIFEST_FILENAME
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest.to_dict(), f, indent=2)
    
    def _analyze_module(self, module_path: Path) -> ModuleManifest:
        """Analyze a module directory and create manifest.
        
        This does static analysis of Python files to extract:
        - Classes and functions
        - Import dependencies
        - External dependencies
        """
        module_name = module_path.name
        manifest = ModuleManifest(
            module=module_name,
            path=str(module_path.relative_to(self.project_root)),
            description=f"{module_name} module",
        )
        
        # Find all Python files
        py_files = list(module_path.glob("*.py"))
        manifest.file_count = len(py_files)
        
        # Track all imports for dependency analysis
        all_imports: Set[str] = set()
        external_deps: Set[str] = set()
        
        # Analyze each Python file
        for py_file in py_files:
            if py_file.name.startswith('_') and py_file.name != '__init__.py':
                continue
            
            # Parse file and extract components
            components = self._analyze_file(py_file, module_path)
            manifest.components.extend(components)
            
            # Extract imports
            imports, externals = self._extract_imports(py_file)
            all_imports.update(imports)
            external_deps.update(externals)
        
        # Determine dependencies (internal modules)
        manifest.dependencies = self._determine_dependencies(all_imports, module_name)
        
        # Set external dependencies
        manifest.external_dependencies = [
            {"name": dep, "purpose": "external library"}
            for dep in sorted(external_deps)
        ]
        
        # Calculate entropy (if MYCO entropy module available)
        self._calculate_entropy(manifest, module_path)
        
        return manifest
    
    def _analyze_file(self, file_path: Path, module_path: Path) -> List[ComponentInfo]:
        """Analyze a single Python file and extract components."""
        components = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse AST
            tree = ast.parse(source)
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    component = self._extract_class_info(node, file_path, module_path, source)
                    components.append(component)
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    # Top-level functions only
                    component = self._extract_function_info(node, file_path, module_path, source)
                    components.append(component)
        
        except Exception as e:
            # If parsing fails, create minimal component from file
            components.append(ComponentInfo(
                name=file_path.stem,
                type="module",
                file=str(file_path.relative_to(module_path)),
            ))
        
        return components
    
    def _extract_class_info(self, node: ast.ClassDef, file_path: Path, module_path: Path, source: str) -> ComponentInfo:
        """Extract information from a class definition."""
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Extract responsibility from docstring
        responsibility = self._extract_responsibility(docstring)
        
        # Extract dependencies from method calls
        dependencies = self._extract_method_dependencies(node)
        
        # Extract external calls (e.g., stripe.PaymentIntent)
        external_calls = self._extract_external_calls(node)
        
        # Parse in-code annotations
        annotations = self._parse_annotations(source, node.lineno)
        
        return ComponentInfo(
            name=node.name,
            type="class",
            file=str(file_path.relative_to(module_path)),
            line_number=node.lineno,
            dependencies=dependencies,
            responsibility=responsibility or annotations.get("responsibility", ""),
            external_calls=external_calls,
            called_by=annotations.get("called_by", []),
        )
    
    def _extract_function_info(self, node: ast.FunctionDef, file_path: Path, module_path: Path, source: str) -> ComponentInfo:
        """Extract information from a function definition."""
        docstring = ast.get_docstring(node) or ""
        responsibility = self._extract_responsibility(docstring)
        
        # Parse annotations
        annotations = self._parse_annotations(source, node.lineno)
        
        return ComponentInfo(
            name=node.name,
            type="function",
            file=str(file_path.relative_to(module_path)),
            line_number=node.lineno,
            responsibility=responsibility or annotations.get("responsibility", ""),
        )
    
    def _extract_responsibility(self, docstring: str) -> str:
        """Extract responsibility from docstring first line."""
        if not docstring:
            return ""
        
        first_line = docstring.strip().split('\n')[0].strip()
        # Remove common prefixes
        for prefix in ["Args:", "Returns:", "Raises:", ":param", ":return"]:
            if first_line.startswith(prefix):
                return ""
        
        return first_line[:200]  # Limit length
    
    def _extract_method_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Extract dependencies from method calls in a class."""
        dependencies = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    # e.g., self.db.query()
                    if isinstance(child.func.value, ast.Attribute):
                        dependencies.add(child.func.value.attr)
        
        return sorted(dependencies)
    
    def _extract_external_calls(self, node: ast.ClassDef) -> List[str]:
        """Extract external library calls (e.g., stripe.PaymentIntent)."""
        external = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute):
                    # Check if it's module.Class pattern
                    if isinstance(child.func.value, ast.Name):
                        module_name = child.func.value.id
                        # Common external libraries
                        if module_name in ['stripe', 'sendgrid', 'boto3', 'redis', 'requests']:
                            external.add(f"{module_name}.{child.func.attr}")
        
        return sorted(external)
    
    def _extract_imports(self, file_path: Path) -> tuple:
        """Extract imports from a Python file.
        
        Returns:
            Tuple of (internal_imports, external_imports)
        """
        internal = set()
        external = set()
        
        # Known external libraries
        EXTERNAL_LIBS = {
            'stripe', 'sendgrid', 'boto3', 'redis', 'requests', 'sqlalchemy',
            'fastapi', 'pydantic', 'uvicorn', 'pytest', 'httpx'
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        if module in EXTERNAL_LIBS:
                            external.add(module)
                        else:
                            internal.add(module)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module.split('.')[0]
                        if module in EXTERNAL_LIBS:
                            external.add(module)
                        else:
                            internal.add(module)
        
        except Exception:
            pass
        
        return internal, external
    
    def _determine_dependencies(self, imports: Set[str], current_module: str) -> List[str]:
        """Determine which internal modules this one depends on."""
        # Known internal modules
        INTERNAL_MODULES = {'api', 'services', 'models', 'utils', 'config'}
        
        dependencies = []
        for imp in imports:
            if imp in INTERNAL_MODULES and imp != current_module:
                dependencies.append(imp)
        
        return sorted(dependencies)
    
    def _parse_annotations(self, source: str, line_number: int) -> Dict[str, Any]:
        """Parse MYCO in-code annotations near a component.
        
        Looks for comments like:
        # ⊕ Responsibility: Process payments
        # ⊕ Called by: api.orders
        # ⊕ External: stripe.PaymentIntent
        """
        annotations = {}
        lines = source.split('\n')
        
        # Look at lines before the component
        start = max(0, line_number - 5)
        for i in range(start, line_number):
            if i < len(lines):
                line = lines[i].strip()
                if line.startswith('# ⊕'):
                    # Parse annotation
                    parts = line[3:].split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower().replace(' ', '_')
                        value = parts[1].strip()
                        
                        # Handle list values (e.g., "Called by: api.orders, api.users")
                        if ',' in value:
                            annotations[key] = [v.strip() for v in value.split(',')]
                        else:
                            annotations[key] = value
        
        return annotations
    
    def _calculate_entropy(self, manifest: ModuleManifest, module_path: Path):
        """Calculate entropy for the module if MYCO entropy module is available."""
        try:
            # Try to import MYCO entropy
            import sys
            myco_path = self.project_root / "myco"
            if myco_path.exists():
                sys.path.insert(0, str(self.project_root))
                from myco.entropy import ImportGraphBuilder, EntropyCalculator
                
                # Build import graph
                builder = ImportGraphBuilder(self.project_root)
                builder.scan()
                
                # Calculate average entropy
                calculator = EntropyCalculator(builder)
                entropies = []
                
                for component in manifest.components:
                    try:
                        # Get module name from file
                        module_name = component.file.replace('/', '.').replace('.py', '')
                        if module_name in calculator._graph.nodes():
                            h = calculator.calculate_module_entropy(module_name)
                            entropies.append(h)
                    except Exception:
                        pass
                
                if entropies:
                    manifest.entropy_avg = sum(entropies) / len(entropies)
                    manifest.entropy_regime = self._get_regime(manifest.entropy_avg)
        
        except Exception:
            # If entropy calculation fails, use defaults
            manifest.entropy_avg = 0.5
            manifest.entropy_regime = "dissipative"
    
    def _get_regime(self, h: float) -> str:
        """Get entropy regime from H value."""
        if h < 0.3:
            return "crystallized"
        elif h <= 0.75:
            return "dissipative"
        else:
            return "diffuse"
    
    def update_all_manifests(self) -> List[ModuleManifest]:
        """Update manifests for all modules in the project.
        
        Call this when project structure changes significantly.
        
        Returns:
            List of updated manifests
        """
        manifests = []
        
        # Find all directories with Python files
        for py_file in self.project_root.rglob("*.py"):
            module_dir = py_file.parent
            
            # Skip hidden directories and __pycache__
            if '__pycache__' in str(module_dir) or module_dir.name.startswith('.'):
                continue
            
            # Create/update manifest
            manifest = self.create_or_update_manifest(str(module_dir))
            manifests.append(manifest)
        
        return manifests


# Convenience functions for agent tools
def get_module_info(project_root: str, module_path: str) -> Optional[dict]:
    """Get information about a module (agent tool)."""
    manager = ModuleManifestManager(Path(project_root))
    manifest = manager.get_manifest(module_path)
    return manifest.to_dict() if manifest else None


def get_dependencies(project_root: str, module_path: str) -> List[str]:
    """Get module dependencies (agent tool)."""
    manager = ModuleManifestManager(Path(project_root))
    return manager.get_dependencies(module_path)


def get_dependents(project_root: str, module_path: str) -> List[str]:
    """Get modules that depend on this one (agent tool)."""
    manager = ModuleManifestManager(Path(project_root))
    return manager.get_dependents(module_path)


def update_manifest(project_root: str, module_path: str) -> Optional[dict]:
    """Update manifest after creating/modifying files (agent tool)."""
    manager = ModuleManifestManager(Path(project_root))
    manifest = manager.create_or_update_manifest(module_path)
    return manifest.to_dict()


def get_refactoring_suggestion(project_root: str, file_path: str) -> Optional[dict]:
    """Get refactoring suggestion for a file (agent tool).
    
    Args:
        project_root: Project root directory
        file_path: Path to file to analyze
        
    Returns:
        Dict with refactoring suggestion or None
    """
    try:
        from myco.entropy import get_refactoring_suggestion as entropy_refactor
        
        # Get file content
        full_path = Path(project_root) / file_path
        if not full_path.exists():
            return None
        
        source = full_path.read_text(encoding='utf-8')
        
        # Get suggestion
        suggestion = entropy_refactor(str(full_path), source_code=source)
        
        return suggestion
        
    except Exception as e:
        return None


# ============================================================================
# DEPENDENCY TRACKING (Phase 1.4)
# ============================================================================

class DependencyTracker:
    """Track dependencies between Python modules.
    
    MYCO Phase 1.4: Know what breaks when you change a file.
    """
    
    _instance: Optional['DependencyTracker'] = None
    
    def __init__(self):
        # file → set of files it depends on
        self._dependencies: Dict[str, Set[str]] = {}
        # file → set of files that depend on it
        self._dependents: Dict[str, Set[str]] = {}
    
    @classmethod
    def get_instance(cls) -> 'DependencyTracker':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def extract_dependencies(self, source_code: str, file_path: str) -> Dict[str, Any]:
        """Extract import dependencies from Python source.
        
        Args:
            source_code: Python source code
            file_path: Path to source file
            
        Returns:
            Dict with dependency information
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return {"imports": [], "from_imports": {}, "local_deps": [], "external_deps": []}
        
        imports = []
        from_imports = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module
                    names = [alias.name for alias in node.names]
                    from_imports[module] = names
        
        # Determine local vs external dependencies
        file_dir = Path(file_path).parent
        local_deps = []
        external_deps = []
        
        # Known external libraries
        EXTERNAL_LIBS = {
            'fastapi', 'sqlalchemy', 'pydantic', 'requests', 'pytest',
            'numpy', 'pandas', 'redis', 'boto3', 'stripe', 'sendgrid'
        }
        
        all_imports = imports + list(from_imports.keys())
        
        for imp in all_imports:
            # Check if it's a local module
            base_module = imp.split('.')[0]
            
            if base_module in EXTERNAL_LIBS:
                external_deps.append(imp)
            else:
                # Try to find local module
                local_path = file_dir / f"{base_module}.py"
                local_init = file_dir / base_module / "__init__.py"
                
                if local_path.exists() or local_init.exists():
                    local_deps.append(imp)
                else:
                    # Assume external if not found locally
                    external_deps.append(imp)
        
        return {
            "imports": imports,
            "from_imports": from_imports,
            "local_deps": local_deps,
            "external_deps": external_deps
        }
    
    def update(self, file_path: str, dependencies: List[str]):
        """Update dependency info for a file.
        
        Args:
            file_path: Path to file
            dependencies: List of files this one depends on
        """
        file_path = str(file_path)
        deps_set = set(dependencies)
        
        # Remove old dependencies
        if file_path in self._dependencies:
            old_deps = self._dependencies[file_path]
            for old_dep in old_deps:
                if old_dep in self._dependents:
                    self._dependents[old_dep].discard(file_path)
        
        # Set new dependencies
        self._dependencies[file_path] = deps_set
        
        # Update dependents
        for dep in deps_set:
            if dep not in self._dependents:
                self._dependents[dep] = set()
            self._dependents[dep].add(file_path)
    
    def get_dependents(self, file_path: str) -> List[str]:
        """Get files that depend on this file.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of files that depend on this one
        """
        file_path = str(file_path)
        return list(self._dependents.get(file_path, set()))
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """Get files this file depends on.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of files this one depends on
        """
        file_path = str(file_path)
        return list(self._dependencies.get(file_path, set()))
    
    def get_affected_files(self, changed_files: List[str]) -> Dict[str, List[str]]:
        """Get all files affected by changes to multiple files.
        
        Args:
            changed_files: List of changed file paths
            
        Returns:
            Dict: changed_file → [affected files]
        """
        affected = {}
        
        for file_path in changed_files:
            dependents = self.get_dependents(file_path)
            if dependents:
                affected[file_path] = dependents
        
        return affected


# Global dependency tracker instance
_dependency_tracker: Optional[DependencyTracker] = None


def get_dependency_tracker() -> DependencyTracker:
    """Get global dependency tracker instance."""
    global _dependency_tracker
    if _dependency_tracker is None:
        _dependency_tracker = DependencyTracker()
    return _dependency_tracker


def track_dependencies(file_path: str, source_code: str) -> Dict[str, Any]:
    """Track dependencies for a file.
    
    MYCO Phase 1.4: Track what depends on what.
    
    Args:
        file_path: Path to file
        source_code: Source code
        
    Returns:
        Dict with dependency info and affected files
    """
    tracker = get_dependency_tracker()
    
    # Extract dependencies
    deps_info = tracker.extract_dependencies(source_code, file_path)
    
    # Convert module names to file paths
    file_dir = Path(file_path).parent
    dep_files = []
    
    for local_dep in deps_info["local_deps"]:
        base_module = local_dep.split('.')[0]
        dep_path = file_dir / f"{base_module}.py"
        if dep_path.exists():
            dep_files.append(str(dep_path))
    
    # Update tracker
    tracker.update(file_path, dep_files)
    
    # Get affected files
    affected = tracker.get_dependents(file_path)
    
    return {
        "file": file_path,
        "dependencies": dep_files,
        "local_deps": deps_info["local_deps"],
        "external_deps": deps_info["external_deps"],
        "affected_files": affected
    }


def get_affected_files(project_root: str, file_path: str) -> List[str]:
    """Get files affected by changes to this file (agent tool).
    
    Args:
        project_root: Project root directory
        file_path: Path to file
        
    Returns:
        List of affected file paths
    """
    tracker = get_dependency_tracker()
    return tracker.get_dependents(file_path)


def get_dependencies(project_root: str, file_path: str) -> List[str]:
    """Get files this file depends on (agent tool).
    
    Args:
        project_root: Project root directory
        file_path: Path to file
        
    Returns:
        List of dependency file paths
    """
    tracker = get_dependency_tracker()
    return tracker.get_dependencies(file_path)
