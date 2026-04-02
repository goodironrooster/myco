# ⊕ H:0.20 | press:architecture | age:0 | drift:+0.00
"""MYCO Architecture Map - Project-wide architectural awareness.

Aggregates module manifests into a lightweight project-wide view.
Used by agents to understand the big picture.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .architecture import ModuleManifest, ModuleManifestManager


@dataclass
class LayerInfo:
    """Information about an architectural layer."""
    name: str
    level: int  # 1 = top (API), 2 = middle (services), 3 = bottom (models)
    description: str = ""
    modules: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    entropy_avg: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "description": self.description,
            "modules": self.modules,
            "depends_on": self.depends_on,
            "entropy_avg": self.entropy_avg,
        }


@dataclass
class CriticalPath:
    """A critical execution path through the architecture."""
    name: str
    path: List[str]  # List of module.component
    description: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
        }


@dataclass
class ArchitectureMap:
    """Project-wide architecture map.
    
    This is the HIGH-LEVEL view that agents load first to understand
    the project structure before diving into details.
    
    Size: ~10-20 KB (lightweight!)
    """
    project: str
    last_updated: str = ""
    
    # Layers (api, services, models, utils)
    layers: List[LayerInfo] = field(default_factory=list)
    
    # Critical execution paths
    critical_paths: List[CriticalPath] = field(default_factory=list)
    
    # Module summaries (just name and path, not full details)
    module_summaries: Dict[str, str] = field(default_factory=dict)
    
    # Overall entropy
    entropy_avg: float = 0.0
    
    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "project": self.project,
            "last_updated": self.last_updated,
            "layers": [l.to_dict() for l in self.layers],
            "critical_paths": [p.to_dict() for p in self.critical_paths],
            "module_summaries": self.module_summaries,
            "entropy_avg": self.entropy_avg,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureMap":
        """Create from dictionary."""
        layers = [
            LayerInfo(**l) if isinstance(l, dict) else l
            for l in data.get("layers", [])
        ]
        critical_paths = [
            CriticalPath(**p) if isinstance(p, dict) else p
            for p in data.get("critical_paths", [])
        ]
        return cls(
            project=data.get("project", ""),
            last_updated=data.get("last_updated", ""),
            layers=layers,
            critical_paths=critical_paths,
            module_summaries=data.get("module_summaries", {}),
            entropy_avg=data.get("entropy_avg", 0.0),
        )


class ArchitectureMapManager:
    """Manage the project-wide architecture map.
    
    This aggregates module manifests into a high-level view.
    """
    
    MAP_FILENAME = ".myco_architecture.json"
    
    # Standard layer definitions
    LAYER_DEFINITIONS = {
        "api": {"level": 1, "description": "HTTP API layer (FastAPI routes)"},
        "services": {"level": 2, "description": "Business logic layer"},
        "models": {"level": 3, "description": "Data models layer (SQLAlchemy)"},
        "utils": {"level": 3, "description": "Utility functions layer"},
        "config": {"level": 3, "description": "Configuration layer"},
    }
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.manifest_manager = ModuleManifestManager(project_root)
        self._map: Optional[ArchitectureMap] = None
    
    def load_map(self) -> Optional[ArchitectureMap]:
        """Load architecture map from disk."""
        map_path = self.project_root / self.MAP_FILENAME
        
        if not map_path.exists():
            return None
        
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._map = ArchitectureMap.from_dict(data)
        return self._map
    
    def generate_map(self) -> ArchitectureMap:
        """Generate architecture map from module manifests.
        
        This should be called after significant project changes.
        """
        # Load or create map
        self._map = ArchitectureMap(
            project=self.project_root.name,
        )
        
        # Collect all manifests
        manifests = self.manifest_manager.update_all_manifests()
        
        # Group by layer
        layer_modules: Dict[str, List[str]] = {}
        for manifest in manifests:
            layer = manifest.layer or self._infer_layer(manifest.module)
            if layer not in layer_modules:
                layer_modules[layer] = []
            layer_modules[layer].append(manifest.module)
            
            # Add to module summaries
            self._map.module_summaries[manifest.module] = manifest.path
        
        # Create layer info
        for layer_name, modules in sorted(layer_modules.items(), key=lambda x: self.LAYER_DEFINITIONS.get(x[0], {}).get("level", 99)):
            layer_def = self.LAYER_DEFINITIONS.get(layer_name, {})
            
            layer_info = LayerInfo(
                name=layer_name,
                level=layer_def.get("level", 99),
                description=layer_def.get("description", f"{layer_name} layer"),
                modules=sorted(modules),
            )
            
            # Determine layer dependencies
            layer_info.depends_on = self._get_layer_dependencies(manifests, layer_name)
            
            # Calculate average entropy
            entropies = [m.entropy_avg for m in manifests if m.layer == layer_name and m.entropy_avg > 0]
            if entropies:
                layer_info.entropy_avg = sum(entropies) / len(entropies)
            
            self._map.layers.append(layer_info)
        
        # Calculate overall entropy
        all_entropies = [m.entropy_avg for m in manifests if m.entropy_avg > 0]
        if all_entropies:
            self._map.entropy_avg = sum(all_entropies) / len(all_entropies)
        
        # Identify critical paths
        self._map.critical_paths = self._identify_critical_paths(manifests)
        
        # Save to disk
        self._save_map()
        
        return self._map
    
    def get_map(self) -> ArchitectureMap:
        """Get architecture map (loads or generates if needed)."""
        if self._map is None:
            self._map = self.load_map()
            if self._map is None:
                self._map = self.generate_map()
        return self._map
    
    def get_module_info(self, module_name: str) -> Optional[dict]:
        """Get detailed info about a module."""
        return self.manifest_manager.get_manifest(module_name).to_dict()
    
    def get_dependencies(self, module_name: str) -> List[str]:
        """Get modules this one depends on."""
        return self.manifest_manager.get_dependencies(module_name)
    
    def get_dependents(self, module_name: str) -> List[str]:
        """Get modules that depend on this one.
        
        CRITICAL for understanding impact of changes!
        """
        return self.manifest_manager.get_dependents(module_name)
    
    def get_component(self, module_name: str, component_name: str) -> Optional[dict]:
        """Get info about a specific component (class/function)."""
        component = self.manifest_manager.get_component(module_name, component_name)
        return component.to_dict() if component else None
    
    def _infer_layer(self, module_name: str) -> str:
        """Infer layer from module name."""
        if module_name in self.LAYER_DEFINITIONS:
            return module_name
        
        # Infer from name patterns
        if 'api' in module_name or 'route' in module_name or 'endpoint' in module_name:
            return "api"
        elif 'service' in module_name:
            return "services"
        elif 'model' in module_name or 'db' in module_name:
            return "models"
        elif 'util' in module_name or 'helper' in module_name:
            return "utils"
        
        return "services"  # Default
    
    def _get_layer_dependencies(self, manifests: List[ModuleManifest], layer_name: str) -> List[str]:
        """Get which layers this layer depends on."""
        depends_on = set()
        
        for manifest in manifests:
            if manifest.layer == layer_name:
                for dep in manifest.dependencies:
                    dep_layer = self._infer_layer(dep)
                    if dep_layer != layer_name:
                        depends_on.add(dep_layer)
        
        return sorted(depends_on)
    
    def _identify_critical_paths(self, manifests: List[ModuleManifest]) -> List[CriticalPath]:
        """Identify critical execution paths through the architecture."""
        paths = []
        
        # Common patterns
        # 1. API → Service → Model
        api_modules = [m.module for m in manifests if m.layer == "api"]
        service_modules = [m.module for m in manifests if m.layer == "services"]
        model_modules = [m.module for m in manifests if m.layer == "models"]
        
        if api_modules and service_modules and model_modules:
            # Find connected paths
            for api in api_modules[:3]:  # Limit to first 3
                api_manifest = self.manifest_manager.get_manifest(api)
                if api_manifest:
                    for dep in api_manifest.dependencies:
                        if dep in service_modules:
                            service_manifest = self.manifest_manager.get_manifest(dep)
                            if service_manifest:
                                for svc_dep in service_manifest.dependencies:
                                    if svc_dep in model_modules:
                                        paths.append(CriticalPath(
                                            name=f"{api} Flow",
                                            path=[f"{api}", f"{dep}", f"{svc_dep}"],
                                            description=f"Request flow: {api} → {dep} → {svc_dep}"
                                        ))
                                        break
        
        return paths
    
    def _save_map(self):
        """Save architecture map to disk."""
        map_path = self.project_root / self.MAP_FILENAME
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(self._map.to_dict(), f, indent=2)


# Agent tools
def load_architecture_map(project_root: str) -> Optional[dict]:
    """Load project architecture map (agent tool)."""
    manager = ArchitectureMapManager(Path(project_root))
    map_data = manager.get_map()
    return map_data.to_dict() if map_data else None


def get_arch_module_info(project_root: str, module_name: str) -> Optional[dict]:
    """Get detailed module info (agent tool)."""
    manager = ArchitectureMapManager(Path(project_root))
    return manager.get_module_info(module_name)


def get_arch_dependencies(project_root: str, module_name: str) -> List[str]:
    """Get module dependencies (agent tool)."""
    manager = ArchitectureMapManager(Path(project_root))
    return manager.get_dependencies(module_name)


def get_arch_dependents(project_root: str, module_name: str) -> List[str]:
    """Get modules that depend on this one (agent tool)."""
    manager = ArchitectureMapManager(Path(project_root))
    return manager.get_dependents(module_name)


def get_arch_component(project_root: str, module_name: str, component_name: str) -> Optional[dict]:
    """Get component info (agent tool)."""
    manager = ArchitectureMapManager(Path(project_root))
    return manager.get_component(module_name, component_name)


def update_architecture_map(project_root: str) -> Optional[dict]:
    """Update architecture map (agent tool)."""
    manager = ArchitectureMapManager(Path(project_root))
    map_data = manager.generate_map()
    return map_data.to_dict() if map_data else None


def get_refactoring_suggestion(project_root: str, file_path: str) -> Optional[dict]:
    """Get refactoring suggestion for a file (agent tool)."""
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
