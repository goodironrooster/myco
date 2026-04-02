# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""World model manager for myco.

Reads .myco/world.json at session start.
Writes it at session end.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class WorldModel:
    """Persistent world model for myco.

    This is the agent's memory across sessions.
    Stored in .myco/world.json at the project root.

    Attributes:
        schema_version: Version of the schema (for migrations)
        last_session: ISO timestamp of last session
        session_count: Number of sessions completed
        entropy_baseline: Global entropy at last session
        entropy_trend: Change in entropy since previous session
        crystallized_modules: Modules with H < 0.3
        diffuse_modules: Modules with H > 0.75
        active_attractors: Currently detected attractor basins
        last_press_type: Last structural intervention applied
        tensegrity_violations: Count of tension/compression violations
        open_intentions: Structural pressures to address
        # Step 3: Self-entropy tracking (myco's own substrate)
        self_entropy_baseline: float = 0.50  # myco/ directory entropy
        self_entropy_trend: float = 0.00  # Change in myco/ entropy
        self_crystallized_count: int = 0  # Count of crystallized modules in myco/
    """
    schema_version: int = 1
    last_session: str = ""
    session_count: int = 0
    entropy_baseline: float = 0.50
    entropy_trend: float = 0.00
    crystallized_modules: list[str] = field(default_factory=list)
    diffuse_modules: list[str] = field(default_factory=list)
    active_attractors: list[str] = field(default_factory=list)
    last_press_type: str = "none"
    tensegrity_violations: int = 0
    open_intentions: list[str] = field(default_factory=list)
    # Step 3: Self-entropy tracking
    self_entropy_baseline: float = 0.50
    self_entropy_trend: float = 0.00
    self_crystallized_count: int = 0

    _path: Optional[Path] = field(default=None, repr=False)
    
    @classmethod
    def load(cls, project_root: Path | str) -> "WorldModel":
        """Load the world model from disk.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            WorldModel instance (bootstrap if file doesn't exist)
        """
        project_root = Path(project_root)
        world_path = project_root / ".myco" / "world.json"
        
        if not world_path.exists():
            # Bootstrap new world model
            world = cls(
                last_session=datetime.utcnow().isoformat() + "Z",
                _path=world_path
            )
            world.save()
            return world
        
        try:
            data = json.loads(world_path.read_text(encoding="utf-8"))
            
            # Handle schema migrations if needed
            schema_version = data.get("schema_version", 1)
            if schema_version < cls.schema_version:
                data = cls._migrate(data, schema_version)
            
            return cls(
                schema_version=data.get("schema_version", 1),
                last_session=data.get("last_session", ""),
                session_count=data.get("session_count", 0),
                entropy_baseline=data.get("entropy_baseline", 0.50),
                entropy_trend=data.get("entropy_trend", 0.00),
                crystallized_modules=data.get("crystallized_modules", []),
                diffuse_modules=data.get("diffuse_modules", []),
                active_attractors=data.get("active_attractors", []),
                last_press_type=data.get("last_press_type", "none"),
                tensegrity_violations=data.get("tensegrity_violations", 0),
                open_intentions=data.get("open_intentions", []),
                _path=world_path
            )
        except (json.JSONDecodeError, IOError) as e:
            # Corrupted file - bootstrap new one
            print(f"Warning: Could not load world.json: {e}. Bootstrapping new world model.")
            world = cls(
                last_session=datetime.utcnow().isoformat() + "Z",
                _path=world_path
            )
            world.save()
            return world
    
    @staticmethod
    def _migrate(data: dict, from_version: int) -> dict:
        """Migrate world model data from older schema version.
        
        Args:
            data: Current data dict
            from_version: Schema version to migrate from
            
        Returns:
            Migrated data dict
        """
        # Add migration logic here as schema evolves
        # For now, just return data as-is
        return data
    
    def save(self) -> None:
        """Save the world model to disk."""
        if self._path is None:
            raise ValueError("Cannot save world model without a path")
        
        # Ensure directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "schema_version": self.schema_version,
            "last_session": self.last_session,
            "session_count": self.session_count,
            "entropy_baseline": self.entropy_baseline,
            "entropy_trend": self.entropy_trend,
            "crystallized_modules": self.crystallized_modules,
            "diffuse_modules": self.diffuse_modules,
            "active_attractors": self.active_attractors,
            "last_press_type": self.last_press_type,
            "tensegrity_violations": self.tensegrity_violations,
            "open_intentions": self.open_intentions,
        }
        
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    
    def start_session(self) -> None:
        """Mark the start of a new session."""
        self.session_count += 1
        self.last_session = datetime.utcnow().isoformat() + "Z"
    
    def end_session(
        self,
        entropy_baseline: float,
        crystallized: list[str],
        diffuse: list[str],
        tensegrity_violations: int = 0
    ) -> None:
        """Mark the end of a session with updated metrics.
        
        Args:
            entropy_baseline: New global entropy
            crystallized: List of crystallized modules
            diffuse: List of diffuse modules
            tensegrity_violations: Count of violations
        """
        # Calculate entropy trend
        old_baseline = self.entropy_baseline
        self.entropy_trend = entropy_baseline - old_baseline
        self.entropy_baseline = entropy_baseline
        self.crystallized_modules = crystallized
        self.diffuse_modules = diffuse
        self.tensegrity_violations = tensegrity_violations
        
        self.save()
    
    def record_press(self, press_type: str) -> None:
        """Record a structural intervention (press) type.
        
        Args:
            press_type: Type of intervention applied
        """
        self.last_press_type = press_type
        self.save()
    
    def add_intention(self, intention: str) -> None:
        """Add an open intention (structural pressure to address).
        
        Args:
            intention: Description of the pressure
        """
        if intention not in self.open_intentions:
            self.open_intentions.append(intention)
            self.save()
    
    def resolve_intention(self, intention: str) -> bool:
        """Remove a resolved intention.
        
        Args:
            intention: Description to match and remove
            
        Returns:
            True if intention was found and removed
        """
        for i, existing in enumerate(self.open_intentions):
            if intention.lower() in existing.lower() or existing.lower() in intention.lower():
                self.open_intentions.pop(i)
                self.save()
                return True
        return False
    
    def add_attractor(self, attractor_name: str) -> None:
        """Record an active attractor basin.
        
        Args:
            attractor_name: Name of the attractor
        """
        if attractor_name not in self.active_attractors:
            self.active_attractors.append(attractor_name)
            self.save()
    
    def clear_attractors(self) -> None:
        """Clear all active attractors."""
        self.active_attractors = []
        self.save()
    
    def to_context_dict(self) -> dict:
        """Convert world model to context contract format.

        Returns:
            Dict for injection into session context
        """
        return {
            "schema_version": self.schema_version,
            "last_session": self.last_session,
            "session_count": self.session_count,
            "entropy_baseline": self.entropy_baseline,
            "entropy_trend": self.entropy_trend,
            "crystallized_modules": self.crystallized_modules,
            "diffuse_modules": self.diffuse_modules,
            "active_attractors": self.active_attractors,
            "last_press_type": self.last_press_type,
            "tensegrity_violations": self.tensegrity_violations,
            "open_intentions": self.open_intentions,
            # Step 3: Self-entropy tracking
            "self_entropy_baseline": self.self_entropy_baseline,
            "self_entropy_trend": self.self_entropy_trend,
            "self_crystallized_count": self.self_crystallized_count,
        }

    def update_self_entropy(self, baseline: float, crystallized_count: int) -> None:
        """Update self-entropy metrics (myco's own substrate).

        Args:
            baseline: New self entropy baseline
            crystallized_count: Count of crystallized modules in myco/
        """
        old_baseline = self.self_entropy_baseline
        self.self_entropy_trend = baseline - old_baseline
        self.self_entropy_baseline = baseline
        self.self_crystallized_count = crystallized_count
        self.save()
    
    def __str__(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "World Model",
            "===========",
            f"Sessions: {self.session_count}",
            f"Last session: {self.last_session}",
            f"Entropy baseline: {self.entropy_baseline:.3f}",
            f"Entropy trend: {self.entropy_trend:+.3f}",
            f"Crystallized modules: {len(self.crystallized_modules)}",
            f"Diffuse modules: {len(self.diffuse_modules)}",
            f"Tensegrity violations: {self.tensegrity_violations}",
            f"Active attractors: {len(self.active_attractors)}",
            f"Last press type: {self.last_press_type}",
        ]
        
        if self.open_intentions:
            lines.append("")
            lines.append("Open intentions:")
            for intention in self.open_intentions:
                lines.append(f"  - {intention}")
        
        return '\n'.join(lines)
