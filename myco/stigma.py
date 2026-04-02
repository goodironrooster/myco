# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Stigmergic annotation system for myco.

Reads and writes ⊕ annotations using Python's ast module.
Never uses regex for parsing code structure.

Annotation storage:
- Primary: .myco/annotations.json (sidecar file with history)
- Fallback: Source file comments (backward compatibility)
"""

import ast
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Annotation pattern - only for reading existing annotations
ANNOTATION_PATTERN = re.compile(
    r'^#\s*⊕\s*H:([\d.]+)\s*\|\s*press:(\w+)\s*\|\s*age:(\d+)\s*\|\s*drift:([+-][\d.]+)'
)


@dataclass
class StigmergicAnnotation:
    """A stigmergic annotation on a source file.

    Attributes:
        H: Shannon coupling entropy (0.00-1.00)
        press: Last intervention type (decompose, interface_inversion, tension_extraction,
               compression_collapse, entropy_drain, attractor_escape)
        age: Sessions since last touch (0 = touched this session)
        drift: Delta between H at write time and H now (+ means coupling grew, - means simplified)
    """
    H: float = 0.50
    press: str = "none"
    age: int = 0
    drift: float = 0.00

    VALID_PRESS_TYPES = {
        "decompose",
        "interface_inversion",
        "tension_extraction",
        "compression_collapse",
        "entropy_drain",
        "attractor_escape",
        "none"
    }

    def __post_init__(self):
        """Validate annotation values."""
        if not 0.0 <= self.H <= 1.0:
            raise ValueError(f"H must be between 0.0 and 1.0, got {self.H}")
        if self.press not in self.VALID_PRESS_TYPES:
            raise ValueError(f"press must be one of {self.VALID_PRESS_TYPES}, got {self.press}")

    def format(self) -> str:
        """Format annotation as a comment string."""
        drift_str = f"{self.drift:+.2f}"
        return f"# ⊕ H:{self.H:.2f} | press:{self.press} | age:{self.age} | drift:{drift_str}"

    @classmethod
    def parse(cls, line: str) -> Optional["StigmergicAnnotation"]:
        """Parse annotation from a comment line.

        Args:
            line: A line that may contain a stigmergic annotation

        Returns:
            StigmergicAnnotation if found, None otherwise
        """
        match = ANNOTATION_PATTERN.match(line.strip())
        if not match:
            return None

        try:
            return cls(
                H=float(match.group(1)),
                press=match.group(2),
                age=int(match.group(3)),
                drift=float(match.group(4))
            )
        except (ValueError, IndexError):
            return None


@dataclass
class AnnotationHistory:
    """History of annotations for a single file."""
    current: StigmergicAnnotation
    history: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return {
            "current": {
                "H": self.current.H,
                "press": self.current.press,
                "age": self.current.age,
                "drift": self.current.drift
            },
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationHistory":
        """Create from dictionary."""
        current_data = data.get("current", {})
        current = StigmergicAnnotation(
            H=current_data.get("H", 0.50),
            press=current_data.get("press", "none"),
            age=current_data.get("age", 0),
            drift=current_data.get("drift", 0.00)
        )
        return cls(current=current, history=data.get("history", []))


def get_annotations_path(project_root: Path | str) -> Path:
    """Get path to annotations sidecar file.
    
    Args:
        project_root: Project root directory
        
    Returns:
        Path to .myco/annotations.json
    """
    project_root = Path(project_root)
    myco_dir = project_root / ".myco"
    myco_dir.mkdir(parents=True, exist_ok=True)
    return myco_dir / "annotations.json"


def load_annotations(project_root: Path | str) -> dict[str, AnnotationHistory]:
    """Load all annotations from sidecar file.
    
    Args:
        project_root: Project root directory
        
    Returns:
        Dict mapping file paths to AnnotationHistory
    """
    annotations_path = get_annotations_path(project_root)
    
    if not annotations_path.exists():
        return {}
    
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        annotations = {}
        for file_path, history_data in data.get("annotations", {}).items():
            annotations[file_path] = AnnotationHistory.from_dict(history_data)
        
        return annotations
    except (json.JSONDecodeError, IOError):
        return {}


def save_annotations(project_root: Path | str, annotations: dict[str, AnnotationHistory]) -> None:
    """Save all annotations to sidecar file.
    
    Args:
        project_root: Project root directory
        annotations: Dict mapping file paths to AnnotationHistory
    """
    annotations_path = get_annotations_path(project_root)
    
    data = {
        "schema_version": 2,
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "annotations": {
            path: history.to_dict()
            for path, history in annotations.items()
        }
    }
    
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def migrate_source_annotations(project_root: Path | str) -> int:
    """Migrate source file annotations to sidecar file.
    
    Scans all Python files for source annotations and migrates them
    to the sidecar file. Source annotations are kept for backward
    compatibility but new writes go to sidecar.
    
    Args:
        project_root: Project root directory
        
    Returns:
        Number of annotations migrated
    """
    project_root = Path(project_root)
    annotations = load_annotations(project_root)
    migrated = 0
    
    for py_file in project_root.rglob("*.py"):
        if any(part.startswith('.') for part in py_file.parts):
            continue
        
        try:
            reader = StigmaReader(py_file)
            annotation = reader.read_annotation()
            
            if annotation:
                rel_path = str(py_file.relative_to(project_root))
                
                if rel_path not in annotations:
                    # Create new history entry
                    annotations[rel_path] = AnnotationHistory(
                        current=annotation,
                        history=[{
                            "session": 0,
                            "H": annotation.H,
                            "press": annotation.press,
                            "drift": annotation.drift,
                            "migrated_from_source": True
                        }]
                    )
                    migrated += 1
        except (SyntaxError, FileNotFoundError):
            continue
    
    if migrated > 0:
        save_annotations(project_root, annotations)
    
    return migrated


class StigmaReader:
    """Reads and writes stigmergic annotations on source files.
    
    Uses Python's ast module to find the first substantive line in a file.
    The annotation is placed immediately before the first non-comment, non-import statement.
    
    Storage:
    - Reads from sidecar file (.myco/annotations.json) first, falls back to source comments
    - Writes to sidecar file only (source comments kept for backward compatibility)
    """

    def __init__(self, file_path: Path | str, project_root: Optional[Path | str] = None):
        """Initialize with a file path.

        Args:
            file_path: Path to the source file
            project_root: Optional project root for sidecar file access
        """
        self.file_path = Path(file_path)
        self.project_root = Path(project_root) if project_root else None
        self._source: Optional[str] = None
        self._tree: Optional[ast.AST] = None
        self._lines: list[str] = []

    def _load(self) -> None:
        """Load and parse the source file."""
        if self._source is not None:
            return

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self._source = self.file_path.read_text(encoding="utf-8")
        self._lines = self._source.splitlines(keepends=True)

        try:
            self._tree = ast.parse(self._source)
        except SyntaxError as e:
            raise SyntaxError(f"Invalid Python syntax in {self.file_path}: {e}")

    def read_annotation(self) -> Optional[StigmergicAnnotation]:
        """Read the stigmergic annotation from the file.
        
        Checks sidecar file first, then falls back to source file comments.

        Returns:
            StigmergicAnnotation if found, None otherwise
        """
        # Try sidecar file first if project_root is set
        if self.project_root:
            annotations = load_annotations(self.project_root)
            rel_path = str(self.file_path.relative_to(self.project_root))
            
            if rel_path in annotations:
                return annotations[rel_path].current
        
        # Fall back to source file
        self._load()

        # Look for annotation in comments before first substantive statement
        for i, line in enumerate(self._lines):
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                annotation = StigmergicAnnotation.parse(stripped)
                if annotation:
                    return annotation
            else:
                # Found first non-comment line, no annotation found
                return None

        return None

    def find_annotation_line(self) -> int:
        """Find the line number where annotation should be placed.

        Returns the line number (0-indexed) where the annotation should be inserted.
        This is the line immediately before the first substantive statement.

        Returns:
            Line number for annotation placement
        """
        self._load()

        # Track if we're still in header (comments, docstrings, imports)
        in_header = True
        header_end = 0

        for node in ast.walk(self._tree):
            # Get the line number of this node
            if isinstance(node, ast.AST) and hasattr(node, 'lineno'):
                line_no = node.lineno - 1  # Convert to 0-indexed

                # Skip module-level docstrings and imports - they're part of header
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue

                # Skip expressions at module level (like docstrings)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        continue

                # Found first substantive statement
                if line_no > header_end:
                    header_end = line_no
                break

        # Find where to insert annotation - before first substantive code
        # but after any existing header comments
        insert_line = 0
        for i in range(header_end):
            line = self._lines[i].strip() if i < len(self._lines) else ""

            # Skip empty lines and pure comments at the top
            if not line or (line.startswith('#') and not StigmergicAnnotation.parse(line)):
                insert_line = i + 1
            else:
                break

        return insert_line

    def write_annotation(self, annotation: StigmergicAnnotation, session_id: str = "") -> None:
        """Write a stigmergic annotation to the sidecar file.
        
        Source file comments are kept for backward compatibility but not modified.

        Args:
            annotation: The annotation to write
            session_id: Optional session ID for history tracking
        """
        if not self.project_root:
            # No project root - fall back to writing source file
            self._write_to_source(annotation)
            return
        
        # Load existing annotations
        annotations = load_annotations(self.project_root)
        rel_path = str(self.file_path.relative_to(self.project_root))
        
        # Get existing history or create new
        if rel_path in annotations:
            history = annotations[rel_path]
            # Add to history
            history.history.append({
                "session": session_id or datetime.utcnow().isoformat(),
                "H": annotation.H,
                "press": annotation.press,
                "drift": annotation.drift
            })
            # Update current
            history.current = annotation
        else:
            # New annotation
            annotations[rel_path] = AnnotationHistory(
                current=annotation,
                history=[{
                    "session": session_id or datetime.utcnow().isoformat(),
                    "H": annotation.H,
                    "press": annotation.press,
                    "drift": annotation.drift
                }]
            )
        
        # Save to sidecar
        save_annotations(self.project_root, annotations)

    def _write_to_source(self, annotation: StigmergicAnnotation) -> None:
        """Write annotation to source file (backward compatibility).
        
        Args:
            annotation: The annotation to write
        """
        self._load()

        existing = self.read_annotation()
        annotation_line = self.find_annotation_line()

        if existing:
            # Replace existing annotation
            self._lines[annotation_line] = annotation.format() + '\n'
        else:
            # Insert new annotation
            self._lines.insert(annotation_line, annotation.format() + '\n')

        # Write back to file
        new_source = ''.join(self._lines)
        self.file_path.write_text(new_source, encoding="utf-8")

        # Clear cache so subsequent reads get fresh data
        self._source = None
        self._tree = None
        self._lines = []

    def update_annotation(
        self,
        H: Optional[float] = None,
        press: Optional[str] = None,
        age: Optional[int] = None,
        drift: Optional[float] = None,
        session_id: str = ""
    ) -> StigmergicAnnotation:
        """Update the annotation with new values.

        Args:
            H: New entropy value (optional)
            press: New press type (optional)
            age: New age (optional)
            drift: New drift (optional)
            session_id: Session ID for history tracking

        Returns:
            The updated annotation
        """
        existing = self.read_annotation() or StigmergicAnnotation()

        if H is not None:
            existing.H = H
        if press is not None:
            existing.press = press
        if age is not None:
            existing.age = age
        if drift is not None:
            existing.drift = drift

        self.write_annotation(existing, session_id)
        return existing
    
    def get_first_substantive_line(self) -> Optional[ast.AST]:
        """Get the first substantive AST node (non-import, non-docstring).
        
        Returns:
            First substantive AST node, or None if file is empty
        """
        self._load()
        
        for node in ast.walk(self._tree):
            if isinstance(node, ast.AST) and hasattr(node, 'lineno'):
                # Skip imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                # Skip module-level docstrings
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        continue
                return node
        
        return None
