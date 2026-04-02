"""MYCO Project Context Manager

Ensures agent always works within MY_project folders.
Provides project isolation and context awareness.
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..utils.logging import LogConfig


@dataclass
class ProjectContext:
    """Context information for current project."""
    name: str
    root: Path
    myco_dir: Path
    src_dir: Optional[Path] = None
    test_dir: Optional[Path] = None
    docs_dir: Optional[Path] = None
    venv_dir: Optional[Path] = None
    git_repo: bool = False
    python_version: str = ""
    dependencies: list[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ProjectContextManager:
    """Manages project context for MYCO agent.
    
    Ensures all agent operations are scoped to the current project.
    Provides project-aware defaults and constraints.
    """
    
    logger = LogConfig.get_logger("gguf.agent.project")
    
    # Project naming patterns
    PROJECT_PREFIXES = ["MY_", "my_"]
    PROJECT_SUFFIXES = ["_project", "_Project", "_PROJECT"]
    
    # Protected directories (agent should not modify)
    PROTECTED_DIRS = {
        "D:\\",
        "C:\\",
        "C:\\Windows",
        "C:\\Program Files",
        "C:\\Program Files (x86)",
    }
    
    def __init__(self, cwd: Optional[Path] = None):
        """Initialize project context manager.
        
        Args:
            cwd: Current working directory (default: Path.cwd())
        """
        self.cwd = cwd or Path.cwd()
        self.current_project: Optional[ProjectContext] = None
        
    def find_project_root(self, start_path: Optional[Path] = None) -> Optional[Path]:
        """Find the project root from a starting path.
        
        Looks for:
        1. MY_project folder
        2. .git directory
        3. pyproject.toml or setup.py
        4. src/ directory with Python files
        
        Args:
            start_path: Starting path (default: self.cwd)
            
        Returns:
            Project root path or None
        """
        start_path = start_path or self.cwd
        
        # Walk up directory tree
        for parent in [start_path] + list(start_path.parents):
            # Check for MY_project pattern
            if parent.name.startswith("MY_") or parent.name.startswith("my_"):
                return parent
            
            # Check for .git directory
            if (parent / ".git").exists():
                return parent
            
            # Check for Python project markers
            if (parent / "pyproject.toml").exists():
                return parent
            if (parent / "setup.py").exists():
                return parent
            
            # Check for src/ directory
            src_dir = parent / "src"
            if src_dir.exists() and src_dir.is_dir():
                py_files = list(src_dir.rglob("*.py"))
                if py_files:
                    return parent
        
        return None
    
    def create_project(self, project_name: str, location: Optional[Path] = None) -> ProjectContext:
        """Create a new MY_project.
        
        Args:
            project_name: Name for the project (will be prefixed with MY_)
            location: Location to create project (default: cwd)
            
        Returns:
            ProjectContext for the new project
        """
        location = location or self.cwd
        
        # Ensure MY_ prefix
        if not project_name.startswith("MY_"):
            project_name = f"MY_{project_name}"
        
        project_root = location / project_name
        
        # Create project structure
        project_root.mkdir(exist_ok=True)
        (project_root / "src").mkdir(exist_ok=True)
        (project_root / "tests").mkdir(exist_ok=True)
        (project_root / "docs").mkdir(exist_ok=True)
        
        # Create .myco directory
        myco_dir = project_root / ".myco"
        myco_dir.mkdir(exist_ok=True)
        
        # Create README
        readme = project_root / "README.md"
        if not readme.exists():
            readme.write_text(f"# {project_name}\n\nProject created by MYCO agent.\n")
        
        # Create .gitignore
        gitignore = project_root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("""__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env
.myco/
*.log
""")
        
        self.logger.info(f"Created project: {project_root}")
        
        return self.load_project(project_root)
    
    def load_project(self, project_root: Path) -> ProjectContext:
        """Load project context from a root path.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            ProjectContext with project information
        """
        project_root = project_root.resolve()
        
        # Detect project structure
        src_dir = None
        test_dir = None
        docs_dir = None
        venv_dir = None
        
        # Check for src/ directory
        if (project_root / "src").exists():
            src_dir = project_root / "src"
        elif (project_root / project_root.name).exists():
            src_dir = project_root / project_root.name
        
        # Check for tests/ directory
        for test_name in ["tests", "test", "Test", "TEST"]:
            if (project_root / test_name).exists():
                test_dir = project_root / test_name
                break
        
        # Check for docs/ directory
        for docs_name in ["docs", "doc", "Docs", "documentation"]:
            if (project_root / docs_name).exists():
                docs_dir = project_root / docs_name
                break
        
        # Check for virtualenv
        for venv_name in ["venv", "env", ".venv", ".env"]:
            if (project_root / venv_name).exists():
                venv_dir = project_root / venv_name
                break
        
        # Check for git repo
        git_repo = (project_root / ".git").exists()
        
        # Get Python version
        python_version = ""
        try:
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        except Exception:
            pass
        
        # Load dependencies
        dependencies = []
        requirements_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
        for req_file in requirements_files:
            req_path = project_root / req_file
            if req_path.exists():
                try:
                    content = req_path.read_text(encoding="utf-8")
                    for line in content.splitlines():
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Handle -r includes
                            if line.startswith("-r"):
                                continue
                            # Extract package name
                            pkg = line.split("==")[0].split(">=")[0].split("<")[0].split("[")[0].strip()
                            if pkg:
                                dependencies.append(pkg)
                except Exception:
                    pass
        
        ctx = ProjectContext(
            name=project_root.name,
            root=project_root,
            myco_dir=project_root / ".myco",
            src_dir=src_dir,
            test_dir=test_dir,
            docs_dir=docs_dir,
            venv_dir=venv_dir,
            git_repo=git_repo,
            python_version=python_version,
            dependencies=dependencies
        )
        
        self.current_project = ctx
        self.logger.info(f"Loaded project context: {ctx.name}")
        
        return ctx
    
    def get_current_project(self) -> Optional[ProjectContext]:
        """Get the current project context.
        
        Returns:
            Current ProjectContext or None
        """
        if self.current_project:
            return self.current_project
        
        # Try to auto-detect
        project_root = self.find_project_root()
        if project_root:
            return self.load_project(project_root)
        
        return None
    
    def ensure_in_project(self, path: Path) -> Path:
        """Ensure a path is within the current project.
        
        Prevents agent from modifying files outside the project.
        
        Args:
            path: Path to check
            
        Returns:
            Resolved path if within project
            
        Raises:
            ValueError: If path is outside project or in protected directory
        """
        path = path.resolve()
        
        # Check protected directories
        for protected in self.PROTECTED_DIRS:
            if str(path).startswith(protected):
                raise ValueError(f"Cannot modify protected directory: {protected}")
        
        # Get current project
        project = self.get_current_project()
        if not project:
            # No project context - allow but warn
            self.logger.warning(f"No project context, allowing: {path}")
            return path
        
        # Ensure path is within project
        try:
            path.relative_to(project.root)
            return path
        except ValueError:
            raise ValueError(f"Path {path} is outside project {project.root}")
    
    def get_project_system_prompt(self) -> str:
        """Generate system prompt section for project context.
        
        Returns:
            System prompt text describing project context
        """
        project = self.get_current_project()
        
        if not project:
            return """
PROJECT CONTEXT:
- No specific project detected
- Working in current directory
- Avoid modifying system directories
"""
        
        sections = [
            f"""PROJECT CONTEXT: {project.name}
- Root: {project.root}
- Python: {project.python_version or 'N/A'}
- Git Repository: {'Yes' if project.git_repo else 'No'}
""",
        ]
        
        if project.src_dir:
            sections.append(f"- Source Directory: {project.src_dir.relative_to(project.root)}")
        if project.test_dir:
            sections.append(f"- Test Directory: {project.test_dir.relative_to(project.root)}")
        if project.docs_dir:
            sections.append(f"- Docs Directory: {project.docs_dir.relative_to(project.root)}")
        if project.venv_dir:
            sections.append(f"- Virtual Environment: {project.venv_dir.relative_to(project.root)}")
        
        if project.dependencies:
            deps_str = ", ".join(project.dependencies[:10])
            if len(project.dependencies) > 10:
                deps_str += f" (+{len(project.dependencies) - 10} more)"
            sections.append(f"- Dependencies: {deps_str}")
        
        sections.append("""
CONSTRAINTS:
- All file operations must be within this project directory
- Do not modify files outside the project root
- Do not modify system directories
- Respect the project structure (src/, tests/, docs/)
""")
        
        return "\n".join(sections)
    
    def suggest_file_location(self, file_type: str, filename: str) -> Path:
        """Suggest the best location for a file based on type.
        
        Args:
            file_type: Type of file (source, test, doc, config)
            filename: Name of the file
            
        Returns:
            Suggested full path
        """
        project = self.get_current_project()
        
        if not project:
            return self.cwd / filename
        
        if file_type == "source" or file_type == "python":
            if project.src_dir:
                return project.src_dir / filename
            return project.root / filename
        
        elif file_type == "test":
            if project.test_dir:
                return project.test_dir / filename
            return project.root / filename
        
        elif file_type == "doc" or file_type == "docs":
            if project.docs_dir:
                return project.docs_dir / filename
            return project.root / "docs" / filename
        
        elif file_type == "config":
            return project.root / filename
        
        else:
            return project.root / filename


# Global project context manager
_project_context: Optional[ProjectContextManager] = None


def get_project_context() -> ProjectContextManager:
    """Get the global project context manager.
    
    Returns:
        ProjectContextManager instance
    """
    global _project_context
    if _project_context is None:
        _project_context = ProjectContextManager()
    return _project_context


def ensure_project_context(project_root: Optional[Path] = None) -> ProjectContext:
    """Ensure project context is loaded.

    Args:
        project_root: Optional project root to load

    Returns:
        Current ProjectContext
    """
    global _project_context
    
    if project_root:
        # Always create fresh context when project_root is explicitly provided
        _project_context = ProjectContextManager()
        ctx_manager = _project_context
        return ctx_manager.load_project(project_root)

    ctx_manager = get_project_context()
    project = ctx_manager.get_current_project()
    if not project:
        # Auto-detect or create default
        detected = ctx_manager.find_project_root()
        if detected:
            return ctx_manager.load_project(detected)

        # Create default MY_project
        return ctx_manager.create_project("MY_project")

    return project
