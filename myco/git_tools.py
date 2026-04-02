# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Git integration for myco.

Provides git awareness without compromising myco's philosophy.
All operations are optional and non-intrusive.
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GitStatus:
    """Git repository status."""
    is_repo: bool = False
    branch: str = ""
    dirty: bool = False
    modified_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_repo": self.is_repo,
            "branch": self.branch,
            "dirty": self.dirty,
            "modified_files": self.modified_files,
            "untracked_files": self.untracked_files,
            "staged_files": self.staged_files,
        }


class GitTools:
    """Git operations for myco.
    
    All operations are optional and fail gracefully if git is not available.
    """
    
    def __init__(self, project_root: Path | str):
        """Initialize git tools.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self._git_available: Optional[bool] = None
    
    def _check_git_available(self) -> bool:
        """Check if git is available."""
        if self._git_available is not None:
            return self._git_available
        
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                timeout=5,
                check=True
            )
            self._git_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            self._git_available = False
        
        return self._git_available
    
    def _run_git(self, *args: str, check: bool = False) -> tuple[bool, str, str]:
        """Run a git command.
        
        Args:
            *args: Git command arguments
            check: Raise exception on non-zero exit
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        if not self._check_git_available():
            return False, "", "Git not available"
        
        try:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.project_root,
                check=check
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.SubprocessError as e:
            return False, "", str(e)
    
    def is_repository(self) -> bool:
        """Check if project root is a git repository.
        
        Returns:
            True if git repository exists
        """
        if not self._check_git_available():
            return False
        
        git_dir = self.project_root / ".git"
        return git_dir.exists()
    
    def get_status(self) -> GitStatus:
        """Get git repository status.
        
        Returns:
            GitStatus with current state
        """
        status = GitStatus()
        
        if not self.is_repository():
            return status
        
        status.is_repo = True
        
        # Get current branch
        success, stdout, _ = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        if success:
            status.branch = stdout.strip()
        
        # Get modified files
        success, stdout, _ = self._run_git("ls-files", "-m")
        if success:
            status.modified_files = [f for f in stdout.strip().split("\n") if f]
        
        # Get untracked files
        success, stdout, _ = self._run_git("ls-files", "--others", "--exclude-standard")
        if success:
            status.untracked_files = [f for f in stdout.strip().split("\n") if f]
        
        # Get staged files
        success, stdout, _ = self._run_git("diff", "--cached", "--name-only")
        if success:
            status.staged_files = [f for f in stdout.strip().split("\n") if f]
        
        # Determine if dirty
        status.dirty = bool(status.modified_files or status.untracked_files or status.staged_files)
        
        return status
    
    def get_diff(self, path: Optional[Path | str] = None) -> str:
        """Get git diff for a file or working directory.
        
        Args:
            path: Optional file path (None for full working directory)
            
        Returns:
            Diff output or empty string if not available
        """
        if not self.is_repository():
            return ""
        
        if path:
            path = Path(path)
            if not path.exists():
                # New file - show as untracked
                return f"New file: {path}"
            
            success, stdout, _ = self._run_git("diff", str(path))
            if success and stdout:
                return stdout
        
        else:
            # Full working directory diff
            success, stdout, _ = self._run_git("diff")
            if success and stdout:
                return stdout
        
        return ""
    
    def get_file_history(self, path: Path | str, limit: int = 5) -> list[dict]:
        """Get recent commit history for a file.
        
        Args:
            path: File path
            limit: Maximum number of commits to return
            
        Returns:
            List of commit info dicts
        """
        if not self.is_repository():
            return []
        
        path = Path(path)
        if not path.exists():
            return []
        
        success, stdout, _ = self._run_git(
            "log", "-n", str(limit), "--pretty=format:%H|%an|%ad|%s",
            "--date=short", "--", str(path)
        )
        
        if not success or not stdout:
            return []
        
        commits = []
        for line in stdout.strip().split("\n"):
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "hash": parts[0][:7],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3]
                })
        
        return commits
    
    def stage_file(self, path: Path | str) -> bool:
        """Stage a file for commit.
        
        Args:
            path: File to stage
            
        Returns:
            True if successful
        """
        if not self.is_repository():
            return False
        
        success, _, _ = self._run_git("add", str(path))
        return success
    
    def commit(message: str, path: Optional[Path | str] = None) -> bool:
        """Create a commit.
        
        Args:
            message: Commit message
            path: Optional specific file to commit
            
        Returns:
            True if successful
        """
        # Note: This is a static method for flexibility
        success, _, _ = GitTools(Path.cwd())._run_git(
            "commit", "-m", message, *(["--", str(path)] if path else [])
        )
        return success


# Global git tools instance (lazy initialization)
_git_tools: Optional[GitTools] = None


def get_git_tools(project_root: Optional[Path | str] = None) -> GitTools:
    """Get git tools instance.
    
    Args:
        project_root: Project root (uses cwd if not provided)
        
    Returns:
        GitTools instance
    """
    global _git_tools
    if _git_tools is None:
        root = Path(project_root) if project_root else Path.cwd()
        _git_tools = GitTools(root)
    return _git_tools


def get_repo_status(project_root: Optional[Path | str] = None) -> GitStatus:
    """Get repository status.
    
    Args:
        project_root: Project root (uses cwd if not provided)
        
    Returns:
        GitStatus (is_repo=False if not a repository)
    """
    return get_git_tools(project_root).get_status()
