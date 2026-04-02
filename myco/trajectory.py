# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Entropy trajectory analysis from git history.

Computes entropy changes over time by analyzing file content at each commit.
Provides velocity (rate of change) and acceleration (change in velocity) metrics.
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .entropy import compute_internal_entropy, ImportGraphBuilder, EntropyCalculator


@dataclass
class EntropyPoint:
    """Entropy measurement at a point in time."""
    commit_hash: str
    timestamp: str
    H_structural: float
    H_internal: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp,
            "H_structural": self.H_structural,
            "H_internal": self.H_internal,
        }


@dataclass
class EntropyTrajectory:
    """Full entropy trajectory for a file."""
    file_path: str
    points: list[EntropyPoint]
    velocity_structural: float
    velocity_internal: float
    acceleration_structural: float
    acceleration_internal: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "points": [p.to_dict() for p in self.points],
            "velocity_structural": self.velocity_structural,
            "velocity_internal": self.velocity_internal,
            "acceleration_structural": self.acceleration_structural,
            "acceleration_internal": self.acceleration_internal,
        }


def get_file_commits(filepath: Path, n_commits: int = 20) -> list[tuple[str, str]]:
    """Get the last N commits that modified a file.
    
    Args:
        filepath: Path to file
        n_commits: Number of commits to retrieve
        
    Returns:
        List of (commit_hash, timestamp) tuples
    """
    try:
        result = subprocess.run(
            ["git", "log", f"-n{str(n_commits)}", "--pretty=format:%H|%ai", "--", str(filepath)],
            capture_output=True,
            text=True,
            cwd=filepath.parent,
            timeout=30
        )
        
        if result.returncode != 0:
            return []
        
        commits = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    commits.append((parts[0], parts[1]))
        
        return commits
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_file_at_commit(filepath: Path, commit_hash: str) -> Optional[str]:
    """Get file content at a specific commit.
    
    Args:
        filepath: Path to file
        commit_hash: Git commit hash
        
    Returns:
        File content or None if file doesn't exist at that commit
    """
    try:
        # Get relative path from git root
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=filepath.parent,
            timeout=10
        )
        
        if result.returncode != 0:
            return None
        
        git_root = Path(result.stdout.strip())
        rel_path = filepath.relative_to(git_root)
        
        # Get file content at commit
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{rel_path}"],
            capture_output=True,
            text=True,
            cwd=git_root,
            timeout=10
        )
        
        if result.returncode != 0:
            return None
        
        return result.stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def compute_structural_entropy_at_commit(
    filepath: Path,
    commit_hash: str,
    git_root: Path
) -> float:
    """Compute structural (import graph) entropy at a specific commit.
    
    Args:
        filepath: Path to file
        commit_hash: Git commit hash
        git_root: Git repository root
        
    Returns:
        Structural entropy value
    """
    try:
        # Get relative path
        rel_path = filepath.relative_to(git_root)
        
        # Get file content at commit
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{rel_path}"],
            capture_output=True,
            text=True,
            cwd=git_root,
            timeout=10
        )
        
        if result.returncode != 0:
            return 0.5
        
        # Write to temp file for analysis
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(result.stdout)
            temp_path = Path(f.name)
        
        try:
            # Build import graph at this commit
            builder = ImportGraphBuilder(git_root)
            builder.scan()
            
            calculator = EntropyCalculator(builder)
            module_name = builder._path_to_module_name(temp_path)
            
            return calculator.calculate_module_entropy(module_name)
        finally:
            temp_path.unlink()
            
    except (subprocess.SubprocessError, FileNotFoundError, IOError):
        return 0.5


def compute_velocity(values: list[float]) -> float:
    """Compute average velocity (first derivative) from a list of values.
    
    Args:
        values: List of entropy values over time (oldest to newest)
        
    Returns:
        Average change per session
    """
    if len(values) < 2:
        return 0.0
    
    # Compute differences
    diffs = [values[i] - values[i-1] for i in range(1, len(values))]
    
    # Average velocity
    return sum(diffs) / len(diffs) if diffs else 0.0


def compute_acceleration(values: list[float]) -> float:
    """Compute acceleration (second derivative) from a list of values.
    
    Args:
        values: List of entropy values over time (oldest to newest)
        
    Returns:
        Change in velocity
    """
    if len(values) < 3:
        return 0.0
    
    # Compute velocities between consecutive points
    velocities = [values[i] - values[i-1] for i in range(1, len(values))]
    
    # Compute acceleration (change in velocity)
    if len(velocities) < 2:
        return 0.0
    
    accel_diffs = [velocities[i] - velocities[i-1] for i in range(1, len(velocities))]
    
    return sum(accel_diffs) / len(accel_diffs) if accel_diffs else 0.0


def compute_entropy_trajectory(
    filepath: Path | str,
    n_commits: int = 20
) -> Optional[EntropyTrajectory]:
    """Compute entropy trajectory for a file from git history.
    
    Args:
        filepath: Path to file
        n_commits: Number of commits to analyze
        
    Returns:
        EntropyTrajectory or None if not a git repository
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return None
    
    # Check if in git repository
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=filepath.parent,
            timeout=10
        )
        
        if result.returncode != 0 or result.stdout.strip() != "true":
            return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    
    # Get git root
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        cwd=filepath.parent,
        timeout=10
    )
    
    if result.returncode != 0:
        return None
    
    git_root = Path(result.stdout.strip())
    
    # Get commits
    commits = get_file_commits(filepath, n_commits)
    
    if not commits:
        return None
    
    # Compute entropy at each commit
    points = []
    H_structural_values = []
    H_internal_values = []
    
    for commit_hash, timestamp in reversed(commits):  # Oldest first
        # Get file content at commit
        content = get_file_at_commit(filepath, commit_hash)
        
        if content is None:
            continue
        
        # Compute internal entropy
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        
        try:
            internal_metrics = compute_internal_entropy(temp_path)
            H_internal = internal_metrics["H_internal"]
        finally:
            temp_path.unlink()
        
        # Compute structural entropy
        H_structural = compute_structural_entropy_at_commit(filepath, commit_hash, git_root)
        
        points.append(EntropyPoint(
            commit_hash=commit_hash,
            timestamp=timestamp,
            H_structural=H_structural,
            H_internal=H_internal
        ))
        
        H_structural_values.append(H_structural)
        H_internal_values.append(H_internal)
    
    if not points:
        return None
    
    # Compute velocity and acceleration
    velocity_structural = compute_velocity(H_structural_values)
    velocity_internal = compute_velocity(H_internal_values)
    acceleration_structural = compute_acceleration(H_structural_values)
    acceleration_internal = compute_acceleration(H_internal_values)
    
    return EntropyTrajectory(
        file_path=str(filepath.relative_to(git_root) if filepath.is_relative_to(git_root) else filepath.name),
        points=points,
        velocity_structural=velocity_structural,
        velocity_internal=velocity_internal,
        acceleration_structural=acceleration_structural,
        acceleration_internal=acceleration_internal
    )


def interpret_trajectory(trajectory: EntropyTrajectory) -> dict:
    """Interpret entropy trajectory and provide guidance.
    
    Args:
        trajectory: EntropyTrajectory object
        
    Returns:
        Dict with interpretation and guidance
    """
    # Get current values (most recent point)
    current = trajectory.points[-1] if trajectory.points else None
    
    if not current:
        return {
            "status": "unknown",
            "guidance": "No entropy data available"
        }
    
    # Interpret velocity
    if trajectory.velocity_structural < -0.02:
        structural_trend = "crystallizing"
        structural_urgency = "high"
    elif trajectory.velocity_structural < 0:
        structural_trend = "trending_crystallized"
        structural_urgency = "medium"
    elif trajectory.velocity_structural > 0.02:
        structural_trend = "diffusing"
        structural_urgency = "high"
    elif trajectory.velocity_structural > 0:
        structural_trend = "trending_diffuse"
        structural_urgency = "medium"
    else:
        structural_trend = "stable"
        structural_urgency = "low"
    
    # Interpret acceleration
    if trajectory.acceleration_structural < -0.01:
        structural_accel = "accelerating_crystallization"
    elif trajectory.acceleration_structural > 0.01:
        structural_accel = "accelerating_diffusion"
    else:
        structural_accel = "stable_velocity"
    
    # Generate guidance
    if structural_trend == "crystallizing" and structural_accel == "accelerating_crystallization":
        guidance = "Urgent: Module is rapidly crystallizing. Plan restructuring within 1-2 sessions."
        priority = 1
    elif structural_trend == "crystallizing":
        guidance = "Module is crystallizing. Monitor closely and plan decompression."
        priority = 2
    elif structural_trend == "stable":
        guidance = "Module is stable. Safe to modify."
        priority = 3
    elif structural_trend == "diffusing" and structural_accel == "accelerating_diffusion":
        guidance = "Module is rapidly becoming diffuse. Consider consolidation."
        priority = 2
    else:
        guidance = f"Module trend: {structural_trend}. {structural_accel}."
        priority = 3
    
    return {
        "status": structural_trend,
        "acceleration": structural_accel,
        "urgency": structural_urgency,
        "priority": priority,
        "guidance": guidance,
        "current_H_structural": current.H_structural,
        "current_H_internal": current.H_internal,
        "velocity_structural": trajectory.velocity_structural,
        "velocity_internal": trajectory.velocity_internal,
    }
