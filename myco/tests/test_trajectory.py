# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Tests for myco trajectory module."""

import pytest
from pathlib import Path
import subprocess


class TestEntropyTrajectory:
    """Tests for entropy trajectory computation."""

    @pytest.fixture
    def git_repo_with_history(self, tmp_path):
        """Create a git repository with file history."""
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
        
        # Create file with initial content
        test_file = tmp_path / "test.py"
        test_file.write_text("def func1(): pass\n")
        subprocess.run(["git", "add", "test.py"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, check=True, capture_output=True)
        
        # Modify file
        test_file.write_text("def func1(): pass\ndef func2(): pass\n")
        subprocess.run(["git", "add", "test.py"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add func2"], cwd=tmp_path, check=True, capture_output=True)
        
        # Modify again
        test_file.write_text("def func1(): pass\ndef func2(): pass\ndef func3(): pass\n")
        subprocess.run(["git", "add", "test.py"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add func3"], cwd=tmp_path, check=True, capture_output=True)
        
        yield tmp_path

    def test_compute_entropy_trajectory(self, git_repo_with_history):
        """Test computing entropy trajectory."""
        from myco.trajectory import compute_entropy_trajectory
        
        test_file = git_repo_with_history / "test.py"
        trajectory = compute_entropy_trajectory(test_file, n_commits=3)
        
        assert trajectory is not None
        assert len(trajectory.points) >= 1
        assert trajectory.file_path == "test.py"

    def test_compute_entropy_trajectory_nonexistent(self, tmp_path):
        """Test trajectory for nonexistent file."""
        from myco.trajectory import compute_entropy_trajectory
        
        trajectory = compute_entropy_trajectory(tmp_path / "nonexistent.py")
        
        assert trajectory is None

    def test_compute_entropy_trajectory_not_git(self, tmp_path):
        """Test trajectory outside git repository."""
        from myco.trajectory import compute_entropy_trajectory
        
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass\n")
        
        trajectory = compute_entropy_trajectory(test_file)
        
        assert trajectory is None

    def test_get_file_commits(self, git_repo_with_history):
        """Test getting file commits."""
        from myco.trajectory import get_file_commits
        
        test_file = git_repo_with_history / "test.py"
        commits = get_file_commits(test_file, n_commits=5)
        
        assert len(commits) == 3  # We made 3 commits

    def test_compute_velocity(self):
        """Test velocity computation."""
        from myco.trajectory import compute_velocity
        
        # Increasing values
        values = [0.3, 0.4, 0.5, 0.6]
        velocity = compute_velocity(values)
        
        assert velocity > 0
        
        # Decreasing values
        values = [0.6, 0.5, 0.4, 0.3]
        velocity = compute_velocity(values)
        
        assert velocity < 0
        
        # Single value
        velocity = compute_velocity([0.5])
        
        assert velocity == 0.0

    def test_compute_acceleration(self):
        """Test acceleration computation."""
        from myco.trajectory import compute_acceleration
        
        # Constant velocity (zero acceleration)
        values = [0.3, 0.4, 0.5, 0.6]
        acceleration = compute_acceleration(values)
        
        assert abs(acceleration) < 0.01
        
        # Increasing velocity (positive acceleration)
        values = [0.3, 0.4, 0.6, 0.9]
        acceleration = compute_acceleration(values)
        
        assert acceleration > 0
        
        # Too few values
        acceleration = compute_acceleration([0.5, 0.6])
        
        assert acceleration == 0.0

    def test_interpret_trajectory_crystallizing(self):
        """Test trajectory interpretation for crystallizing module."""
        from myco.trajectory import EntropyTrajectory, EntropyPoint, interpret_trajectory
        
        # Create trajectory with decreasing entropy (crystallizing)
        trajectory = EntropyTrajectory(
            file_path="test.py",
            points=[
                EntropyPoint("abc123", "2024-01-01", 0.5, 0.5),
                EntropyPoint("def456", "2024-01-02", 0.4, 0.4),
                EntropyPoint("ghi789", "2024-01-03", 0.3, 0.3),
                EntropyPoint("jkl012", "2024-01-04", 0.2, 0.2),
            ],
            velocity_structural=-0.1,
            velocity_internal=-0.1,
            acceleration_structural=-0.02,
            acceleration_internal=-0.02
        )
        
        interpretation = interpret_trajectory(trajectory)
        
        assert interpretation["status"] == "crystallizing"
        assert interpretation["priority"] <= 2

    def test_interpret_trajectory_stable(self):
        """Test trajectory interpretation for stable module."""
        from myco.trajectory import EntropyTrajectory, EntropyPoint, interpret_trajectory
        
        # Create trajectory with stable entropy
        trajectory = EntropyTrajectory(
            file_path="test.py",
            points=[
                EntropyPoint("abc123", "2024-01-01", 0.5, 0.5),
                EntropyPoint("def456", "2024-01-02", 0.5, 0.5),
                EntropyPoint("ghi789", "2024-01-03", 0.5, 0.5),
            ],
            velocity_structural=0.0,
            velocity_internal=0.0,
            acceleration_structural=0.0,
            acceleration_internal=0.0
        )
        
        interpretation = interpret_trajectory(trajectory)
        
        assert interpretation["status"] == "stable"
        assert interpretation["urgency"] == "low"

    def test_interpret_trajectory_empty(self):
        """Test trajectory interpretation with no points."""
        from myco.trajectory import EntropyTrajectory, interpret_trajectory
        
        trajectory = EntropyTrajectory(
            file_path="test.py",
            points=[],
            velocity_structural=0.0,
            velocity_internal=0.0,
            acceleration_structural=0.0,
            acceleration_internal=0.0
        )
        
        interpretation = interpret_trajectory(trajectory)
        
        assert interpretation["status"] == "unknown"
