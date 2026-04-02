# ⊕ H:0.25 | press:quality | age:0 | drift:+0.00
"""MYCO Quality Tracking - Track code quality over time.

MYCO Vision:
- Quality trends (improving/stable/degrading)
- Entropy tracking over sessions
- Test coverage tracking
- Project health dashboard

Architecture:
- Quality logs (.myco/quality/) - Per-file quality history
- Project health (.myco/health.json) - Overall project metrics
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any


@dataclass
class QualityMeasurement:
    """Single quality measurement for a file."""
    
    session_id: str
    timestamp: str
    entropy_before: float
    entropy_after: float
    entropy_delta: float
    tests_created: int = 0
    tests_passed: int = 0
    syntax_valid: bool = True
    dependencies_tracked: int = 0
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "entropy_before": self.entropy_before,
            "entropy_after": self.entropy_after,
            "entropy_delta": self.entropy_delta,
            "tests_created": self.tests_created,
            "tests_passed": self.tests_passed,
            "syntax_valid": self.syntax_valid,
            "dependencies_tracked": self.dependencies_tracked
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "QualityMeasurement":
        return cls(**data)


@dataclass
class FileQualityTrend:
    """Quality trend for a single file."""
    
    file_path: str
    measurements: List[QualityMeasurement] = field(default_factory=list)
    trend: str = "stable"  # "improving", "stable", "degrading"
    trend_slope: float = 0.0
    avg_entropy: float = 0.0
    min_entropy: float = 0.0
    max_entropy: float = 0.0
    total_tests: int = 0
    
    def calculate_trend(self):
        """Calculate trend from measurements."""
        if len(self.measurements) < 2:
            self.trend = "stable"
            self.trend_slope = 0.0
            return
        
        # Get entropy values
        entropies = [m.entropy_after for m in self.measurements]
        
        # Calculate average
        self.avg_entropy = sum(entropies) / len(entropies)
        self.min_entropy = min(entropies)
        self.max_entropy = max(entropies)
        
        # Calculate trend slope (linear regression)
        n = len(entropies)
        sum_x = n * (n - 1) / 2
        sum_y = sum(entropies)
        sum_xy = sum(i * e for i, e in enumerate(entropies))
        sum_x2 = n * (n - 1) * (2 * n - 1) / 6
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            self.trend_slope = 0.0
        else:
            self.trend_slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Determine trend direction
        threshold = 0.01  # Minimum slope to consider significant
        if self.trend_slope < -threshold:
            self.trend = "improving"  # Entropy decreasing (good)
        elif self.trend_slope > threshold:
            self.trend = "degrading"  # Entropy increasing (bad)
        else:
            self.trend = "stable"
        
        # Count total tests
        self.total_tests = sum(m.tests_created for m in self.measurements)
    
    def to_dict(self) -> dict:
        self.calculate_trend()
        return {
            "file_path": self.file_path,
            "measurements": [m.to_dict() for m in self.measurements],
            "trend": self.trend,
            "trend_slope": self.trend_slope,
            "avg_entropy": self.avg_entropy,
            "min_entropy": self.min_entropy,
            "max_entropy": self.max_entropy,
            "total_tests": self.total_tests
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FileQualityTrend":
        measurements = [QualityMeasurement.from_dict(m) for m in data.get("measurements", [])]
        trend = cls(
            file_path=data.get("file_path", ""),
            measurements=measurements
        )
        trend.calculate_trend()
        return trend


class QualityTracker:
    """Track code quality over time.
    
    MYCO Phase 2.2: Quality feedback loops.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.quality_dir = self.project_root / ".myco" / "quality"
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        
        self.health_file = self.project_root / ".myco" / "health.json"
        
        # In-memory cache
        self._trends: Dict[str, FileQualityTrend] = {}
        self._load_trends()
    
    def _load_trends(self):
        """Load quality trends from disk."""
        trend_files = list(self.quality_dir.glob("*.json"))
        
        for trend_file in trend_files:
            try:
                with open(trend_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    trend = FileQualityTrend.from_dict(data)
                    self._trends[trend.file_path] = trend
            except Exception:
                pass  # Skip corrupted files
    
    def _save_trend(self, trend: FileQualityTrend):
        """Save trend to disk."""
        # Use file path hash as filename
        file_hash = hash(trend.file_path) & 0xFFFFFFFF
        trend_file = self.quality_dir / f"{file_hash:08x}.json"
        
        with open(trend_file, 'w', encoding='utf-8') as f:
            json.dump(trend.to_dict(), f, indent=2)
    
    def record_change(
        self,
        file_path: str,
        session_id: str,
        entropy_before: float,
        entropy_after: float,
        tests_created: int = 0,
        tests_passed: int = 0,
        syntax_valid: bool = True,
        dependencies_tracked: int = 0
    ):
        """Record quality change for a file.
        
        Args:
            file_path: Path to file
            session_id: Session identifier
            entropy_before: Entropy before change
            entropy_after: Entropy after change
            tests_created: Number of tests created
            tests_passed: Number of tests passed
            syntax_valid: Whether syntax is valid
            dependencies_tracked: Number of dependencies tracked
        """
        # Get or create trend
        if file_path not in self._trends:
            self._trends[file_path] = FileQualityTrend(file_path=file_path)
        
        trend = self._trends[file_path]
        
        # Create measurement
        measurement = QualityMeasurement(
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            entropy_delta=entropy_after - entropy_before,
            tests_created=tests_created,
            tests_passed=tests_passed,
            syntax_valid=syntax_valid,
            dependencies_tracked=dependencies_tracked
        )
        
        # Add to trend
        trend.measurements.append(measurement)
        
        # Keep only last 50 measurements
        if len(trend.measurements) > 50:
            trend.measurements = trend.measurements[-50:]
        
        # Recalculate trend
        trend.calculate_trend()
        
        # Save to disk
        self._save_trend(trend)
        
        # Update project health
        self._update_project_health()
    
    def get_trend(self, file_path: str) -> Optional[FileQualityTrend]:
        """Get quality trend for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            FileQualityTrend or None
        """
        return self._trends.get(file_path)
    
    def get_trend_summary(self, file_path: str) -> Dict[str, Any]:
        """Get trend summary for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict with trend summary
        """
        trend = self.get_trend(file_path)
        
        if not trend:
            return {
                "file": file_path,
                "trend": "unknown",
                "message": "No quality data yet"
            }
        
        # Generate message
        if trend.trend == "improving":
            message = f"✓ Improving (H: {trend.max_entropy:.2f} → {trend.avg_entropy:.2f})"
        elif trend.trend == "degrading":
            message = f"⚠ Degrading (H: {trend.min_entropy:.2f} → {trend.avg_entropy:.2f})"
        else:
            message = f"→ Stable (H: {trend.avg_entropy:.2f})"
        
        return {
            "file": file_path,
            "trend": trend.trend,
            "trend_slope": trend.trend_slope,
            "avg_entropy": trend.avg_entropy,
            "total_tests": trend.total_tests,
            "measurements": len(trend.measurements),
            "message": message
        }
    
    def _update_project_health(self):
        """Update project health metrics."""
        if not self._trends:
            return
        
        # Calculate overall metrics
        all_entropies = [t.avg_entropy for t in self._trends.values() if t.avg_entropy > 0]
        all_tests = sum(t.total_tests for t in self._trends.values())
        
        improving = sum(1 for t in self._trends.values() if t.trend == "improving")
        stable = sum(1 for t in self._trends.values() if t.trend == "stable")
        degrading = sum(1 for t in self._trends.values() if t.trend == "degrading")
        
        # Calculate health score
        total_files = len(self._trends)
        health_score = (improving * 1.0 + stable * 0.7 + degrading * 0.3) / max(total_files, 1)
        
        # Determine health regime
        if health_score > 0.8:
            health_regime = "excellent"
        elif health_score > 0.6:
            health_regime = "healthy"
        elif health_score > 0.4:
            health_regime = "moderate"
        else:
            health_regime = "needs_attention"
        
        # Save health metrics
        health = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_files": total_files,
            "avg_entropy": sum(all_entropies) / max(len(all_entropies), 1),
            "total_tests": all_tests,
            "improving": improving,
            "stable": stable,
            "degrading": degrading,
            "health_score": health_score,
            "health_regime": health_regime
        }
        
        with open(self.health_file, 'w', encoding='utf-8') as f:
            json.dump(health, f, indent=2)
    
    def get_project_health(self) -> Dict[str, Any]:
        """Get overall project health.
        
        Returns:
            Dict with project health metrics
        """
        if self.health_file.exists():
            with open(self.health_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_files": 0,
            "avg_entropy": 0.0,
            "total_tests": 0,
            "improving": 0,
            "stable": 0,
            "degrading": 0,
            "health_score": 1.0,
            "health_regime": "unknown"
        }
    
    def get_files_needing_attention(self) -> List[str]:
        """Get files with degrading quality.
        
        Returns:
            List of file paths needing attention
        """
        return [
            file_path for file_path, trend in self._trends.items()
            if trend.trend == "degrading"
        ]


# ============================================================================
# Agent Tools
# ============================================================================

def record_quality_change(
    project_root: str,
    file_path: str,
    session_id: str,
    entropy_before: float,
    entropy_after: float,
    tests_created: int = 0,
    tests_passed: int = 0
) -> bool:
    """Record quality change (agent tool)."""
    try:
        tracker = QualityTracker(Path(project_root))
        tracker.record_change(
            file_path, session_id, entropy_before, entropy_after,
            tests_created, tests_passed
        )
        return True
    except Exception:
        return False


def get_quality_trend(project_root: str, file_path: str) -> dict:
    """Get quality trend for file (agent tool)."""
    try:
        tracker = QualityTracker(Path(project_root))
        return tracker.get_trend_summary(file_path)
    except Exception:
        return {"trend": "unknown", "message": "Error getting trend"}


def get_project_health(project_root: str) -> dict:
    """Get project health (agent tool)."""
    try:
        tracker = QualityTracker(Path(project_root))
        return tracker.get_project_health()
    except Exception:
        return {"health_regime": "unknown"}


def get_files_needing_attention(project_root: str) -> List[str]:
    """Get files needing attention (agent tool)."""
    try:
        tracker = QualityTracker(Path(project_root))
        return tracker.get_files_needing_attention()
    except Exception:
        return []
