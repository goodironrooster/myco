# ⊕ H:0.25 | press:self_improvement | age:0 | drift:+0.00
"""MYCO Agent Self-Improvement - Agents learn from past success.

MYCO Vision:
- Agents improve over time (not just remember)
- Quality-driven pattern adoption
- Success-based learning
- Continuous improvement loop

Architecture:
- Success analyzer (finds what worked)
- Pattern recommender (suggests best patterns)
- Quality advisor (warns about degrading approaches)
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

from .session_memory import SessionMemory, SessionRecord, PatternLibrary
from .quality import QualityTracker


@dataclass
class SuccessPattern:
    """A pattern that led to successful outcomes."""
    
    pattern_name: str
    task_type: str
    success_count: int
    avg_quality_improvement: float
    example: str
    last_used: str
    files_created: List[str] = None
    
    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "task_type": self.task_type,
            "success_count": self.success_count,
            "avg_quality_improvement": self.avg_quality_improvement,
            "example": self.example,
            "last_used": self.last_used,
            "files_created": self.files_created or []
        }


class SuccessAnalyzer:
    """Analyze past sessions to find what works.
    
    MYCO Phase 2.3: Agents learn from success.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.memory = SessionMemory(project_root)
        self.quality = QualityTracker(project_root)
    
    def analyze_task_success(self, task_description: str) -> Dict[str, Any]:
        """Analyze what worked for similar tasks.
        
        Args:
            task_description: Current task description
            
        Returns:
            Dict with success analysis:
            {
                "similar_sessions": [...],
                "successful_patterns": [...],
                "quality_trends": {...},
                "recommendations": [...]
            }
        """
        # Find similar sessions
        similar = self.memory.find_similar_sessions(task_description, limit=10)
        
        if not similar:
            return {
                "similar_sessions": [],
                "successful_patterns": [],
                "quality_trends": {},
                "recommendations": ["No similar sessions found. This is a new task type."]
            }
        
        # Separate successful from unsuccessful
        successful = [s for s in similar if s.success]
        unsuccessful = [s for s in similar if not s.success]
        
        # Extract patterns from successful sessions
        pattern_counts = {}
        for session in successful:
            for pattern in session.patterns_used:
                if pattern not in pattern_counts:
                    pattern_counts[pattern] = {
                        "count": 0,
                        "sessions": [],
                        "quality_improvements": []
                    }
                pattern_counts[pattern]["count"] += 1
                pattern_counts[pattern]["sessions"].append(session.session_id)
                
                # Check quality improvement for this session's files
                for file_path in session.files_created:
                    trend = self.quality.get_trend(file_path)
                    if trend and trend.trend == "improving":
                        pattern_counts[pattern]["quality_improvements"].append(
                            -trend.trend_slope  # Negative slope = improvement
                        )
        
        # Build successful patterns list
        successful_patterns = []
        for pattern_name, data in pattern_counts.items():
            avg_improvement = (
                sum(data["quality_improvements"]) / len(data["quality_improvements"])
                if data["quality_improvements"] else 0.0
            )
            
            successful_patterns.append({
                "pattern": pattern_name,
                "success_count": data["count"],
                "avg_quality_improvement": avg_improvement,
                "sessions": data["sessions"]
            })
        
        # Sort by success count and quality improvement
        successful_patterns.sort(
            key=lambda p: (p["success_count"], p["avg_quality_improvement"]),
            reverse=True
        )
        
        # Generate recommendations
        recommendations = []
        
        if successful_patterns:
            top_pattern = successful_patterns[0]
            recommendations.append(
                f"Use '{top_pattern['pattern']}' pattern - "
                f"successful in {top_pattern['success_count']} sessions "
                f"with {top_pattern['avg_quality_improvement']:.2%} avg quality improvement"
            )
        
        if unsuccessful:
            # Analyze what went wrong
            common_errors = {}
            for session in unsuccessful:
                for error in session.errors:
                    error_type = error.get("type", "unknown")
                    common_errors[error_type] = common_errors.get(error_type, 0) + 1
            
            if common_errors:
                most_common = max(common_errors.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Avoid {most_common[0]} errors - occurred in {most_common[1]} failed sessions"
                )
        
        # Get lessons from successful sessions
        all_lessons = []
        for session in successful:
            all_lessons.extend(session.lessons)
        
        if all_lessons:
            unique_lessons = list(set(all_lessons))
            recommendations.extend([f"Lesson: {lesson}" for lesson in unique_lessons[:3]])
        
        return {
            "similar_sessions": [s.to_dict() for s in similar],
            "successful_sessions": [s.to_dict() for s in successful],
            "unsuccessful_sessions": [s.to_dict() for s in unsuccessful],
            "successful_patterns": successful_patterns,
            "quality_trends": self._get_quality_summary(successful),
            "recommendations": recommendations
        }
    
    def _get_quality_summary(self, sessions: List[SessionRecord]) -> Dict[str, Any]:
        """Get quality summary for successful sessions."""
        if not sessions:
            return {}
        
        all_entropies = []
        all_tests = 0
        
        for session in sessions:
            for file_path, entropy in session.entropy_changes.items():
                all_entropies.append(entropy)
            all_tests += session.tests_passed
        
        return {
            "avg_entropy": sum(all_entropies) / max(len(all_entropies), 1),
            "total_tests": all_tests,
            "sessions_analyzed": len(sessions)
        }
    
    def get_best_approach(self, task_type: str) -> Optional[SuccessPattern]:
        """Get the best approach for a task type.
        
        Args:
            task_type: Type of task (create, modify, refactor, etc.)
            
        Returns:
            Best success pattern or None
        """
        # Get all sessions of this type
        sessions = [
            s for s in self.memory._sessions
            if s.task_type == task_type and s.success
        ]
        
        if not sessions:
            return None
        
        # Count patterns
        pattern_stats = {}
        for session in sessions:
            for pattern in session.patterns_used:
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {
                        "count": 0,
                        "quality_improvements": []
                    }
                pattern_stats[pattern]["count"] += 1
                
                # Check quality trends
                for file_path in session.files_created:
                    trend = self.quality.get_trend(file_path)
                    if trend and trend.trend == "improving":
                        pattern_stats[pattern]["quality_improvements"].append(
                            -trend.trend_slope
                        )
        
        # Find best pattern
        best_pattern = None
        best_score = 0
        
        for pattern_name, stats in pattern_stats.items():
            avg_improvement = (
                sum(stats["quality_improvements"]) / len(stats["quality_improvements"])
                if stats["quality_improvements"] else 0.0
            )
            
            # Score = success_count * (1 + avg_improvement)
            score = stats["count"] * (1.0 + avg_improvement)
            
            if score > best_score:
                best_score = score
                best_pattern = pattern_name
        
        if not best_pattern:
            return None
        
        # Find example session
        example_session = next(
            s for s in sessions
            if best_pattern in s.patterns_used
        )
        
        return SuccessPattern(
            pattern_name=best_pattern,
            task_type=task_type,
            success_count=pattern_stats[best_pattern]["count"],
            avg_quality_improvement=best_score / pattern_stats[best_pattern]["count"] - 1,
            example=f"See session {example_session.session_id}",
            last_used=example_session.timestamp,
            files_created=example_session.files_created
        )


class PatternRecommender:
    """Recommend patterns based on success analysis.
    
    MYCO Phase 2.3: Suggest best patterns for task.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.analyzer = SuccessAnalyzer(project_root)
        self.library = PatternLibrary(project_root)
    
    def recommend_for_task(self, task_description: str, task_type: str) -> List[Dict[str, Any]]:
        """Get pattern recommendations for a task.
        
        Args:
            task_description: Task description
            task_type: Task type (create, modify, refactor, etc.)
            
        Returns:
            List of pattern recommendations
        """
        recommendations = []
        
        # Get success analysis
        analysis = self.analyzer.analyze_task_success(task_description)
        
        # Add successful patterns
        for pattern in analysis.get("successful_patterns", [])[:3]:
            recommendations.append({
                "type": "successful_pattern",
                "pattern": pattern["pattern"],
                "reason": f"Successful in {pattern['success_count']} sessions",
                "quality_impact": f"{pattern['avg_quality_improvement']:.2%} avg improvement",
                "confidence": "high" if pattern["success_count"] >= 3 else "medium"
            })
        
        # Get best approach for task type
        best = self.analyzer.get_best_approach(task_type)
        if best:
            recommendations.append({
                "type": "best_approach",
                "pattern": best.pattern_name,
                "reason": f"Best for {task_type} tasks",
                "quality_impact": f"{best.avg_quality_improvement:.2%} improvement",
                "confidence": "high"
            })
        
        # Get patterns from library
        library_patterns = self.library.get_patterns_for(task_type)
        for pattern in library_patterns[:2]:
            if pattern["name"] not in [r["pattern"] for r in recommendations]:
                recommendations.append({
                    "type": "library_pattern",
                    "pattern": pattern["name"],
                    "reason": pattern["description"],
                    "quality_impact": f"Used {pattern.get('success_count', 0)} times",
                    "confidence": "medium"
                })
        
        return recommendations
    
    def warn_about_antipatterns(self, task_type: str) -> List[Dict[str, Any]]:
        """Get warnings about anti-patterns to avoid.
        
        Args:
            task_type: Task type
            
        Returns:
            List of warnings
        """
        from .session_memory import AntiPatternTracker
        
        tracker = AntiPatternTracker(self.project_root)
        warnings = tracker.get_warnings_for(task_type)
        
        return [
            {
                "type": "antipattern_warning",
                "pattern": w["name"],
                "reason": w["consequence"],
                "avoid": w["fix"],
                "confidence": "high"
            }
            for w in warnings
        ]


class QualityAdvisor:
    """Advise on quality based on trends.
    
    MYCO Phase 2.3: Quality-driven recommendations.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.quality = QualityTracker(project_root)
    
    def get_quality_advice(self, file_path: str) -> Dict[str, Any]:
        """Get quality advice for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dict with quality advice
        """
        trend = self.quality.get_trend(file_path)
        
        if not trend:
            return {
                "file": file_path,
                "advice": "No quality data yet. Create the file to start tracking.",
                "action": "proceed"
            }
        
        if trend.trend == "improving":
            return {
                "file": file_path,
                "advice": f"Quality improving (H: {trend.max_entropy:.2f} → {trend.avg_entropy:.2f})",
                "action": "continue",
                "confidence": "high"
            }
        
        elif trend.trend == "stable":
            return {
                "file": file_path,
                "advice": f"Quality stable (H: {trend.avg_entropy:.2f})",
                "action": "monitor",
                "confidence": "medium"
            }
        
        else:  # degrading
            return {
                "file": file_path,
                "advice": f"Quality degrading (H: {trend.min_entropy:.2f} → {trend.avg_entropy:.2f})",
                "action": "refactor",
                "reason": "Entropy increasing over time",
                "confidence": "high"
            }
    
    def get_project_advice(self) -> Dict[str, Any]:
        """Get overall project quality advice.
        
        Returns:
            Dict with project-level advice
        """
        health = self.quality.get_project_health()
        
        if health["health_regime"] == "excellent":
            return {
                "regime": "excellent",
                "advice": "Project health is excellent. Continue current practices.",
                "action": "continue"
            }
        
        elif health["health_regime"] == "healthy":
            return {
                "regime": "healthy",
                "advice": "Project health is good. Monitor degrading files.",
                "action": "monitor"
            }
        
        elif health["health_regime"] == "moderate":
            return {
                "regime": "moderate",
                "advice": f"Project needs attention. {health['degrading']} files degrading.",
                "action": "improve"
            }
        
        else:  # needs_attention
            return {
                "regime": "needs_attention",
                "advice": f"Project needs immediate attention. Health score: {health['health_score']:.2f}",
                "action": "refactor",
                "priority": "high"
            }


# ============================================================================
# Agent Tools
# ============================================================================

def analyze_task_success(project_root: str, task: str) -> dict:
    """Analyze what worked for similar tasks (agent tool)."""
    try:
        analyzer = SuccessAnalyzer(Path(project_root))
        return analyzer.analyze_task_success(task)
    except Exception as e:
        return {"error": str(e)}


def get_pattern_recommendations(project_root: str, task: str, task_type: str) -> list:
    """Get pattern recommendations for task (agent tool)."""
    try:
        recommender = PatternRecommender(Path(project_root))
        return recommender.recommend_for_task(task, task_type)
    except Exception:
        return []


def get_antipattern_warnings(project_root: str, task_type: str) -> list:
    """Get anti-pattern warnings (agent tool)."""
    try:
        recommender = PatternRecommender(Path(project_root))
        return recommender.warn_about_antipatterns(task_type)
    except Exception:
        return []


def get_quality_advice(project_root: str, file_path: str) -> dict:
    """Get quality advice for file (agent tool)."""
    try:
        advisor = QualityAdvisor(Path(project_root))
        return advisor.get_quality_advice(file_path)
    except Exception:
        return {"error": str(e)}


def get_project_quality_advice(project_root: str) -> dict:
    """Get project-level quality advice (agent tool)."""
    try:
        advisor = QualityAdvisor(Path(project_root))
        return advisor.get_project_advice()
    except Exception:
        return {"error": str(e)}
