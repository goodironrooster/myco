# ⊕ H:0.25 | press:session_memory | age:0 | drift:+0.00
"""MYCO Session Memory - Agents learn from past sessions.

MYCO Vision:
- Stigmergic learning (traces guide future agents)
- Cross-session continuity (remember what worked)
- Pattern extraction (build library of successful approaches)
- Mistake tracking (don't repeat errors)

Architecture:
- Session logs (.myco/sessions/) - Per-session records
- Pattern library (.myco/patterns.json) - Successful patterns
- Quality trends (.myco/quality.json) - Quality over time
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any


@dataclass
class SessionRecord:
    """Record of a single agent session."""
    
    session_id: str
    timestamp: str
    task: str
    task_type: str  # "create", "modify", "refactor", "fix", "test"
    
    # Files
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    
    # Quality metrics
    entropy_changes: Dict[str, float] = field(default_factory=dict)
    tests_created: int = 0
    tests_passed: int = 0
    
    # Errors and recovery
    errors: List[Dict[str, Any]] = field(default_factory=list)
    errors_recovered: int = 0
    
    # Patterns used
    patterns_used: List[str] = field(default_factory=list)
    
    # Outcome
    success: bool = True
    lessons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "SessionRecord":
        return cls(**data)


class SessionMemory:
    """Store and retrieve session memories.
    
    MYCO Phase 2.1: Agents learn from past sessions.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.sessions_dir = self.project_root / ".myco" / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._sessions: List[SessionRecord] = []
        self._load_sessions()
    
    def _load_sessions(self):
        """Load sessions from disk."""
        session_files = list(self.sessions_dir.glob("*.json"))
        
        for session_file in session_files:
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._sessions.append(SessionRecord.from_dict(data))
            except Exception:
                pass  # Skip corrupted files
        
        # Sort by timestamp (newest first)
        self._sessions.sort(key=lambda s: s.timestamp, reverse=True)
    
    def record_session(self, record: SessionRecord):
        """Record a session outcome.
        
        Args:
            record: Session record to save
        """
        # Save to disk
        session_file = self.sessions_dir / f"{record.session_id}.json"
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(record.to_dict(), f, indent=2)
        
        # Add to cache
        self._sessions.insert(0, record)
    
    def find_similar_sessions(self, task_description: str, limit: int = 5) -> List[SessionRecord]:
        """Find past sessions with similar tasks.
        
        Uses simple keyword matching. For production, use embeddings.
        
        Args:
            task_description: Current task description
            limit: Maximum number of sessions to return
            
        Returns:
            List of similar session records
        """
        # Extract keywords from task
        keywords = set(task_description.lower().split())
        
        # Score sessions by keyword overlap
        scored = []
        for session in self._sessions:
            session_words = set(session.task.lower().split())
            overlap = len(keywords & session_words)
            
            if overlap > 0:
                scored.append((overlap, session))
        
        # Sort by score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [session for _, session in scored[:limit]]
    
    def get_lessons(self, task_description: str) -> List[str]:
        """Get lessons from similar past sessions.
        
        Args:
            task_description: Current task description
            
        Returns:
            List of lessons learned
        """
        similar = self.find_similar_sessions(task_description, limit=5)
        
        lessons = []
        for session in similar:
            lessons.extend(session.lessons)
        
        # Remove duplicates
        return list(set(lessons))
    
    def get_successful_patterns(self, task_type: str) -> List[str]:
        """Get patterns that worked for similar tasks.
        
        Args:
            task_type: Type of task (create, modify, refactor, etc.)
            
        Returns:
            List of successful patterns
        """
        patterns = []
        
        for session in self._sessions:
            if session.task_type == task_type and session.success:
                patterns.extend(session.patterns_used)
        
        # Count pattern frequency
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Return most common patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [pattern for pattern, _ in sorted_patterns[:10]]
    
    def get_session_count(self) -> int:
        """Get total number of sessions."""
        return len(self._sessions)
    
    def get_success_rate(self, task_type: str = None) -> float:
        """Get success rate for sessions.
        
        Args:
            task_type: Optional task type filter
            
        Returns:
            Success rate (0.0 to 1.0)
        """
        if task_type:
            sessions = [s for s in self._sessions if s.task_type == task_type]
        else:
            sessions = self._sessions
        
        if not sessions:
            return 1.0
        
        successful = sum(1 for s in sessions if s.success)
        return successful / len(sessions)


class PatternLibrary:
    """Store and retrieve coding patterns.
    
    MYCO Phase 2.2: Build library of successful approaches.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.patterns_file = self.project_root / ".myco" / "patterns.json"
        self.patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._patterns: Dict[str, Dict[str, Any]] = {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load patterns from disk."""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    self._patterns = json.load(f)
            except Exception:
                self._patterns = {}
    
    def _save_patterns(self):
        """Save patterns to disk."""
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(self._patterns, f, indent=2)
    
    def add_pattern(self, name: str, description: str, example: str, 
                    task_type: str, success_count: int = 1):
        """Add or update a pattern.
        
        Args:
            name: Pattern name (e.g., "service_layer")
            description: Pattern description
            example: Code example
            task_type: Type of task this pattern helps with
            success_count: Number of times this pattern succeeded
        """
        if name in self._patterns:
            # Update existing pattern
            pattern = self._patterns[name]
            pattern["success_count"] = pattern.get("success_count", 0) + success_count
            pattern["last_used"] = datetime.utcnow().isoformat() + "Z"
        else:
            # Add new pattern
            self._patterns[name] = {
                "name": name,
                "description": description,
                "example": example,
                "task_type": task_type,
                "success_count": success_count,
                "created": datetime.utcnow().isoformat() + "Z",
                "last_used": datetime.utcnow().isoformat() + "Z"
            }
        
        self._save_patterns()
    
    def get_patterns_for(self, task_type: str) -> List[Dict[str, Any]]:
        """Get patterns for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            List of pattern dicts, sorted by success count
        """
        patterns = [
            p for p in self._patterns.values()
            if p.get("task_type") == task_type
        ]
        
        # Sort by success count (most successful first)
        patterns.sort(key=lambda p: p.get("success_count", 0), reverse=True)
        
        return patterns
    
    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns.
        
        Returns:
            List of all pattern dicts
        """
        return list(self._patterns.values())


class AntiPatternTracker:
    """Track mistakes to avoid repeating them.
    
    MYCO Phase 2.3: Learn from mistakes.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.antipatterns_file = self.project_root / ".myco" / "antipatterns.json"
        self.antipatterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._antipatterns: Dict[str, Dict[str, Any]] = {}
        self._load_antipatterns()
    
    def _load_antipatterns(self):
        """Load anti-patterns from disk."""
        if self.antipatterns_file.exists():
            try:
                with open(self.antipatterns_file, 'r', encoding='utf-8') as f:
                    self._antipatterns = json.load(f)
            except Exception:
                self._antipatterns = {}
    
    def _save_antipatterns(self):
        """Save anti-patterns to disk."""
        with open(self.antipatterns_file, 'w', encoding='utf-8') as f:
            json.dump(self._antipatterns, f, indent=2)
    
    def record_mistake(self, name: str, description: str, consequence: str, 
                       fix: str, task_type: str):
        """Record a mistake to avoid.
        
        Args:
            name: Mistake name (e.g., "god_class")
            description: What was done wrong
            consequence: What went wrong
            fix: How to fix/avoid
            task_type: Type of task where this occurred
        """
        if name in self._antipatterns:
            # Update existing
            antipattern = self._antipatterns[name]
            antipattern["occurrence_count"] = antipattern.get("occurrence_count", 0) + 1
            antipattern["last_seen"] = datetime.utcnow().isoformat() + "Z"
        else:
            # Add new
            self._antipatterns[name] = {
                "name": name,
                "description": description,
                "consequence": consequence,
                "fix": fix,
                "task_type": task_type,
                "occurrence_count": 1,
                "created": datetime.utcnow().isoformat() + "Z",
                "last_seen": datetime.utcnow().isoformat() + "Z"
            }
        
        self._save_antipatterns()
    
    def get_warnings_for(self, task_type: str) -> List[Dict[str, Any]]:
        """Get warnings for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            List of anti-pattern warnings
        """
        warnings = [
            ap for ap in self._antipatterns.values()
            if ap.get("task_type") == task_type
        ]
        
        # Sort by occurrence count (most common first)
        warnings.sort(key=lambda w: w.get("occurrence_count", 0), reverse=True)
        
        return warnings


# ============================================================================
# Agent Tools
# ============================================================================

def record_session(project_root: str, session_data: dict) -> bool:
    """Record session outcome (agent tool)."""
    try:
        memory = SessionMemory(Path(project_root))
        record = SessionRecord.from_dict(session_data)
        memory.record_session(record)
        return True
    except Exception:
        return False


def get_similar_sessions(project_root: str, task: str, limit: int = 3) -> List[dict]:
    """Get similar past sessions (agent tool)."""
    try:
        memory = SessionMemory(Path(project_root))
        sessions = memory.find_similar_sessions(task, limit)
        return [s.to_dict() for s in sessions]
    except Exception:
        return []


def get_lessons(project_root: str, task: str) -> List[str]:
    """Get lessons from past sessions (agent tool)."""
    try:
        memory = SessionMemory(Path(project_root))
        return memory.get_lessons(task)
    except Exception:
        return []


def add_pattern(project_root: str, name: str, description: str, 
                example: str, task_type: str) -> bool:
    """Add successful pattern (agent tool)."""
    try:
        library = PatternLibrary(Path(project_root))
        library.add_pattern(name, description, example, task_type)
        return True
    except Exception:
        return False


def get_patterns(project_root: str, task_type: str) -> List[dict]:
    """Get patterns for task type (agent tool)."""
    try:
        library = PatternLibrary(Path(project_root))
        return library.get_patterns_for(task_type)
    except Exception:
        return []


def record_mistake(project_root: str, name: str, description: str,
                   consequence: str, fix: str, task_type: str) -> bool:
    """Record mistake to avoid (agent tool)."""
    try:
        tracker = AntiPatternTracker(Path(project_root))
        tracker.record_mistake(name, description, consequence, fix, task_type)
        return True
    except Exception:
        return False


def get_warnings(project_root: str, task_type: str) -> List[dict]:
    """Get warnings for task type (agent tool)."""
    try:
        tracker = AntiPatternTracker(Path(project_root))
        return tracker.get_warnings_for(task_type)
    except Exception:
        return []
