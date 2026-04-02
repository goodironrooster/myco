# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Session logging for myco.

Logs all session activity to .myco/session.log.
Provides audit trail and debugging information.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: str
    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "event_type": self.event_type,
            "message": self.message,
            "data": self.data,
        }
    
    def to_json(self) -> str:
        """Convert to JSON line."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_line: str) -> "LogEntry":
        """Create from JSON line."""
        data = json.loads(json_line)
        return cls(
            timestamp=data["timestamp"],
            level=data["level"],
            event_type=data["event_type"],
            message=data["message"],
            data=data.get("data", {}),
        )


class SessionLogger:
    """Session logger for myco.
    
    Logs all session activity to .myco/session.log in JSON Lines format.
    Each session gets a unique session ID.
    """
    
    def __init__(self, project_root: Path | str, session_id: Optional[str] = None):
        """Initialize the session logger.
        
        Args:
            project_root: Root directory of the project
            session_id: Optional session ID (generated if not provided)
        """
        self.project_root = Path(project_root)
        self.log_path = self.project_root / ".myco" / "session.log"
        self.session_id = session_id or self._generate_session_id()
        self._entries: list[LogEntry] = []
        
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def log(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        **data: Any
    ) -> LogEntry:
        """Log an event.
        
        Args:
            event_type: Type of event (e.g., "tool_call", "gate_check", "entropy_change")
            message: Human-readable message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            **data: Additional data to log
            
        Returns:
            The created LogEntry
        """
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level,
            event_type=event_type,
            message=message,
            data={
                "session_id": self.session_id,
                **data
            }
        )
        
        self._entries.append(entry)
        
        # Write to file immediately
        self._write_entry(entry)
        
        return entry
    
    def _write_entry(self, entry: LogEntry) -> None:
        """Write a single entry to the log file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry.to_json() + "\n")
    
    def debug(self, event_type: str, message: str, **data: Any) -> LogEntry:
        """Log a debug event."""
        return self.log(event_type, message, level="DEBUG", **data)
    
    def info(self, event_type: str, message: str, **data: Any) -> LogEntry:
        """Log an info event."""
        return self.log(event_type, message, level="INFO", **data)
    
    def warning(self, event_type: str, message: str, **data: Any) -> LogEntry:
        """Log a warning event."""
        return self.log(event_type, message, level="WARNING", **data)
    
    def error(self, event_type: str, message: str, **data: Any) -> LogEntry:
        """Log an error event."""
        return self.log(event_type, message, level="ERROR", **data)
    
    def log_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        success: bool = True
    ) -> LogEntry:
        """Log a tool call."""
        return self.info(
            "tool_call",
            f"Tool '{tool_name}' executed: {'success' if success else 'failed'}",
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            success=success
        )
    
    def log_gate_check(
        self,
        file_path: str,
        action_type: str,
        permitted: bool,
        reason: str = "",
        entropy_before: float = 0.0,
        entropy_after: float = 0.0
    ) -> LogEntry:
        """Log a gate check."""
        return self.info(
            "gate_check",
            f"Gate {'PERMIT' if permitted else 'BLOCK'}: {action_type} on {file_path}",
            file_path=file_path,
            action_type=action_type,
            permitted=permitted,
            reason=reason,
            entropy_before=entropy_before,
            entropy_after=entropy_after
        )
    
    def log_entropy_change(
        self,
        before: float,
        after: float,
        modules_affected: list[str]
    ) -> LogEntry:
        """Log an entropy change."""
        delta = after - before
        return self.info(
            "entropy_change",
            f"Entropy changed: {before:.3f} → {after:.3f} (Δ={delta:+.3f})",
            entropy_before=before,
            entropy_after=after,
            entropy_delta=delta,
            modules_affected=modules_affected
        )
    
    def log_attractor_event(
        self,
        attractor_name: str,
        perturbation: str,
        turn_detected: int
    ) -> LogEntry:
        """Log an attractor detection event."""
        return self.warning(
            "attractor_detected",
            f"Attractor '{attractor_name}' detected, applying '{perturbation}'",
            attractor_name=attractor_name,
            perturbation=perturbation,
            turn_detected=turn_detected
        )
    
    def log_tensegrity_violation(
        self,
        importer: str,
        imported: str,
        violation_type: str
    ) -> LogEntry:
        """Log a tensegrity violation."""
        return self.warning(
            "tensegrity_violation",
            f"Tensegrity violation: {importer} → {imported}",
            importer=importer,
            imported=imported,
            violation_type=violation_type
        )
    
    def log_session_start(self, task: str) -> LogEntry:
        """Log session start."""
        return self.info(
            "session_start",
            f"Session started: {task[:100]}",
            task=task
        )
    
    def log_session_end(
        self,
        iterations: int,
        tokens: int,
        joules: float,
        entropy_delta: float,
        files_modified: Optional[list[str]] = None
    ) -> LogEntry:
        """Log session end."""
        return self.info(
            "session_end",
            f"Session completed: {iterations} iterations, {len(files_modified) if files_modified else 0} files modified",
            iterations=iterations,
            tokens=tokens,
            joules=joules,
            entropy_delta=entropy_delta,
            files_modified=files_modified or []
        )
    
    def get_entries(self, filter_type: Optional[str] = None) -> list[LogEntry]:
        """Get log entries, optionally filtered by type.
        
        Args:
            filter_type: Filter by event type (optional)
            
        Returns:
            List of LogEntry objects
        """
        if filter_type:
            return [e for e in self._entries if e.event_type == filter_type]
        return self._entries.copy()
    
    def read_log_file(self) -> list[LogEntry]:
        """Read all entries from the log file.
        
        Returns:
            List of LogEntry objects from file
        """
        entries = []
        
        if not self.log_path.exists():
            return entries
        
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(LogEntry.from_json(line))
                    except json.JSONDecodeError:
                        continue
        
        return entries
    
    def get_session_entries(self) -> list[LogEntry]:
        """Get entries for the current session only.
        
        Returns:
            List of LogEntry objects for current session
        """
        all_entries = self.read_log_file()
        return [
            e for e in all_entries
            if e.data.get("session_id") == self.session_id
        ]
    
    def clear_log(self) -> None:
        """Clear the log file."""
        if self.log_path.exists():
            self.log_path.unlink()
        self._entries = []


# Global logger instance
_logger: Optional[SessionLogger] = None


def get_logger(project_root: Optional[Path | str] = None) -> SessionLogger:
    """Get the global session logger.
    
    Args:
        project_root: Root directory (uses cwd if not provided)
        
    Returns:
        SessionLogger instance
    """
    global _logger
    if _logger is None:
        root = Path(project_root) if project_root else Path.cwd()
        _logger = SessionLogger(root)
    return _logger


def log_session_start(task: str) -> LogEntry:
    """Log session start using global logger."""
    return get_logger().log_session_start(task)


def log_session_end(
    iterations: int,
    tokens: int,
    joules: float,
    entropy_delta: float,
    files_modified: Optional[list[str]] = None
) -> LogEntry:
    """Log session end using global logger."""
    return get_logger().log_session_end(iterations, tokens, joules, entropy_delta, files_modified)
