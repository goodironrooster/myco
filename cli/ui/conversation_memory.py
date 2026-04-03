"""MYCO Conversation Memory - Session tracking with stigmergic awareness.

Features:
- Action history with verification status
- Entropy tracking per action
- Time stamps and attribution
- Pattern detection
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class ActionRecord:
    """Record of a single action in the conversation."""
    
    tool_name: str
    arguments: dict
    success: bool
    verified: bool = False
    entropy_before: float = 0.0
    entropy_after: float = 0.0
    entropy_delta: float = 0.0
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    error: Optional[str] = None
    file_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ActionRecord":
        """Create from dictionary."""
        return cls(**data)


class ConversationMemory:
    """Tracks conversation actions with stigmergic awareness."""
    
    def __init__(self, project_root: Path, max_history: int = 100):
        """Initialize conversation memory.
        
        Args:
            project_root: Root directory of the project
            max_history: Maximum number of actions to keep in memory
        """
        self.project_root = project_root
        self.max_history = max_history
        self.console = Console()
        
        # Action history
        self.actions: list[ActionRecord] = []
        
        # Session metadata
        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.task_description: Optional[str] = None
        
        # Pattern tracking
        self.tool_usage: dict[str, int] = {}
        self.error_patterns: list[dict] = []
        self.success_rate: float = 1.0
        
        # Try to load MYCO vision
        self.has_myco = False
        try:
            from myco.entropy import calculate_substrate_health
            self.calculate_health = calculate_substrate_health
            self.has_myco = True
        except Exception:
            self.calculate_health = None
    
    def record_action(
        self,
        tool_name: str,
        arguments: dict,
        success: bool,
        verified: bool = False,
        error: Optional[str] = None,
        duration: float = 0.0,
    ):
        """Record an action in the conversation memory.
        
        Args:
            tool_name: Name of the tool used
            arguments: Tool arguments
            success: Whether the action succeeded
            verified: Whether the result was verified
            error: Error message if failed
            duration: Action duration in seconds
        """
        # Extract file path if present
        file_path = arguments.get("path") or arguments.get("file_path")
        
        # Calculate entropy delta for file operations
        entropy_before = 0.0
        entropy_after = 0.0
        entropy_delta = 0.0
        
        if self.has_myco and file_path and file_path.endswith('.py'):
            try:
                from myco.entropy import ImportGraphBuilder, EntropyCalculator
                builder = ImportGraphBuilder(self.project_root)
                graph = builder.build()
                calc = EntropyCalculator(graph)
                
                # Try to get entropy before/after
                module_name = str(Path(file_path).relative_to(self.project_root))
                module_name = module_name.replace('/', '.').replace('\\', '.').replace('.py', '')
                
                # For now, just track that entropy was measured
                entropy_after = 0.5  # Placeholder
                entropy_delta = 0.0
            except Exception:
                pass
        
        # Create action record
        action = ActionRecord(
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            verified=verified,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            entropy_delta=entropy_delta,
            timestamp=time.time(),
            duration=duration,
            error=error,
            file_path=file_path,
        )
        
        # Add to history
        self.actions.append(action)
        
        # Trim if needed
        if len(self.actions) > self.max_history:
            self.actions = self.actions[-self.max_history:]
        
        # Update statistics
        self._update_stats(tool_name, success)
    
    def _update_stats(self, tool_name: str, success: bool):
        """Update usage statistics.
        
        Args:
            tool_name: Name of the tool used
            success: Whether the action succeeded
        """
        # Track tool usage
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1
        
        # Calculate success rate
        total = len(self.actions)
        successes = sum(1 for a in self.actions if a.success)
        self.success_rate = successes / total if total > 0 else 1.0
        
        # Track error patterns
        if not success:
            self.error_patterns.append({
                "tool": tool_name,
                "timestamp": time.time(),
            })
    
    def get_recent_actions(self, limit: int = 10) -> list[ActionRecord]:
        """Get most recent actions.
        
        Args:
            limit: Maximum number of actions to return
            
        Returns:
            List of recent action records
        """
        return self.actions[-limit:]
    
    def get_failed_actions(self) -> list[ActionRecord]:
        """Get all failed actions.
        
        Returns:
            List of failed action records
        """
        return [a for a in self.actions if not a.success]
    
    def get_verified_actions(self) -> list[ActionRecord]:
        """Get all verified actions.
        
        Returns:
            List of verified action records
        """
        return [a for a in self.actions if a.verified]
    
    def get_actions_for_file(self, file_path: str) -> list[ActionRecord]:
        """Get actions related to a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of action records for the file
        """
        return [a for a in self.actions if a.file_path == file_path]
    
    def detect_patterns(self) -> dict:
        """Detect patterns in the conversation.
        
        Returns:
            Dictionary with detected patterns
        """
        patterns = {
            "circular": False,
            "stalled": False,
            "scattered": False,
            "repeated_errors": [],
        }
        
        # Check for circular discussion (same tool used repeatedly without progress)
        if len(self.actions) >= 3:
            recent_tools = [a.tool_name for a in self.actions[-3:]]
            if len(set(recent_tools)) == 1:
                patterns["circular"] = True
        
        # Check for stalled conversation (no verified actions recently)
        recent_verified = sum(1 for a in self.actions[-5:] if a.verified)
        if recent_verified == 0 and len(self.actions) >= 5:
            patterns["stalled"] = True
        
        # Check for scattered focus (too many different files)
        recent_files = set(a.file_path for a in self.actions[-10:] if a.file_path)
        if len(recent_files) > 5:
            patterns["scattered"] = True
        
        # Check for repeated errors
        error_counts: dict[str, int] = {}
        for a in self.actions[-10:]:
            if a.error:
                key = f"{a.tool_name}: {a.error[:50]}"
                error_counts[key] = error_counts.get(key, 0) + 1
        
        patterns["repeated_errors"] = [
            {"pattern": k, "count": v}
            for k, v in error_counts.items()
            if v >= 2
        ]
        
        return patterns

    def compute_health(self) -> dict:
        """Compute conversation health metrics.

        Returns:
            Dict with coherence, progress, entropy scores and detected issues
        """
        patterns = self.detect_patterns()

        # Coherence: how focused is the conversation?
        coherence = 1.0
        if patterns["circular"]:
            coherence -= 0.3
        if patterns["scattered"]:
            coherence -= 0.2
        if patterns["repeated_errors"]:
            coherence -= 0.1 * len(patterns["repeated_errors"])
        coherence = max(0.0, min(1.0, coherence))

        # Progress: ratio of verified actions
        progress = self.success_rate
        if patterns["stalled"]:
            progress *= 0.6
        progress = max(0.0, min(1.0, progress))

        # Entropy: inverse of error rate
        total = len(self.actions)
        errors = sum(1 for a in self.actions if a.error)
        entropy_score = 1.0 - (errors / total if total > 0 else 0.0)

        # Generate suggestions
        suggestions = []
        if patterns["circular"]:
            suggestions.append("💡 Summarize decisions and move to implementation")
        if patterns["stalled"]:
            suggestions.append("💡 Try a different approach or break the task into smaller steps")
        if patterns["scattered"]:
            suggestions.append("💡 Focus on one file at a time")
        if patterns["repeated_errors"]:
            for err in patterns["repeated_errors"]:
                suggestions.append(f"💡 Repeated error in {err['pattern']} — consider a different tool or approach")

        return {
            "coherence": coherence,
            "progress": progress,
            "entropy_score": entropy_score,
            "patterns": patterns,
            "suggestions": suggestions,
            "has_issues": bool(suggestions),
        }

    def render_health_panel(self) -> Panel:
        """Render conversation health monitor panel.

        Returns:
            Rich Panel with health metrics and suggestions
        """
        health = self.compute_health()

        def bar(value: float, width: int = 10) -> str:
            filled = int(value * width)
            empty = width - filled
            return "▓" * filled + "░" * empty

        def status(value: float) -> str:
            if value >= 0.7:
                return f"[green]✓ Good[/green]"
            elif value >= 0.4:
                return f"[yellow]⚠ Moderate[/yellow]"
            else:
                return f"[red]✗ Poor[/red]"

        lines = [
            f"Coherence: [{bar(health['coherence'])}] {health['coherence']:.2f}  {status(health['coherence'])}",
            f"Progress:  [{bar(health['progress'])}] {health['progress']:.2f}  {status(health['progress'])}",
            f"Entropy:   [{bar(health['entropy_score'])}] {health['entropy_score']:.2f}  {status(health['entropy_score'])}",
        ]

        if health["suggestions"]:
            lines.append("")
            for s in health["suggestions"]:
                lines.append(s)

        border = "green" if not health["has_issues"] else "yellow"
        panel = Panel(
            "\n".join(lines),
            title="[bold cyan]Conversation Health[/bold cyan]",
            border_style=border,
        )

        return panel

    def render_panel(self, limit: int = 10) -> Panel:
        """Render conversation memory as a panel.
        
        Args:
            limit: Number of recent actions to show
            
        Returns:
            Rich Panel with conversation memory
        """
        recent = self.get_recent_actions(limit)
        
        # Create table
        table = Table(show_header=True, header_style="bold cyan", show_lines=False)
        table.add_column("Status", style="dim", width=4)
        table.add_column("Tool", style="cyan", width=20)
        table.add_column("Target", style="white", width=30)
        table.add_column("Entropy", style="yellow", width=12)
        table.add_column("Time", style="dim", width=8)
        
        current_time = time.time()
        for action in recent:
            # Status icon
            if action.verified:
                status = "[green]✓[/green]"
            elif action.success:
                status = "[dim]○[/dim]"
            else:
                status = "[red]✗[/red]"
            
            # Tool name
            tool = action.tool_name
            
            # Target (file or first argument)
            if action.file_path:
                target = Path(action.file_path).name
            elif action.arguments:
                first_arg = list(action.arguments.values())[0]
                target = str(first_arg)[:25] + "..." if len(str(first_arg)) > 25 else str(first_arg)
            else:
                target = "-"
            
            # Entropy delta
            if action.entropy_delta != 0.0:
                delta_str = f"{action.entropy_delta:+.2f}"
                if action.entropy_delta > 0:
                    entropy = f"[yellow]{delta_str}[/yellow]"
                else:
                    entropy = f"[green]{delta_str}[/green]"
            else:
                entropy = "-"
            
            # Time ago
            elapsed = current_time - action.timestamp
            if elapsed < 60:
                time_str = f"{int(elapsed)}s"
            elif elapsed < 3600:
                time_str = f"{int(elapsed/60)}m"
            else:
                time_str = f"{int(elapsed/3600)}h"
            
            table.add_row(status, tool, target, entropy, time_str)
        
        # Calculate session stats
        session_time = current_time - self.session_start
        verified_count = len(self.get_verified_actions())
        failed_count = len(self.get_failed_actions())
        
        # Create summary text
        summary_lines = [
            f"[bold]Session:[/bold] {self.session_id} | [bold]Duration:[/bold] {int(session_time/60)}m {int(session_time%60)}s",
            f"[bold]Actions:[/bold] {len(self.actions)} | [green]Verified:[/green] {verified_count} | [red]Failed:[/red] {failed_count}",
            f"[bold]Success Rate:[/bold] {self.success_rate:.0%}",
        ]
        
        # Detect patterns
        patterns = self.detect_patterns()
        if patterns["circular"]:
            summary_lines.append("[yellow]⚠ Pattern: Circular discussion detected[/yellow]")
        if patterns["stalled"]:
            summary_lines.append("[yellow]⚠ Pattern: Conversation stalled (no verified actions)[/yellow]")
        if patterns["scattered"]:
            summary_lines.append("[yellow]⚠ Pattern: Scattered focus (too many files)[/yellow]")
        if patterns["repeated_errors"]:
            for err in patterns["repeated_errors"]:
                summary_lines.append(f"[red]⚠ Repeated error: {err['pattern']}[/red]")
        
        # Combine summary and table
        from io import StringIO
        from rich.console import Console as RichConsole
        
        buffer = StringIO()
        temp_console = RichConsole(file=buffer, force_terminal=True)
        temp_console.print(table)
        table_text = buffer.getvalue()
        
        content = "\n".join(summary_lines) + "\n\n" + table_text
        
        panel = Panel(
            content,
            title="[bold cyan]Conversation Memory[/bold cyan]",
            border_style="cyan",
        )
        
        return panel
    
    def save(self, path: Optional[Path] = None):
        """Save conversation memory to file.
        
        Args:
            path: Path to save to (default: .myco-internal/sessions/{session_id}.json)
        """
        if path is None:
            path = self.project_root / ".myco-internal" / "sessions" / f"{self.session_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "task_description": self.task_description,
            "actions": [a.to_dict() for a in self.actions],
            "tool_usage": self.tool_usage,
            "success_rate": self.success_rate,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, session_id: str):
        """Load conversation memory from file.
        
        Args:
            session_id: ID of session to load
        """
        path = self.project_root / ".myco-internal" / "sessions" / f"{session_id}.json"
        
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.session_id = data["session_id"]
        self.session_start = data["session_start"]
        self.task_description = data.get("task_description")
        self.tool_usage = data.get("tool_usage", {})
        self.success_rate = data.get("success_rate", 1.0)
        self.actions = [ActionRecord.from_dict(a) for a in data.get("actions", [])]


# Global instance
_memory: Optional[ConversationMemory] = None


def get_conversation_memory(project_root: Path) -> ConversationMemory:
    """Get or create global conversation memory instance.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        ConversationMemory instance
    """
    global _memory
    if _memory is None or _memory.project_root != project_root:
        _memory = ConversationMemory(project_root)
    return _memory
