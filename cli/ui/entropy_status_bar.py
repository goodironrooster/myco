"""MYCO Entropy Status Bar - Real-time entropy visualization.

Displays:
- Current project entropy state
- Entropy budget status
- Visual entropy bar
- Iteration tracking
"""

from pathlib import Path
from typing import Optional, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class EntropyStatusBar:
    """Real-time entropy status bar for MYCO interactive mode."""
    
    def __init__(self, project_root: Path):
        """Initialize entropy status bar.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.console = Console()
        self.last_health: Optional[Any] = None
        self.current_file: Optional[str] = None
        self.current_entropy: float = 0.0
        self.entropy_delta: float = 0.0
        self.budget_status: str = "OK"
        self.iteration: int = 0
        self.max_iterations: int = 20
        
        # Try to load MYCO vision
        self.has_myco = False
        try:
            from myco.entropy import calculate_substrate_health, get_regime_intervention
            self.calculate_health = calculate_substrate_health
            self.get_regime_intervention = get_regime_intervention
            self.has_myco = True
        except Exception:
            self.calculate_health = None
            self.get_regime_intervention = None
    
    def _get_regime(self, h: float) -> str:
        """Get entropy regime from H value."""
        if h < 0.3:
            return "crystallized"
        elif h > 0.75:
            return "diffuse"
        else:
            return "dissipative"
    
    def _get_regime_color(self, regime: str) -> str:
        """Get color for entropy regime."""
        colors = {
            "crystallized": "blue",
            "dissipative": "green",
            "diffuse": "yellow",
        }
        return colors.get(regime, "white")
    
    def _create_entropy_bar(self, entropy: float, width: int = 10) -> str:
        """Create visual entropy bar.
        
        Args:
            entropy: Entropy value (0.0-1.0)
            width: Width of bar in characters
            
        Returns:
            String representation of entropy bar
        """
        filled = int(entropy * width)
        empty = width - filled
        
        # Use different characters for filled/empty
        if entropy < 0.3:
            bar = "▓" * filled + "░" * empty
        elif entropy < 0.75:
            bar = "▓" * filled + "░" * empty
        else:
            bar = "█" * filled + "░" * empty
        
        return bar
    
    def update(self, file_path: Optional[str] = None, content: Optional[str] = None):
        """Update status bar with current entropy state.
        
        Args:
            file_path: Path to file being modified (optional)
            content: Proposed content (optional)
        """
        if self.has_myco and self.calculate_health:
            try:
                # Refresh substrate health
                self.last_health = self.calculate_health(self.project_root)
                
                # Calculate entropy delta if file and content provided
                if file_path and content:
                    from myco.entropy import check_entropy_budget
                    path = Path(file_path)
                    current = path.read_text() if path.exists() else ""
                    within, curr_h, prop_h, msg = check_entropy_budget(current, content)
                    self.current_entropy = prop_h
                    self.entropy_delta = prop_h - curr_h
                    self.budget_status = "OK" if within else "EXCEEDED"
                    self.current_file = file_path
            except Exception:
                pass
    
    def render(self) -> Panel:
        """Render the status bar.
        
        Returns:
            Rich Panel with status bar content
        """
        lines = []
        
        # MYCO Vision status
        if self.has_myco and self.last_health:
            # Handle both dict and object returns
            if isinstance(self.last_health, dict):
                files = self.last_health.get('files', [])
                avg_entropy = self.last_health.get('avg_entropy', 0)
                crystallized = self.last_health.get('crystallized_count', 0)
                dissipative = self.last_health.get('dissipative_count', 0)
                diffuse = self.last_health.get('diffuse_count', 0)
            else:
                files = getattr(self.last_health, 'files', [])
                avg_entropy = getattr(self.last_health, 'avg_entropy', 0)
                crystallized = getattr(self.last_health, 'crystallized_count', 0)
                dissipative = getattr(self.last_health, 'dissipative_count', 0)
                diffuse = getattr(self.last_health, 'diffuse_count', 0)
            
            regime = self._get_regime(avg_entropy)
            regime_color = self._get_regime_color(regime)
            entropy_bar = self._create_entropy_bar(avg_entropy)
            
            lines.append(f"[bold]MYCO Substrate Health[/bold]")
            lines.append(f"Files: {len(files)} | Avg H: {avg_entropy:.2f} [{entropy_bar}] {regime}")
            lines.append(f"Crystallized: {crystallized} | Dissipative: {dissipative} | Diffuse: {diffuse}")
            
            # Current file entropy delta
            if self.current_file:
                delta_str = f"{self.entropy_delta:+.2f}"
                if self.entropy_delta > 0.1:
                    delta_color = "red"
                elif self.entropy_delta > 0:
                    delta_color = "yellow"
                else:
                    delta_color = "green"
                
                lines.append(f"Current: {Path(self.current_file).name} (ΔH: [{delta_color}]{delta_str}[/{delta_color}])")
        else:
            lines.append(f"[bold]MYCO Status[/bold]")
            lines.append("MYCO Vision: Not available")
        
        # Iteration tracking
        lines.append(f"Iteration: {self.iteration}/{self.max_iterations}")
        
        # Create panel
        text = Text("\n".join(lines))
        panel = Panel(
            text,
            title="[bold cyan]MYCO Vision[/bold cyan]",
            border_style="cyan",
        )
        
        return panel
    
    def show_budget_warning(self, file_path: str, message: str, curr_h: float, prop_h: float):
        """Show entropy budget warning.
        
        Args:
            file_path: Path to file
            message: Warning message
            curr_h: Current entropy
            prop_h: Proposed entropy
        """
        delta = prop_h - curr_h
        delta_str = f"{delta:+.2f}"
        
        if delta > 0.15:
            color = "red"
            icon = "✗"
        elif delta > 0:
            color = "yellow"
            icon = "⚠"
        else:
            color = "green"
            icon = "✓"
        
        lines = [
            f"[bold]{icon} Entropy Budget Alert[/bold]",
            f"",
            f"File: {Path(file_path).name}",
            f"Current H: {curr_h:.2f} → Proposed H: {prop_h:.2f}",
            f"Delta: [{color}]{delta_str}[/{color}]",
            f"",
            f"[dim]{message}[/dim]",
        ]
        
        panel = Panel(
            "\n".join(lines),
            title="[bold yellow]MYCO Gate[/bold yellow]",
            border_style=color,
        )
        
        self.console.print(panel)
    
    def show_intervention_suggestion(self, file_path: str, regime: str, suggestions: list):
        """Show MYCO intervention suggestion.
        
        Args:
            file_path: Path to file needing intervention
            regime: Current entropy regime
            suggestions: List of suggested actions
        """
        regime_colors = {
            "crystallized": "blue",
            "dissipative": "green",
            "diffuse": "yellow",
        }
        color = regime_colors.get(regime, "white")
        
        lines = [
            f"[bold {color}]⚠ {regime.title()} Regime Detected[/bold {color}]",
            f"",
            f"File: {Path(file_path).name}",
            f"",
            f"[bold]Recommended Actions:[/bold]",
        ]
        
        for i, suggestion in enumerate(suggestions[:3], 1):
            lines.append(f"  {i}. {suggestion}")
        
        panel = Panel(
            "\n".join(lines),
            title="[bold yellow]MYCO Intervention[/bold yellow]",
            border_style=color,
        )
        
        self.console.print(panel)


# Global instance for easy access
_status_bar: Optional[EntropyStatusBar] = None


def get_status_bar(project_root: Path) -> EntropyStatusBar:
    """Get or create global status bar instance.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        EntropyStatusBar instance
    """
    global _status_bar
    if _status_bar is None or _status_bar.project_root != project_root:
        _status_bar = EntropyStatusBar(project_root)
    return _status_bar
