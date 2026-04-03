"""MYCO Entropy Visualization - Visual representations of entropy state.

Features:
- Entropy gradient maps
- Module-level entropy bars
- Project-wide distribution
- Change tracking
"""

from pathlib import Path
from typing import Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.bar import Bar


class EntropyVisualizer:
    """Visualize entropy states and changes."""
    
    def __init__(self, project_root: Path):
        """Initialize entropy visualizer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.console = Console()
        self.last_health: Optional[Any] = None
        
        # Try to load MYCO vision
        self.has_myco = False
        try:
            from myco.entropy import calculate_substrate_health
            self.calculate_health = calculate_substrate_health
            self.has_myco = True
        except Exception:
            self.calculate_health = None
    
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
    
    def _get_regime_icon(self, regime: str) -> str:
        """Get icon for entropy regime."""
        icons = {
            "crystallized": "🔒",
            "dissipative": "⚗️",
            "diffuse": "⚠️",
        }
        return icons.get(regime, "•")
    
    def _create_entropy_bar(self, entropy: float, width: int = 20) -> Text:
        """Create visual entropy bar.
        
        Args:
            entropy: Entropy value (0.0-1.0)
            width: Width of bar in characters
            
        Returns:
            Rich Text with colored bar
        """
        filled = int(entropy * width)
        empty = width - filled
        
        # Determine color based on entropy
        if entropy < 0.3:
            color = "blue"
            char = "▓"
        elif entropy < 0.75:
            color = "green"
            char = "▓"
        else:
            color = "yellow"
            char = "█"
        
        # Build bar
        bar_text = Text()
        bar_text.append(char * filled, style=color)
        bar_text.append("░" * empty, style="dim")
        
        return bar_text
    
    def refresh(self):
        """Refresh substrate health data."""
        if self.has_myco:
            try:
                from myco.entropy import analyze_entropy
                self.last_health = analyze_entropy(self.project_root)
            except Exception:
                pass
    
    def render_gradient_map(self, path: Optional[str] = None, limit: int = 20) -> Panel:
        """Render entropy gradient map for a directory.

        Args:
            path: Subdirectory path (relative to project root)
            limit: Maximum files to show

        Returns:
            Rich Panel with gradient map
        """
        if not self.last_health:
            self.refresh()

        if not self.last_health:
            return Panel(
                "[dim]MYCO Vision not available[/dim]",
                title="Entropy Gradient Map",
                border_style="dim",
            )

        # Collect all files from entropy report
        report = self.last_health
        all_files = []

        # Each list contains module names strings
        for module in getattr(report, 'crystallized', []):
            all_files.append({"path": module.replace(".", "/") + ".py", "entropy": 0.2, "regime": "crystallized"})
        for module in getattr(report, 'dissipative', []):
            all_files.append({"path": module.replace(".", "/") + ".py", "entropy": 0.5, "regime": "dissipative"})
        for module in getattr(report, 'diffuse', []):
            all_files.append({"path": module.replace(".", "/") + ".py", "entropy": 0.8, "regime": "diffuse"})

        # Filter by path if provided
        if path:
            path_obj = Path(path)
            all_files = [f for f in all_files if f["path"].startswith(str(path_obj))]

        # Sort by entropy (highest first)
        all_files = sorted(all_files, key=lambda f: f["entropy"], reverse=True)[:limit]

        # Create table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Regime", width=4)
        table.add_column("File", style="white")
        table.add_column("Entropy", justify="right", style="yellow")
        table.add_column("Bar", width=22)
        table.add_column("Status", style="dim")

        for f in all_files:
            entropy = f["entropy"]
            rel_path = f["path"]
            regime = f["regime"]
            regime_color = self._get_regime_color(regime)
            regime_icon = self._get_regime_icon(regime)

            # Truncate long paths
            if len(rel_path) > 40:
                rel_path = "..." + rel_path[-37:]

            # Status based on regime
            if regime == "diffuse":
                status = "[yellow]Needs refactoring[/yellow]"
            elif regime == "crystallized":
                status = "[blue]Rigid - avoid changes[/blue]"
            else:
                status = "[green]Healthy[/green]"

            # Entropy bar
            bar = self._create_entropy_bar(entropy)

            table.add_row(
                f"[{regime_color}]{regime_icon}[/{regime_color}]",
                rel_path,
                f"{entropy:.2f}",
                bar,
                status,
            )

        # Summary
        if all_files:
            entropies = [f["entropy"] for f in all_files]
            avg = sum(entropies) / len(entropies)
            summary = f"[bold]Average Entropy:[/bold] {avg:.2f} | [bold]Files:[/bold] {len(all_files)}"
        else:
            summary = "[dim]No Python files tracked yet. Modify a file to start tracking.[/dim]"

        # Render table to string
        from io import StringIO
        from rich.console import Console as RichConsole
        buffer = StringIO()
        temp_console = RichConsole(file=buffer, force_terminal=True, width=100)
        temp_console.print(table)
        table_text = buffer.getvalue()

        panel = Panel(
            summary + "\n\n" + table_text,
            title="[bold cyan]Entropy Gradient Map[/bold cyan]",
            border_style="cyan",
        )

        return panel
    
    def render_distribution(self) -> Panel:
        """Render entropy distribution across project.
        
        Returns:
            Rich Panel with distribution chart
        """
        if not self.last_health:
            self.refresh()
        
        if not self.last_health:
            return Panel(
                "[dim]MYCO Vision not available[/dim]",
                title="Entropy Distribution",
                border_style="dim",
            )
        
        # Get counts
        if isinstance(self.last_health, dict):
            crystallized = self.last_health.get('crystallized_count', 0)
            dissipative = self.last_health.get('dissipative_count', 0)
            diffuse = self.last_health.get('diffuse_count', 0)
            total = len(self.last_health.get('files', []))
        else:
            crystallized = getattr(self.last_health, 'crystallized_count', 0)
            dissipative = getattr(self.last_health, 'dissipative_count', 0)
            diffuse = getattr(self.last_health, 'diffuse_count', 0)
            total = len(getattr(self.last_health, 'files', []))
        
        # Create distribution bars
        def make_bar(count: int, total: int, color: str, label: str) -> str:
            if total == 0:
                pct = 0
            else:
                pct = count / total * 100
            bar_len = int(pct / 5)  # Max 20 chars
            bar = "█" * bar_len + "░" * (20 - bar_len)
            return f"[{color}]{label:12} [{bar}] {count:3} ({pct:5.1f}%)[/{color}]"
        
        lines = [
            make_bar(crystallized, total, "blue", "Crystallized"),
            make_bar(dissipative, total, "green", "Dissipative"),
            make_bar(diffuse, total, "yellow", "Diffuse"),
            "",
            f"[bold]Total Files:[/bold] {total}",
        ]
        
        # Visual distribution
        if total > 0:
            lines.append("")
            lines.append("[bold]Distribution:[/bold]")
            
            # Simple ASCII chart
            c_pct = crystallized / total * 100 if total > 0 else 0
            d_pct = dissipative / total * 100 if total > 0 else 0
            df_pct = diffuse / total * 100 if total > 0 else 0
            
            chart_line = ""
            chart_line += f"[blue]{'█' * int(c_pct/5)}[/blue]"
            chart_line += f"[green]{'█' * int(d_pct/5)}[/green]"
            chart_line += f"[yellow]{'█' * int(df_pct/5)}[/yellow]"
            lines.append(f"  {chart_line}")
            lines.append("  [dim]0% ───────────────────────────────────────────────── 100%[/dim]")
        
        panel = Panel(
            "\n".join(lines),
            title="[bold cyan]Entropy Distribution[/bold cyan]",
            border_style="cyan",
        )
        
        return panel
    
    def render_file_entropy(self, file_path: str, content: Optional[str] = None) -> Panel:
        """Render entropy info for a specific file.
        
        Args:
            file_path: Path to the file
            content: Optional content to analyze
            
        Returns:
            Rich Panel with file entropy info
        """
        path = Path(file_path)
        
        if not path.exists():
            return Panel(
                f"[red]File not found: {file_path}[/red]",
                title="File Entropy",
                border_style="red",
            )
        
        # Try to get entropy from substrate health
        entropy = None
        regime = None
        
        if self.last_health:
            if isinstance(self.last_health, dict):
                files = self.last_health.get('files', [])
            else:
                files = getattr(self.last_health, 'files', [])
            
            for f in files:
                fpath = f.get('path', f) if isinstance(f, dict) else getattr(f, 'path', str(f))
                if str(fpath) == str(path):
                    entropy = f.get('entropy', 0) if isinstance(f, dict) else getattr(f, 'entropy', 0)
                    regime = self._get_regime(entropy)
                    break
        
        # Build content
        lines = [
            f"[bold]File:[/bold] {path.name}",
            f"[bold]Path:[/bold] {path.relative_to(self.project_root) if path.is_relative_to(self.project_root) else path}",
            "",
        ]
        
        if entropy is not None:
            regime_color = self._get_regime_color(regime)
            regime_icon = self._get_regime_icon(regime)
            bar = self._create_entropy_bar(entropy)
            
            lines.extend([
                f"[bold]Entropy:[/bold] {entropy:.2f}",
                f"[bold]Regime:[/bold] [{regime_color}]{regime_icon} {regime.title()}[/{regime_color}]",
                f"[bold]Bar:[/bold] {bar}",
                "",
            ])
            
            # Regime-specific advice
            if regime == "crystallized":
                lines.append("[blue]💡 This file is rigid. Avoid direct modifications.[/blue]")
                lines.append("[dim]Consider: decompose or interface_inversion[/dim]")
            elif regime == "diffuse":
                lines.append("[yellow]⚠️ This file needs refactoring.[/yellow]")
                lines.append("[dim]Consider: extract module or compression_collapse[/dim]")
            else:
                lines.append("[green]✓ This file is in healthy dissipative regime.[/green]")
                lines.append("[dim]Safe to make targeted changes[/dim]")
        else:
            if file_path.endswith('.py'):
                lines.append("[yellow]Entropy not calculated yet[/yellow]")
                lines.append("[dim]Run substrate_health to analyze[/dim]")
            else:
                lines.append("[dim]Non-Python file - entropy tracking not available[/dim]")
        
        panel = Panel(
            "\n".join(lines),
            title="[bold cyan]File Entropy[/bold cyan]",
            border_style="cyan",
        )
        
        return panel


# Global instance
_visualizer: Optional[EntropyVisualizer] = None


def get_entropy_visualizer(project_root: Path) -> EntropyVisualizer:
    """Get or create global entropy visualizer instance.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        EntropyVisualizer instance
    """
    global _visualizer
    if _visualizer is None or _visualizer.project_root != project_root:
        _visualizer = EntropyVisualizer(project_root)
    return _visualizer
