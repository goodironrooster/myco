"""MYCO Trajectory Prediction - Entropy trend analysis and forecasting.

Shows how entropy is changing over time and predicts when intervention will be needed.
"""

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TrajectoryDisplay:
    """Display entropy trajectory and predictions."""

    def __init__(self, project_root: Path):
        """Initialize trajectory display.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.console = Console()

    def render(self) -> Panel:
        """Render trajectory prediction panel.

        Returns:
            Rich Panel with trajectory analysis
        """
        lines = []

        # Get substrate health trend from annotations
        try:
            from myco.stigma import load_annotations
            annotations = load_annotations(self.project_root)
        except Exception:
            annotations = {}

        if not annotations:
            return Panel(
                "[dim]No trajectory data yet[/dim]\n[dim]Modify files across sessions to build history[/dim]",
                title="[bold cyan]Trajectory Prediction[/bold cyan]",
                border_style="dim",
            )

        # Analyze drift across all annotated files
        files_with_drift = []
        for path, hist in annotations.items():
            h = hist.current.H
            drift = hist.current.drift
            age = hist.current.age
            files_with_drift.append({
                "path": path,
                "H": h,
                "drift": drift,
                "age": age,
                "history_len": len(hist.history),
            })

        # Sort by drift magnitude (most changing first)
        files_with_drift.sort(key=lambda f: abs(f["drift"]), reverse=True)

        # Calculate overall trend
        total_drift = sum(f["drift"] for f in files_with_drift)
        avg_drift = total_drift / len(files_with_drift) if files_with_drift else 0
        avg_H = sum(f["H"] for f in files_with_drift) / len(files_with_drift) if files_with_drift else 0

        # Determine trend direction
        if avg_drift > 0.05:
            trend = "↗️  Increasing"
            trend_color = "yellow"
            urgency = "Moderate — schedule refactoring soon"
        elif avg_drift > 0.01:
            trend = "↗️  Slightly increasing"
            trend_color = "green"
            urgency = "Low — monitor closely"
        elif avg_drift < -0.05:
            trend = "↘️  Decreasing"
            trend_color = "green"
            urgency = "Good — code is improving"
        else:
            trend = "→  Stable"
            trend_color = "green"
            urgency = "Good — entropy is stable"

        # Predict sessions until intervention needed
        files_at_risk = [f for f in files_with_drift if f["H"] > 0.6]
        if files_at_risk and avg_drift > 0:
            max_H = max(f["H"] for f in files_at_risk)
            sessions_to_diffuse = max(0, int((0.75 - max_H) / avg_drift)) if avg_drift > 0 else 999
            if sessions_to_diffuse < 5:
                urgency = f"⚠️  {sessions_to_diffuse} session(s) until diffuse regime"
                trend_color = "red"
            elif sessions_to_diffuse < 10:
                urgency = f"~{sessions_to_diffuse} sessions until intervention needed"
                trend_color = "yellow"

        lines.append(f"[bold]Overall Trend:[/bold] [{trend_color}]{trend}[/{trend_color}]")
        lines.append(f"[bold]Avg Entropy:[/bold] {avg_H:.2f}")
        lines.append(f"[bold]Avg Drift/Session:[/bold] {avg_drift:+.3f}")
        lines.append(f"[bold]Files Tracked:[/bold] {len(files_with_drift)}")
        lines.append("")
        lines.append(f"[bold]Prediction:[/bold] {urgency}")

        # Show top files by drift
        if files_with_drift:
            lines.append("")
            lines.append("[bold]Files by Change Rate:[/bold]")

            table = Table(show_header=True, header_style="bold cyan", show_lines=False)
            table.add_column("File", style="white", width=35)
            table.add_column("H", justify="right", style="yellow", width=6)
            table.add_column("Drift", justify="right", width=8)
            table.add_column("Age", justify="center", style="dim", width=5)
            table.add_column("Status", style="dim", width=20)

            for f in files_with_drift[:10]:
                path = f["path"]
                if len(path) > 33:
                    path = "..." + path[-30:]

                drift_str = f"{f['drift']:+.3f}"
                if f["drift"] > 0.05:
                    drift_style = "red"
                elif f["drift"] > 0:
                    drift_style = "yellow"
                elif f["drift"] < -0.05:
                    drift_style = "green"
                else:
                    drift_style = "dim"

                if f["H"] > 0.75:
                    status = "[red]Diffuse — refactor[/red]"
                elif f["H"] > 0.6:
                    status = "[yellow]Approaching risk[/yellow]"
                elif f["H"] < 0.3:
                    status = "[blue]Rigid — avoid changes[/blue]"
                else:
                    status = "[green]Healthy[/green]"

                table.add_row(
                    path,
                    f"{f['H']:.2f}",
                    f"[{drift_style}]{drift_str}[/{drift_style}]",
                    str(f["age"]),
                    status,
                )

            # Render table to string
            from io import StringIO
            from rich.console import Console as RichConsole
            buffer = StringIO()
            temp_console = RichConsole(file=buffer, force_terminal=True, width=100)
            temp_console.print(table)
            table_text = buffer.getvalue()

            lines.append(table_text)

        content = "\n".join(lines)

        border = "red" if "⚠" in urgency else ("yellow" if "soon" in urgency.lower() else "green")
        panel = Panel(
            content,
            title="[bold cyan]Trajectory Prediction[/bold cyan]",
            border_style=border,
        )

        return panel


# Global instance
_trajectory: Optional[TrajectoryDisplay] = None


def get_trajectory_display(project_root: Path) -> TrajectoryDisplay:
    """Get or create global trajectory display instance.

    Args:
        project_root: Root directory of the project

    Returns:
        TrajectoryDisplay instance
    """
    global _trajectory
    if _trajectory is None or _trajectory.project_root != project_root:
        _trajectory = TrajectoryDisplay(project_root)
    return _trajectory
