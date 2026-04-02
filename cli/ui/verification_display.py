"""Verification dashboard UI components."""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


class VerificationDisplay:
    """Display verification results."""

    STATUS_ICONS = {
        "pass": "OK",
        "fail": "X",
        "warn": "?",
        "skip": "--",
    }

    STATUS_COLORS = {
        "pass": "green",
        "fail": "red",
        "warn": "yellow",
        "skip": "dim",
    }

    def __init__(self):
        self.console = Console()

    def show_dashboard(self, summary: dict):
        """Show verification dashboard.

        Args:
            summary: Verification summary from VerificationDashboard
        """
        self.console.print()

        # Header
        project_type = summary.get("project_type", "unknown")
        health_score = summary.get("health_score", 0)

        title = f"[bold]Project Health: {project_type.title()}[/bold]"
        if health_score >= 80:
            subtitle = f"[green]OK Healthy ({health_score:.0f}%)[/green]"
        elif health_score >= 50:
            subtitle = f"[yellow]? Needs Attention ({health_score:.0f}%)[/yellow]"
        else:
            subtitle = f"[red]X Needs Help ({health_score:.0f}%)[/red]"

        # Results table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="bold")
        table.add_column("Status")
        table.add_column("Details")

        for result in summary.get("results", []):
            status = result["status"]
            icon = self.STATUS_ICONS.get(status, "?")
            color = self.STATUS_COLORS.get(status, "white")

            table.add_row(
                result["name"],
                f"[{color}]{icon} {status.upper()}[/{color}]",
                result["message"][:50] + "..."
                if len(result["message"]) > 50
                else result["message"],
            )

        # Summary stats
        stats_text = f"""
[bold]Summary:[/bold]
  [green]OK Passed:[/green] {summary.get("passed", 0)}
  [red]X Failed:[/red] {summary.get("failed", 0)}
  [yellow]? Warnings:[/yellow] {summary.get("warned", 0)}
  [dim]-- Skipped:[/dim] {summary.get("skipped", 0)}
"""

        content = f"{stats_text}\n{table}"

        panel = Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style="cyan",
        )

        self.console.print(panel)

    def show_check_result(self, result: dict):
        """Show individual check result.

        Args:
            result: Verification result dict
        """
        status = result.get("status", "unknown")
        icon = self.STATUS_ICONS.get(status, "?")
        color = self.STATUS_COLORS.get(status, "white")

        self.console.print(
            f"[{color}]{icon}[/{color}] {result.get('name', 'Check')}: {result.get('message', '')}"
        )

    def show_progress(self, message: str):
        """Show progress indicator.

        Args:
            message: Message to display
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        )
        task = progress.add_task(message, total=None)

        return progress, task

    def show_error(self, message: str):
        """Show error message.

        Args:
            message: Error message
        """
        panel = Panel(
            f"[red]{message}[/red]",
            title="[red bold]X Error[/red bold]",
            border_style="red",
        )
        self.console.print(panel)


def display_verification_summary(summary: dict):
    """Display verification summary (convenience function).

    Args:
        summary: Verification summary from VerificationDashboard
    """
    display = VerificationDisplay()
    display.show_dashboard(summary)
