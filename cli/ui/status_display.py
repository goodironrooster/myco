"""MYCO Terminal UI Components using Rich."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text


class TaskStatus(Enum):
    """Status indicators for tasks."""

    PENDING = "[]"
    RUNNING = ">"
    VERIFIED = "OK"
    FAILED = "X"
    ASSUMED = "?"
    APPROVAL = "?"


@dataclass
class TaskStep:
    """A single step in a task."""

    name: str
    status: TaskStatus = TaskStatus.PENDING
    details: str = ""
    verified: bool = False


class StatusDisplay:
    """Display task status in terminal with Rich."""

    def __init__(self):
        self.console = Console()
        self.steps: list[TaskStep] = []
        self.task_name = "MYCO Task"
        self._live: Optional[Live] = None

    def start_task(self, name: str):
        """Start a new task."""
        self.task_name = name
        self.steps.clear()
        self.console.print(f"\n[bold blue]{name}[/bold blue]")
        self.console.print("[dim]Starting...[/dim]\n")

    def add_step(self, name: str, details: str = ""):
        """Add a new step to the task."""
        step = TaskStep(name=name, status=TaskStatus.RUNNING, details=details)
        self.steps.append(step)
        self._render_step(step, len(self.steps) - 1)

    def complete_step(self, index: int, verified: bool = True, details: str = ""):
        """Mark a step as complete with verification status."""
        if 0 <= index < len(self.steps):
            step = self.steps[index]
            step.status = TaskStatus.VERIFIED if verified else TaskStatus.ASSUMED
            step.verified = verified
            if details:
                step.details = details
            self._render_step(step, index)

    def fail_step(self, index: int, error: str = ""):
        """Mark a step as failed."""
        if 0 <= index < len(self.steps):
            step = self.steps[index]
            step.status = TaskStatus.FAILED
            step.details = error
            self._render_step(step, index)

    def _render_step(self, step: TaskStep, index: int):
        """Render a single step."""
        status_icons = {
            TaskStatus.PENDING: "[dim]--[/dim]",
            TaskStatus.RUNNING: "[blue]>>[/blue]",
            TaskStatus.VERIFIED: "[green]OK[/green]",
            TaskStatus.FAILED: "[red]X[/red]",
            TaskStatus.ASSUMED: "[yellow]?[/yellow]",
            TaskStatus.APPROVAL: "[magenta]?[/magenta]",
        }

        icon = status_icons[step.status]
        name = f"[bold]{step.name}[/bold]" if step.status == TaskStatus.RUNNING else step.name

        if step.details:
            details = f"[dim] - {step.details}[/dim]"
        else:
            details = ""

        # Move cursor up and clear line if not first step
        if index > 0:
            self.console.print(f"  {icon} {name}{details}")
        else:
            self.console.print(f"  {icon} {name}{details}")

    def finish_task(self, success: bool = True):
        """Finish the task with summary."""
        verified_count = sum(1 for s in self.steps if s.verified)
        failed_count = sum(1 for s in self.steps if s.status == TaskStatus.FAILED)
        assumed_count = sum(1 for s in self.steps if s.status == TaskStatus.ASSUMED)

        self.console.print()

        if success and failed_count == 0:
            self.console.print(f"[bold green]OK Task completed successfully[/bold green]")
        else:
            self.console.print(f"[bold yellow]? Task completed with issues[/bold yellow]")

        self.console.print(f"  [green]OK Verified:[/green] {verified_count}/{len(self.steps)}")
        if assumed_count > 0:
            self.console.print(
                f"  [yellow]? Assumed:[/yellow] {assumed_count}/{len(self.steps)} [dim](needs human check)[/dim]"
            )
        if failed_count > 0:
            self.console.print(f"  [red]X Failed:[/red] {failed_count}/{len(self.steps)}")

        self.console.print()


class VerificationPanel:
    """Display verification results in a panel."""

    def __init__(self):
        self.console = Console()

    def show_verification(self, actions: list[dict]):
        """Show verification report."""
        verified = [a for a in actions if a.get("verified", False)]
        failed = [a for a in actions if not a.get("success", True)]

        panel_text = Text()
        panel_text.append(f"Actions: {len(actions)}\n", style="bold")
        panel_text.append(f"OK Verified: {len(verified)}\n", style="green")

        if failed:
            panel_text.append(f"X Failed: {len(failed)}\n", style="red")
            panel_text.append("\nFailed actions (needs attention):\n", style="bold yellow")
            for action in failed:
                panel_text.append(
                    f"  • {action.get('tool', 'unknown')}: {action.get('args', {})}\n",
                    style="yellow",
                )

        panel = Panel(
            panel_text, title="[bold]MYCO Verification Report[/bold]", border_style="blue"
        )
        self.console.print(panel)


class ApprovalPrompt:
    """Prompt user for approval."""

    def __init__(self):
        self.console = Console()

    def request_approval(self, action: str, details: dict) -> str:
        """Request user approval for an action."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Action:[/bold] {action}\n"
                f"[bold]Details:[/bold] {details}\n\n"
                f"[yellow]This action requires your approval.[/yellow]",
                title="[magenta bold]? Approval Required[/magenta bold]",
                border_style="magenta",
            )
        )

        response = self.console.input(
            "[bold]Approve?[/bold] [green]y[/green]/[red]n[/red]/[blue]e[/blue]dit: "
        )
        return response.lower().strip()


# Convenience functions for simple usage
def show_status(message: str, status: TaskStatus = TaskStatus.RUNNING):
    """Show a simple status message."""
    console = Console()
    icons = {
        TaskStatus.PENDING: "[dim]--[/dim]",
        TaskStatus.RUNNING: "[blue]>>[/blue]",
        TaskStatus.VERIFIED: "[green]OK[/green]",
        TaskStatus.FAILED: "[red]X[/red]",
        TaskStatus.ASSUMED: "[yellow]?[/yellow]",
        TaskStatus.APPROVAL: "[magenta]?[/magenta]",
    }
    icon = icons.get(status, "[]")
    console.print(f"{icon} {message}")


def show_error(message: str):
    """Show an error message."""
    console = Console()
    console.print(f"[bold red]X {message}[/bold red]")


def show_success(message: str):
    """Show a success message."""
    console = Console()
    console.print(f"[bold green]OK {message}[/bold green]")
