"""Approval prompt UI components."""

import threading
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class _InputGetter:
    """Helper class to get input in a separate thread."""

    def __init__(self, prompt: str, console: Console):
        self.prompt = prompt
        self.console = console
        self.result = None
        self.ready = threading.Event()

    def get_input(self):
        """Get input (runs in separate thread)."""
        try:
            self.result = self.console.input(self.prompt)
        except Exception:
            self.result = ""
        finally:
            self.ready.set()


class ApprovalPrompt:
    """Prompt user for approval of risky operations."""

    def __init__(self):
        self.console = Console()

    def request_approval(
        self,
        command: str,
        rule_description: str = "",
        estimated_size_mb: Optional[float] = None,
        shows_modifications: list[str] = None,
        impact_info: Optional[dict] = None,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Request user approval for an action.

        Args:
            command: Command to execute
            rule_description: Description of what this command does
            estimated_size_mb: Estimated download size in MB
            shows_modifications: List of files/dirs that will be modified
            impact_info: Enhanced impact information from CommandImpactEstimator
            timeout_seconds: Timeout in seconds (default: no timeout)

        Returns:
            User response: 'y', 'n', 'e', 'r', 'a', or 't' for timeout
        """
        self.console.print()

        # Build panel content
        content = Text()
        content.append(f"Command: ", style="bold")
        content.append(f"{command}\n\n")

        if rule_description:
            content.append(f"Action: ", style="bold")
            content.append(f"{rule_description}\n\n")

        # Show enhanced impact info if available
        if impact_info:
            # Package info
            pkg_info = impact_info.get("package_info")
            if pkg_info and pkg_info.get("package_count", 0) > 0:
                content.append(f"Impact:\n", style="bold")
                content.append(f"  📦 {pkg_info['package_count']} packages", style="cyan")
                if pkg_info.get("estimated_size_mb", 0) > 0:
                    content.append(f"  💾 ~{pkg_info['estimated_size_mb']:.1f} MB\n", style="cyan")
                if pkg_info.get("packages"):
                    first_pkgs = ", ".join(pkg_info["packages"][:3])
                    content.append(f"  First: {first_pkgs}\n", style="dim")
                if pkg_info.get("note"):
                    content.append(f"  Note: {pkg_info['note']}\n", style="dim")
                content.append("\n")

            # Environment info
            env_info = impact_info.get("environment")
            if env_info:
                content.append(f"Scope: ", style="bold")
                scope = env_info.get("scope", "unknown")
                content.append(f"{scope}\n", style="yellow" if scope == "system-wide" else "green")

                if env_info.get("modifies"):
                    content.append(f"Modifies:\n", style="bold")
                    for mod in env_info["modifies"][:5]:
                        content.append(f"  • {mod}\n", style="dim")
                    content.append("\n")

                if env_info.get("warning"):
                    content.append(f"{env_info['warning']}\n\n", style="yellow bold")

            # Summary
            summary = impact_info.get("summary", [])
            if summary and not pkg_info:
                content.append(f"Impact:\n", style="bold")
                for item in summary[:5]:
                    content.append(f"  {item}\n", style="dim")
                content.append("\n")
        else:
            # Fallback to old format
            if estimated_size_mb:
                content.append(f"Estimated: ", style="bold")
                content.append(f"~{estimated_size_mb:.1f} MB\n\n")

            if shows_modifications:
                content.append(f"Modifies:\n", style="bold")
                for mod in shows_modifications:
                    content.append(f"  • {mod}\n")
                content.append("\n")

        timeout_msg = ""
        if timeout_seconds:
            timeout_msg = f" (times out in {timeout_seconds}s)"

        content.append(f"This action requires your approval.{timeout_msg}", style="yellow bold")

        # Create panel
        panel = Panel(
            content,
            title="[magenta bold]? Approval Required[/magenta bold]",
            border_style="magenta",
        )

        self.console.print(panel)

        # Get response with optional timeout
        self.console.print()
        prompt = (
            "[bold]Approve?[/bold] "
            "[green]y[/green]es / "
            "[red]n[/red]o / "
            "[blue]e[/blue]dit / "
            "[cyan]r[/cyan]emember / "
            "[magenta]a[/magenta]ll for this session: "
        )

        if timeout_seconds:
            response = self._get_input_with_timeout(prompt, timeout_seconds)
            if response is None:
                self.console.print(
                    f"\n[yellow]T Approval timed out after {timeout_seconds}s - auto-denying[/yellow]"
                )
                return "t"  # timeout
        else:
            response = self.console.input(prompt)

        return response.lower().strip() if response else ""

    def _get_input_with_timeout(self, prompt: str, timeout_seconds: int) -> Optional[str]:
        """Get input with timeout.

        Args:
            prompt: Prompt to display
            timeout_seconds: Timeout in seconds

        Returns:
            User input or None if timed out
        """
        input_getter = _InputGetter(prompt, self.console)
        thread = threading.Thread(target=input_getter.get_input, daemon=True)
        thread.start()

        # Wait for input or timeout
        if input_getter.ready.wait(timeout=timeout_seconds):
            return input_getter.result
        else:
            return None

    def request_edit(self, original_command: str) -> str:
        """Allow user to edit the command.

        Args:
            original_command: Original command to edit

        Returns:
            Edited command or original if unchanged
        """
        self.console.print()
        self.console.print(f"[dim]Original:[/dim] {original_command}")

        edited = self.console.input("[bold]Edit command:[/bold] ")

        return edited.strip() if edited.strip() else original_command

    def show_blocked(self, command: str, reason: str):
        """Show that a command is blocked.

        Args:
            command: Command that was blocked
            reason: Reason for blocking
        """
        self.console.print()

        content = Text()
        content.append(f"Command: {command}\n\n", style="bold")
        content.append(f"Status: ", style="bold red")
        content.append("BLOCKED\n\n", style="red")
        content.append(f"Reason: {reason}\n\n", style="yellow")
        content.append("This command is blocked for safety reasons.", style="red bold")

        panel = Panel(
            content,
            title="[red bold]X Command Blocked[/red bold]",
            border_style="red",
        )

        self.console.print(panel)

    def show_auto_approved(self, command: str, reason: str = ""):
        """Show that a command was auto-approved.

        Args:
            command: Command that was approved
            reason: Reason for auto-approval
        """
        content = Text()
        content.append(f"Command: {command}\n", style="dim")
        if reason:
            content.append(f"Reason: {reason}", style="green")

        self.console.print(f"[green]OK[/green] Auto-approved: {command}")
        if reason:
            self.console.print(f"[dim]  {reason}[/dim]")


# Convenience function for simple usage
def prompt_approval(
    command: str,
    description: str = "",
    size_mb: Optional[float] = None,
    modifications: list[str] = None,
) -> tuple[bool, str]:
    """Simple approval prompt.

    Args:
        command: Command to approve
        description: What this command does
        size_mb: Estimated size in MB
        modifications: Files/dirs that will be modified

    Returns:
        Tuple of (approved: bool, final_command: str)
    """
    prompter = ApprovalPrompt()

    while True:
        response = prompter.request_approval(
            command=command,
            rule_description=description,
            estimated_size_mb=size_mb,
            shows_modifications=modifications,
        )

        if response in ("y", "yes", ""):
            return True, command
        elif response in ("n", "no"):
            return False, command
        elif response == "e":
            command = prompter.request_edit(command)
        elif response == "r":
            # Remember choice - for now just approve
            return True, command
        elif response == "a":
            # Approve all for session - for now just approve
            return True, command
        else:
            prompter.console.print("[yellow]Invalid response. Please try again.[/yellow]")
