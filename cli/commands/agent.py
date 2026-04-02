"""Agent CLI commands."""

import click
from pathlib import Path

from ..agent.core import Agent
from ..ui import show_status
from ..utils.config import Config
from ..utils.logging import LogConfig


@click.group()
@click.pass_context
def agent(ctx):
    """Autonomous AI agent for file editing and automation."""
    pass


@agent.command()
@click.argument("task")
@click.option(
    "--model",
    "-m",
    "model_name",
    default=None,
    help="Model name (default: from server)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show verbose progress",
)
@click.option(
    "--no-ui",
    is_flag=True,
    help="Disable enhanced UI (use plain text output)",
)
@click.option(
    "--no-approval",
    is_flag=True,
    help="Disable approval prompts (run all commands without asking)",
)
@click.option(
    "--max-iterations",
    "-i",
    type=int,
    default=30,  # Increased for complex tasks with 128K context
    help="Maximum tool use iterations",
)
@click.pass_context
def run(ctx, task, model_name, verbose, no_ui, no_approval, max_iterations):
    """Run an autonomous agent to complete a task.

    The agent can:
    - Read and write files
    - Edit existing code
    - Run shell commands
    - Search for text

    \b
    Examples:
      gguf agent run "Create a Python script that lists all files in current directory"
      gguf agent run "Add docstrings to all functions in main.py"
      gguf agent run "Find and fix all TODO comments in cli/ folder"
    """
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    # Get server configuration
    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)
    base_url = f"http://{host}:{port}"

    # Get model name
    if not model_name:
        # Try to get from server
        try:
            import requests

            response = requests.get(f"{base_url}/v1/models", timeout=5)
            models = response.json().get("data", [])
            if models:
                model_name = models[0].get("id")
        except Exception:
            model_name = "Qwen3.5-9B-Q4_0"

    # Show configuration
    show_status(f"Model: {model_name}")
    show_status(f"Server: {base_url}")

    use_ui = not no_ui
    require_approval = not no_approval

    if use_ui:
        click.echo()
        if require_approval:
            click.echo("[bold blue]MYCO Agent with Verification + Approval Gates[/bold blue]")
        else:
            click.echo("[bold yellow]MYCO Agent (No Approval Prompts)[/bold yellow]")
        click.echo("=" * 50)
    else:
        click.echo(f"\nUsing model: {model_name}")
        click.echo(f"Max iterations: {max_iterations}")
        if require_approval:
            click.echo("Approval prompts: enabled")
        else:
            click.echo("Approval prompts: disabled")

    click.echo(f"\nAI Task: {task}\n")

    # Create and run agent with MYCO vision integration
    agent_instance = Agent(
        base_url=base_url,
        model=model_name,
        max_iterations=max_iterations,
        require_approval=require_approval,
        project_root=str(Path.cwd()),  # MYCO: Pass project root for world model
    )

    try:
        result = agent_instance.run(task, verbose=verbose, use_ui=use_ui)

        click.echo("\n" + "=" * 50)
        if "OK Verified" in result or "verified" in result.lower():
            click.echo("[bold green]OK Task Complete (Verified)[/bold green]")
        else:
            click.echo("[bold yellow]? Task Complete (Review Recommended)[/bold yellow]")
        click.echo("=" * 50)
        click.echo(f"\n{result}\n")

    except KeyboardInterrupt:
        click.echo("\n\n? Agent interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        click.echo(click.style("X ", fg="red") + f"Agent failed: {e}")
        raise SystemExit(1)


@agent.command()
@click.pass_context
def tools(ctx):
    """List available agent tools."""
    click.echo("\n-- Available Agent Tools\n")
    click.echo("=" * 60)

    tools_info = [
        ("read_file", "Read file contents", "path, [lines]"),
        ("write_file", "Write content to file", "path, content"),
        ("edit_file", "Edit file (replace text)", "path, old_text, new_text"),
        ("list_files", "List files in directory", "path, [pattern]"),
        ("run_command", "Run shell command", "command, [timeout]"),
        ("search_text", "Search text in file", "path, query, [max_results]"),
    ]

    for name, desc, params in tools_info:
        click.echo(f"\n{click.style(name, bold=True, fg='cyan')}")
        click.echo(f"  {desc}")
        click.echo(f"  Parameters: {params}")

    click.echo("\n" + "=" * 60)
    click.echo('\nUsage: gguf agent run "your task"')
    click.echo('Example: gguf agent run "Create a hello.py file"\n')


@agent.command()
@click.option("--limit", "-n", type=int, default=20, help="Number of entries to show")
@click.pass_context
def history(ctx, limit):
    """Show approval history."""
    from ..agent.approval import ApprovalManager

    manager = ApprovalManager()
    history = manager.get_approval_history(limit=limit)

    if not history:
        click.echo("No approval history found.")
        return

    click.echo("\n" + "=" * 60)
    click.echo("Approval History")
    click.echo("=" * 60)

    for entry in history:
        timestamp = entry.get("timestamp", "Unknown")[:19]  # Remove microseconds
        command = entry.get("command", "Unknown")
        approved = entry.get("approved", False)
        remembered = entry.get("remembered", False)

        status = "[green]✓ Approved[/green]" if approved else "[red]✗ Denied[/red]"
        if remembered:
            status += " [dim](remembered)[/dim]"

        click.echo(f"\n{timestamp}")
        click.echo(f"  Command: {command}")
        click.echo(f"  Status: {status}")

    click.echo("\n" + "=" * 60)
    click.echo()


@agent.command()
@click.confirmation_option(prompt="Are you sure you want to clear approval history?")
@click.pass_context
def clear_history(ctx):
    """Clear approval history."""
    from ..agent.approval import ApprovalManager

    manager = ApprovalManager()
    manager.clear_history()

    click.echo(click.style("✓ ", fg="green") + "Approval history cleared.")


# Alias for interactive mode - can also use: gguf interactive
from ..commands.interactive import interactive as agent_interactive

agent.add_command(agent_interactive, name="interactive")
