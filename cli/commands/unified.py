"""MYCO Unified Command - One command to rule them all."""

import socket
import sys
import time

import click
import requests
from rich.console import Console
from rich.panel import Panel

from ..core.model_manager import ModelManager
from ..core.server_manager import ServerManager
from ..utils.config import Config
from ..utils.logging import LogConfig


console = Console()


def _find_server_exe():
    """Find llama-server in common locations."""
    from pathlib import Path
    import shutil

    candidates = [
        Path.cwd() / "llama-cpp" / "bin" / "llama-server.exe",
        Path.cwd() / "llama-cpp" / "llama-server.exe",
        Path.cwd() / "llama-server.exe",
        Path.cwd() / "bin" / "llama-server.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Search in PATH
    server_path = shutil.which("llama-server.exe")
    if server_path:
        return Path(server_path)

    # Check common installation paths
    import os

    program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
    for pf in [program_files, program_files.replace(" (x86)", "")]:
        for subdir in ["llama.cpp", "llama-bin", "llama-server"]:
            path = Path(pf) / subdir / "llama-server.exe"
            if path.exists():
                return path

    return None


def _check_server_running(host="127.0.0.1", port=1234):
    """Check if server is already running."""
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def _find_model():
    """Find a model to use."""
    models_dir = Path.cwd()
    manager = ModelManager(models_dir)
    models = manager.list_models()
    if models:
        return models[0].path
    return None


@click.command()
@click.option(
    "--task",
    "-t",
    "task_text",
    default=None,
    help="Task to complete (if not interactive mode)",
)
@click.option(
    "--model",
    "-m",
    "model_name",
    default=None,
    help="Model name",
)
@click.option(
    "--no-approval",
    is_flag=True,
    help="Skip approval prompts",
)
@click.option(
    "--cuda/--no-cuda",
    default=True,
    help="Use CUDA GPU acceleration (default: auto)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
@click.pass_context
def myco(ctx, task_text, model_name, no_approval, cuda, verbose):
    """MYCO - Your AI Coding Assistant.

    One command to start chatting or running tasks with your local AI.

    \b
    Usage:
      myco                    # Start interactive chat
      myco -t "task"          # Run a single task
      myco --server          # Just start server (no interactive)

    \b
    Examples:
      myco
      myco -t "Create a hello.py file"
      myco -t "Run the tests"
      myco --no-approval -t "Install numpy"
    """
    logger = LogConfig.get_logger("gguf")
    config = Config()

    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)

    # Check if server is already running
    server_running = _check_server_running(host, port)

    if not server_running:
        console.print("\n[cyan]Starting MYCO Server...[/cyan]\n")

        # Find server
        server_exe = _find_server_exe()
        if not server_exe:
            console.print("[red]llama-server.exe not found![/red]")
            console.print("Please:")
            console.print(
                "  1. Download llama.cpp from https://github.com/ggml-org/llama.cpp/releases"
            )
            console.print("  2. Place llama-server.exe in the MYCO directory or in PATH")
            console.print("  3. For CUDA support, use the CUDA build of llama.cpp")
            raise SystemExit(1)

        # Find model
        model_path = _find_model()
        if not model_path:
            console.print("[red]No GGUF model found![/red]")
            console.print("Place a .gguf model file in the current directory.")
            raise SystemExit(1)

        console.print(f"[dim]Server:[/dim] {server_exe.name}")
        console.print(f"[dim]Model:[/dim] {model_path.name}")
        console.print(f"[dim]CUDA:[/dim] {'enabled' if cuda else 'disabled'}")
        console.print()

        # Start server
        manager = ServerManager(server_exe, host, port)

        with console.status(
            f"[bold cyan]Loading model into {'GPU' if cuda else 'CPU'}...[/bold cyan]"
        ) as status:
            status.update(f"Loading {model_path.name}...")

            # Build command with CUDA if requested
            from pathlib import Path

            context = config.get("server", "context_length", default=8192)

            status_result = manager.start(
                model_path=model_path,
                context_length=context,
                background=True,
                quiet=not verbose,
            )

            if not status_result.running:
                console.print(f"[red]Failed to start server: {status_result.message}[/red]")
                raise SystemExit(1)

            # Wait for server to be ready
            for _ in range(30):
                if _check_server_running(host, port):
                    break
                time.sleep(1)

        console.print(f"[green]✓ Server ready at http://{host}:{port}[/green]\n")

    # If task provided, run it
    if task_text:
        from .agent import run as agent_run

        # Call agent run programmatically
        ctx.invoke(
            agent_run,
            task=task_text,
            model_name=model_name,
            verbose=verbose,
            no_ui=False,
            no_approval=no_approval,
            max_iterations=10,
        )
    else:
        # Start interactive mode
        from .interactive import interactive as agent_interactive

        ctx.invoke(agent_interactive, model_name=model_name, no_approval=no_approval)


# Import Path for later
from pathlib import Path
