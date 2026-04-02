"""Server control commands."""

from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.model_manager import ModelManager
from ..core.server_manager import ServerManager
from ..utils.config import Config
from ..utils.logging import LogConfig


@click.group()
@click.pass_context
def server(ctx):
    """Control the llama.cpp server."""
    pass


@server.command()
@click.option(
    "--model",
    "-m",
    "model_path",
    type=click.Path(exists=True),
    help="Path to GGUF model file",
)
@click.option(
    "--host",
    "-h",
    default=None,
    help="Server host (default: from config)",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Server port (default: from config)",
)
@click.option(
    "--context",
    "-c",
    type=int,
    default=None,
    help="Context length (default: from config)",
)
@click.option(
    "--threads",
    "-t",
    type=int,
    default=None,
    help="Number of threads (default: auto)",
)
@click.option(
    "--cuda/--no-cuda",
    default=True,
    help="Use CUDA GPU acceleration (default: auto-detect)",
)
@click.option(
    "--gpu-layer",
    "-g",
    type=int,
    default=None,
    help="Number of layers to offload to GPU (default: all)",
)
@click.option(
    "--batch-size",
    "-bs",
    type=int,
    default=256,
    help="Batch size for CUDA (default: 256, optimal for 40+ tok/s)",
)
@click.option(
    "--flash-attn",
    is_flag=True,
    default=True,
    help="Enable flash attention (default: on for CUDA)",
)
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    help="Run in foreground (block terminal)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show verbose server output (disable --log-disable)",
)
@click.pass_context
def start(ctx, model_path, host, port, context, threads, cuda, gpu_layer, batch_size, flash_attn, foreground, verbose):
    """Start the llama.cpp server."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    # Get configuration values
    host = host or config.get("server", "host", default="127.0.0.1")
    port = port or config.get("server", "port", default=1234)
    context = context or config.get("server", "context_length", default=8192)
    threads = threads or config.get("server", "threads")

    # Find model if not specified
    if model_path is None:
        models_dir = Path.cwd()
        manager = ModelManager(models_dir)
        models = manager.list_models()
        if not models:
            click.echo(click.style("✗ ", fg="red") + "No GGUF models found in current directory.")
            click.echo("Use --model to specify a model file.")
            raise SystemExit(1)
        model_path = models[0].path
        click.echo(f"Using model: {model_path.name}")
    else:
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path

    # Find llama-server.exe
    server_exe = _find_server_exe()
    if server_exe is None:
        click.echo(
            click.style("✗ ", fg="red") + "llama-server.exe not found. Please install llama.cpp."
        )
        click.echo("Download from: https://github.com/ggml-org/llama.cpp/releases")
        raise SystemExit(1)

    # Start server
    server_manager = ServerManager(server_exe, host, port)
    console = Console()

    # Show loading indicator while model loads
    if not foreground:
        click.echo(f"Loading model: {model_path.name}...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading model into GPU...", total=None)

            # GPU layers: None = auto-detect, 0 = CPU only, >0 = GPU layers
            gpu_layers = 99 if cuda else 0
            if gpu_layer is not None:
                gpu_layers = gpu_layer

            status = server_manager.start(
                model_path=model_path,
                context_length=context,
                threads=threads,
                background=True,
                quiet=not verbose,
                gpu_layers=gpu_layers,
                batch_size=batch_size,
                flash_attn=flash_attn,
            )
    else:
        gpu_layers = 99 if cuda else 0
        if gpu_layer is not None:
            gpu_layers = gpu_layer

        status = server_manager.start(
            model_path=model_path,
            context_length=context,
            threads=threads,
            background=not foreground,
            quiet=not verbose,
            gpu_layers=gpu_layers,
            batch_size=batch_size,
            flash_attn=flash_attn,
        )

    if status.running:
        click.echo(click.style("✓ ", fg="green") + f"Server started on {status.url}")
        click.echo(f"  Model: {status.model}")
        click.echo(f"  Context: {context:,} tokens")
        click.echo(f"  GPU: {'enabled' if cuda else 'CPU only'}")
        click.echo(f"  Batch Size: {batch_size}")
        click.echo(f"  Flash Attention: {'on' if flash_attn else 'off'}")
        click.echo(f"  API:   {status.api_url}")
        click.echo(f'\nReady to use! Run: python -m cli.main myco "your task"')
        if foreground and status.pid:
            # Wait for process
            try:
                import subprocess

                subprocess.Popen(["taskkill", "/F", "/PID", str(status.pid)]).wait()
            except Exception:
                pass
    else:
        click.echo(click.style("✗ ", fg="red") + f"Failed to start: {status.message}")
        raise SystemExit(1)


@server.command()
@click.pass_context
def stop(ctx):
    """Stop the running server."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    # Get host/port from config
    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)

    # Find server executable (needed for ServerManager init)
    server_exe = _find_server_exe() or Path("dummy")

    server_manager = ServerManager(server_exe, host, port)
    status = server_manager.stop()

    if not status.running:
        click.echo(click.style("✓ ", fg="green") + status.message)
    else:
        click.echo(click.style("⚠ ", fg="yellow") + status.message)


@server.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def status(ctx, as_json):
    """Check server status."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    # Get host/port from config
    host = config.get("server", "host", default="127.0.0.1")
    port = config.get("server", "port", default=1234)

    # Find server executable (needed for ServerManager init)
    server_exe = _find_server_exe() or Path("dummy")

    server_manager = ServerManager(server_exe, host, port)
    server_status = server_manager.status()

    if as_json:
        import json

        click.echo(
            json.dumps(
                {
                    "running": server_status.running,
                    "pid": server_status.pid,
                    "url": server_status.url,
                    "message": server_status.message,
                },
                indent=2,
            )
        )
    else:
        if server_status.running:
            click.echo(click.style("● ", fg="green") + "Server is running")
            click.echo(f"  URL: {server_status.url}")
            click.echo(f"  PID: {server_status.pid}")
            click.echo(f"  Status: {server_status.message}")
        else:
            click.echo(click.style("○ ", fg="yellow") + "Server is not running")


def _find_server_exe() -> Path | None:
    """Find llama-server.exe in common locations."""
    candidates = [
        Path.cwd() / "llama-cpp" / "bin" / "llama-server.exe",
        Path.cwd() / "llama-cpp" / "llama-server.exe",
        Path.cwd() / "llama-server.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Search in PATH
    import shutil

    server_path = shutil.which("llama-server.exe")
    if server_path:
        return Path(server_path)

    return None
