"""MYCO CLI - Single entry point.

Usage:
    myco                    # Start interactive agent (auto-starts server)
    myco "task"             # Run a single task then exit
    myco --help             # Show help
"""

import sys
import time
from pathlib import Path

import click
import requests
from rich.console import Console

console = Console()

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT = 200_000  # 200K context window
DEFAULT_PORT = 1234
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "Qwen3.5-9B-Q4_0.gguf"
GPU_LAYERS = 99  # All layers on CUDA GPU — no CPU fallback


def _check_server_running(host: str, port: int) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=1)
        return r.status_code == 200
    except Exception:
        return False


def _find_server_exe() -> Path | None:
    # MYCO project root (parent of cli/)
    project_root = Path(__file__).resolve().parent.parent
    # CUDA build first (llama-src compiled with LLAMA_CUDA=1)
    candidates = [
        project_root / "llama-src" / "build" / "bin" / "llama-server.exe",
        project_root / "llama-cpp" / "bin" / "llama-server.exe",
        project_root / "llama-cpp" / "llama-server.exe",
        project_root / "llama-server.exe",
        Path.cwd() / "llama-src" / "build" / "bin" / "llama-server.exe",
        Path.cwd() / "llama-cpp" / "bin" / "llama-server.exe",
        Path.cwd() / "llama-cpp" / "llama-server.exe",
        Path.cwd() / "llama-server.exe",
        Path.cwd() / "bin" / "llama-server.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    import shutil
    p = shutil.which("llama-server.exe")
    return Path(p) if p else None


def _find_model() -> Path | None:
    # 1. Hardcoded default (Qwen3.5-9B in project root)
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    # 2. Scan current directory
    for p in Path.cwd().glob("*.gguf"):
        return p
    # 3. Recursive scan
    for p in Path.cwd().rglob("*.gguf"):
        return p
    return None


def _start_server(model_path: Path, host: str, port: int, context: int, verbose: bool):
    """Start llama.cpp server with CUDA GPU — no CPU fallback."""
    from cli.core.server_manager import ServerManager

    server_exe = _find_server_exe()
    if not server_exe:
        console.print("[red]llama-server.exe not found![/red]")
        console.print("Please install llama.cpp (CUDA build) and place llama-server.exe in PATH or current directory.")
        raise SystemExit(1)

    console.print(f"[dim]Server:[/dim] {server_exe.name}")
    console.print(f"[dim]Model:[/dim] {model_path.name}")
    console.print(f"[dim]GPU:[/dim] CUDA  [dim]GPU layers:[/dim] {GPU_LAYERS}  [dim]Context:[/dim] {context:,}")
    console.print(f"[dim]Flash Attention:[/dim] on  [dim]Batch size:[/dim] 256")
    console.print()

    manager = ServerManager(server_exe, host, port)

    with console.status(f"[bold cyan]Loading {model_path.name} into GPU ({GPU_LAYERS} layers)...[/bold cyan]"):
        status = manager.start(
            model_path=model_path,
            context_length=context,
            background=True,
            quiet=not verbose,
            gpu_layers=GPU_LAYERS,
            batch_size=256,
            flash_attn=True,
        )

    if not status.running:
        console.print(f"[red]Failed to start server: {status.message}[/red]")
        console.print("[yellow]Tip: Make sure llama.cpp was built with CUDA support (LLAMA_CUDA=1)[/yellow]")
        raise SystemExit(1)

    # Wait for health
    for _ in range(30):
        if _check_server_running(host, port):
            break
        time.sleep(1)

    # Verify CUDA GPU is actually being used
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        gpu_pids = set()
        for line in result.stdout.strip().splitlines():
            pid_str = line.strip()
            if pid_str.isdigit():
                gpu_pids.add(int(pid_str))

        if status.pid and status.pid not in gpu_pids:
            console.print("[red]✗ CUDA GPU not detected![/red]")
            console.print("[yellow]MYCO requires CUDA GPU. CPU-only mode is not supported.[/yellow]")
            console.print("[yellow]Make sure llama.cpp was built with LLAMA_CUDA=1[/yellow]")
            manager.stop()
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception:
        pass  # nvidia-smi unavailable, skip verification

    console.print(f"[green]✓ Server ready at http://{host}:{port}[/green]\n")


def _resolve_model_name(host: str, port: int) -> str | None:
    try:
        r = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
        models = r.json().get("data", [])
        if models:
            return models[0].get("id")
    except Exception:
        pass
    return None


def _enter_interactive(model_name: str | None, no_approval: bool):
    """Start the interactive agent loop."""
    from cli.commands.interactive import interactive as _interactive_fn

    # interactive() is a Click command that uses @click.pass_context.
    # We build a minimal context so it can read config.
    from cli.utils.config import Config
    config = Config()

    # Create a Click context manually
    ctx = click.Context(_interactive_fn)
    ctx.obj = {"config": config}

    ctx.invoke(_interactive_fn, model_name=model_name, no_approval=no_approval)


# ---------------------------------------------------------------------------
# Single command
# ---------------------------------------------------------------------------

@click.command()
@click.argument("task", required=False, default=None)
@click.option("--no-approval", is_flag=True, help="Skip approval prompts")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--context", "-c", type=int, default=DEFAULT_CONTEXT,
              help=f"Context window (default: {DEFAULT_CONTEXT:,})")
def myco(task, no_approval, verbose, context):
    """MYCO — AI coding agent with entropy-aware code quality.

    Starts an interactive session with full toolset and MYCO Vision.
    Optionally pass a TASK string to run a single task and exit.

    The llama.cpp server is auto-started with CUDA GPU, flash attention,
    and the specified context window.
    """
    from cli.utils.config import Config
    from cli.utils.logging import LogConfig

    LogConfig.get_logger("myco")
    config = Config()

    host = config.get("server", "host", default=DEFAULT_HOST)
    port = config.get("server", "port", default=DEFAULT_PORT)

    # --- Auto-start server if needed ---
    if not _check_server_running(host, port):
        console.print("\n[cyan]Starting MYCO Server...[/cyan]\n")

        model_path = _find_model()
        if not model_path:
            console.print("[red]No .gguf model found in current directory.[/red]")
            console.print("Download a model and place it here, then run `myco` again.")
            raise SystemExit(1)

        _start_server(model_path, host, port, context, verbose)
    else:
        console.print(f"[green]✓ Server already running at http://{host}:{port}[/green]\n")

    # --- Resolve model name from server ---
    model_name = _resolve_model_name(host, port)

    # --- Enter interactive agent ---
    _enter_interactive(model_name=model_name, no_approval=no_approval)


if __name__ == "__main__":
    myco()
