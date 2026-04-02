"""Model management commands."""

from pathlib import Path

import click

from ..core.model_manager import ModelManager
from ..utils.config import Config
from ..utils.logging import LogConfig


@click.group()
@click.pass_context
def model(ctx):
    """Manage GGUF model files."""
    pass


@model.command("list")
@click.pass_context
def list_models(ctx):
    """List all GGUF models in the current directory."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    # Get models directory (current directory by default)
    models_dir = Path.cwd()

    manager = ModelManager(models_dir)
    models = manager.list_models()

    if not models:
        click.echo("No GGUF models found in current directory.")
        return

    click.echo(f"\nFound {len(models)} model(s):\n")
    click.echo(f"{'Name':<40} {'Size':<12} {'Quant':<10} {'Hash':<18}")
    click.echo("-" * 82)

    for m in models:
        click.echo(
            f"{m.name:<40} {m.size_human:<12} {m.quantization or 'Unknown':<10} {m.sha256:<18}"
        )

    click.echo()


@model.command()
@click.argument("model_path", type=click.Path(exists=False))
@click.pass_context
def info(ctx, model_path):
    """Show detailed information about a model."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    path = Path(model_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    manager = ModelManager(path.parent)

    try:
        info = manager.get_model_info(path)

        click.echo("\nModel Information:")
        click.echo("=" * 50)
        click.echo(f"Name:          {info.name}")
        click.echo(f"Path:          {info.path}")
        click.echo(f"Size:          {info.size_human} ({info.size_bytes:,} bytes)")
        click.echo(f"SHA256:        {info.sha256}")
        click.echo(f"Architecture:  {info.architecture or 'Unknown'}")
        click.echo(f"Quantization:  {info.quantization or 'Unknown'}")
        if info.parameter_count:
            click.echo(f"Parameters:    {info.parameter_count}")
        click.echo()

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    except ValueError as e:
        click.echo(f"Error: Invalid GGUF file - {e}", err=True)
        raise SystemExit(1)


@model.command()
@click.argument("model_path", type=click.Path(exists=False))
@click.pass_context
def validate(ctx, model_path):
    """Validate a GGUF model file."""
    logger = LogConfig.get_logger("gguf")
    config = ctx.obj.get("config", Config())

    path = Path(model_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    manager = ModelManager(path.parent)
    is_valid, message = manager.validate_model(path)

    if is_valid:
        click.echo(click.style("✓ ", fg="green") + message)
        raise SystemExit(0)
    else:
        click.echo(click.style("✗ ", fg="red") + message, err=True)
        raise SystemExit(1)
