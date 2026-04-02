"""GGUF CLI - Main entry point."""

import sys
from pathlib import Path

import click

__version__ = "1.0.0"

from .commands.agent import agent
from .commands.chat import chat
from .commands.interactive import interactive
from .commands.model import model
from .commands.server import server
from .commands.verify import verify
from .utils.config import Config
from .utils.logging import LogConfig


@click.group()
@click.version_option(version=__version__, prog_name="gguf")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.pass_context
def main(ctx, verbose, config):
    """GGUF - Local AI coding assistant."""
    ctx.obj = {"config": Config.load(config) if config else Config()}
    if verbose:
        LogConfig.set_verbose(True)


# Register commands
main.add_command(agent)
main.add_command(chat)
main.add_command(interactive)
main.add_command(model)
main.add_command(server)
main.add_command(verify)


if __name__ == "__main__":
    main()
