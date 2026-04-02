"""Verification CLI commands."""

import click

from ..agent.verification import verify_project, VerificationDashboard
from ..ui.verification_display import display_verification_summary


@click.group()
@click.pass_context
def verify(ctx):
    """Verify project health and status."""
    pass


@verify.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=".",
    help="Project path to verify",
)
@click.option(
    "--runtime/--no-runtime",
    default=False,
    help="Include runtime checks (server, tests, deps)",
)
@click.pass_context
def check(ctx, path, runtime):
    """Run project health verification.

    Checks:
    - Project type detection
    - Dependencies configuration
    - Test files presence
    - Linting configuration
    - Git status
    - Python syntax (for Python projects)

    With --runtime:
    - Server connection
    - Test execution
    - Dependency installation status

    \b
    Examples:
      gguf verify check
      gguf verify check --path ./myproject
      gguf verify check --runtime
    """
    click.echo("\n🔍 Running project verification...\n")

    try:
        summary = verify_project(path)
        display_verification_summary(summary)

        if runtime:
            click.echo("\n" + "=" * 50)
            click.echo("🔄 Runtime Checks")
            click.echo("=" * 50 + "\n")

            dashboard = VerificationDashboard(path)
            runtime_results = dashboard.run_runtime_checks()

            for result in runtime_results:
                status_icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "○"}.get(
                    result.status, "?"
                )
                color = {"pass": "green", "fail": "red", "warn": "yellow", "skip": "dim"}.get(
                    result.status, "white"
                )
                click.echo(f"[{color}]{status_icon}[/{color}] {result.name}: {result.message}")

    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg="red"))
        raise SystemExit(1)


@verify.command()
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    default=".",
    help="Project path",
)
@click.pass_context
def status(ctx, path):
    """Show quick project status.

    \b
    Examples:
      gguf verify status
    """
    try:
        dashboard = VerificationDashboard(path)
        info = dashboard.project_info

        click.echo(f"\n📁 Project: {path}")
        click.echo(f"   Type: {info.project_type}")
        click.echo(f"   Has requirements: {'✓' if info.has_requirements else '✗'}")
        click.echo(f"   Has tests: {'✓' if info.has_tests else '✗'}")
        click.echo(f"   Has lint: {'✓' if info.has_lint else '✗'}")

        if info.pending_verifications:
            click.echo(f"\n⚠️  Pending:")
            for p in info.pending_verifications:
                click.echo(f"   - {p}")

    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg="red"))
        raise SystemExit(1)
