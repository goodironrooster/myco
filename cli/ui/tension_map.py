"""MYCO Dependency Tension Map - Visualize coupling stress between modules.

Shows which files are tightly coupled and where refactoring would help most.
"""

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TensionMap:
    """Visualize dependency tension between modules."""

    def __init__(self, project_root: Path):
        """Initialize tension map.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.console = Console()
        self._tensegrity_data = None

    def _load_tensegrity(self):
        """Load tensegrity analysis data."""
        if self._tensegrity_data is not None:
            return

        try:
            from myco.tensegrity import TensegrityClassifier
            classifier = TensegrityClassifier(self.project_root)
            classifier.scan()
            self._tensegrity_data = classifier
        except Exception:
            self._tensegrity_data = None

    def render(self, limit: int = 15) -> Panel:
        """Render dependency tension map.

        Args:
            limit: Maximum number of high-tension edges to show

        Returns:
            Rich Panel with tension map
        """
        self._load_tensegrity()

        if not self._tensegrity_data:
            return Panel(
                "[dim]Tensegrity analysis not available[/dim]\n[dim]Ensure project has Python files with imports[/dim]",
                title="[bold cyan]Dependency Tension Map[/bold cyan]",
                border_style="dim",
            )

        classifier = self._tensegrity_data

        # Get the internal import graph
        if not classifier._import_graph:
            return Panel(
                "[dim]No import graph available[/dim]\n[dim]Ensure project has Python files with imports[/dim]",
                title="[bold cyan]Dependency Tension Map[/bold cyan]",
                border_style="dim",
            )

        graph = classifier._import_graph.get_internal_graph()

        # Collect import edges with tension scores
        edges = []
        for importer, imported in graph.edges():
            # Tension = coupling stress (how many shared dependencies)
            importer_deps = set(graph.successors(importer))
            imported_deps = set(graph.successors(imported))
            shared = len(importer_deps & imported_deps)
            tension = min(1.0, shared / 5.0)  # Normalize: 5+ shared = max tension

            # Also factor in degree centrality
            importer_degree = graph.degree(importer)
            imported_degree = graph.degree(imported)
            degree_factor = min(1.0, (importer_degree + imported_degree) / 20.0)

            combined_tension = 0.6 * tension + 0.4 * degree_factor
            edges.append({
                "importer": importer,
                "imported": imported,
                "tension": combined_tension,
                "shared_deps": shared,
            })

        # Sort by tension (highest first)
        edges.sort(key=lambda e: e["tension"], reverse=True)
        high_tension = [e for e in edges if e["tension"] > 0.3][:limit]

        # Create table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("From", style="white", width=30)
        table.add_column("To", style="white", width=30)
        table.add_column("Tension", justify="right", style="yellow", width=10)
        table.add_column("Bar", width=22)
        table.add_column("Shared", justify="center", style="dim", width=8)
        table.add_column("Advice", style="dim", width=25)

        for edge in high_tension:
            importer = edge["importer"]
            imported = edge["imported"]
            tension = edge["tension"]
            shared = edge["shared_deps"]

            # Truncate long module names
            if len(importer) > 28:
                importer = "..." + importer[-25:]
            if len(imported) > 28:
                imported = "..." + imported[-25:]

            # Tension bar
            filled = int(tension * 20)
            empty = 20 - filled
            if tension > 0.7:
                bar = Text("█" * filled + "░" * empty, style="red")
                advice = "[red]Extract interface[/red]"
            elif tension > 0.5:
                bar = Text("█" * filled + "░" * empty, style="yellow")
                advice = "[yellow]Consider decoupling[/yellow]"
            else:
                bar = Text("▓" * filled + "░" * empty, style="green")
                advice = "[green]OK[/green]"

            table.add_row(
                importer,
                imported,
                f"{tension:.2f}",
                bar,
                str(shared),
                advice,
            )

        # Summary
        total_edges = len(edges)
        high_count = len([e for e in edges if e["tension"] > 0.5])
        avg_tension = sum(e["tension"] for e in edges) / len(edges) if edges else 0

        summary_lines = [
            f"[bold]Total Dependencies:[/bold] {total_edges} | [bold]High Tension:[/bold] {high_count}",
            f"[bold]Average Tension:[/bold] {avg_tension:.2f}",
        ]

        if high_count > 0:
            summary_lines.append("")
            summary_lines.append(f"[yellow]⚠ {high_count} high-tension edges — consider refactoring[/yellow]")

        # Render table to string
        from io import StringIO
        from rich.console import Console as RichConsole
        buffer = StringIO()
        temp_console = RichConsole(file=buffer, force_terminal=True, width=120)
        temp_console.print(table)
        table_text = buffer.getvalue()

        content = "\n".join(summary_lines) + "\n\n" + table_text

        panel = Panel(
            content,
            title="[bold cyan]Dependency Tension Map[/bold cyan]",
            border_style="cyan",
        )

        return panel


# Global instance
_tension_map: Optional[TensionMap] = None


def get_tension_map(project_root: Path) -> TensionMap:
    """Get or create global tension map instance.

    Args:
        project_root: Root directory of the project

    Returns:
        TensionMap instance
    """
    global _tension_map
    if _tension_map is None or _tension_map.project_root != project_root:
        _tension_map = TensionMap(project_root)
    return _tension_map
