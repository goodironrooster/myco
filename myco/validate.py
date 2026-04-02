# ⊕ H:0.50 | press:bootstrap | age:0 | drift:+0.00
"""Validation harness for myco interventions.

Analyzes past interventions to determine whether they correlated with entropy improvement.
This is read-only analysis - no modifications to codebase.

Usage:
    myco validate  # Analyze all sessions
    myco validate --limit 10  # Last 10 sessions
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .entropy import analyze_entropy, ImportGraphBuilder, EntropyCalculator
from .session_log import SessionLogger, LogEntry
from .world import WorldModel


@dataclass
class InterventionRecord:
    """Record of a single intervention and its outcome."""
    session_id: str
    timestamp: str
    file_path: str
    intervention_type: str
    H_before: float
    H_after: float
    neighbor_H_before: float
    neighbor_H_after: float
    sessions_elapsed: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "file_path": self.file_path,
            "intervention_type": self.intervention_type,
            "H_before": self.H_before,
            "H_after": self.H_after,
            "neighbor_H_before": self.neighbor_H_before,
            "neighbor_H_after": self.neighbor_H_after,
            "sessions_elapsed": self.sessions_elapsed,
        }


def find_intervention_entries(session_entries: list[LogEntry]) -> list[LogEntry]:
    """Find tool call entries that represent interventions.
    
    Interventions are: write_file, edit_file operations.
    We consider any file modification as a potential intervention.
    """
    interventions = []
    for entry in session_entries:
        if entry.event_type == "tool_call":
            # Check if this was a file modification
            tool_name = entry.data.get("tool_name", "")
            if not tool_name:
                # Try alternate key
                tool_name = entry.data.get("tool", "")
            if tool_name in ("write_file", "edit_file"):
                interventions.append(entry)
    
    return interventions


def compute_neighbor_entropy(
    project_root: Path,
    file_path: Path,
    builder: ImportGraphBuilder,
    calculator: EntropyCalculator
) -> float:
    """Compute average entropy of modules that import or are imported by the target.
    
    Args:
        project_root: Project root
        file_path: Target file
        builder: Import graph builder
        calculator: Entropy calculator
        
    Returns:
        Average entropy of neighboring modules
    """
    module_name = builder._path_to_module_name(file_path)
    
    if module_name not in builder.modules:
        return 0.0
    
    module_info = builder.modules[module_name]
    neighbors = set(module_info.imports + module_info.imported_by)
    
    if not neighbors:
        return 0.0
    
    neighbor_entropies = []
    for neighbor in neighbors:
        if neighbor in builder.modules:
            neighbor_path = builder.modules[neighbor].path
            if neighbor_path.exists():
                H = calculator.calculate_module_entropy(neighbor)
                neighbor_entropies.append(H)
    
    return sum(neighbor_entropies) / len(neighbor_entropies) if neighbor_entropies else 0.0


def analyze_intervention(
    project_root: Path,
    intervention_entry: LogEntry,
    world: WorldModel
) -> Optional[InterventionRecord]:
    """Analyze a single intervention's impact.
    
    Args:
        project_root: Project root
        intervention_entry: Log entry for the intervention
        world: World model for session timing
        
    Returns:
        InterventionRecord or None if analysis fails
    """
    # Get file path from tool call
    tool_args = intervention_entry.data.get("tool_args", {})
    if not tool_args:
        tool_args = intervention_entry.data.get("arguments", {})
    
    file_path_str = tool_args.get("path", "")
    
    if not file_path_str:
        return None
    
    file_path = project_root / file_path_str
    
    if not file_path.exists():
        return None
    
    # Get intervention type from press or tool result
    result = intervention_entry.data.get("result", "")
    intervention_type = "unknown"
    for press_type in ["decompose", "interface_inversion", "tension_extraction", 
                       "compression_collapse", "entropy_drain", "attractor_escape"]:
        if press_type in result.lower():
            intervention_type = press_type
            break
    
    # Build current import graph
    builder = ImportGraphBuilder(project_root)
    builder.scan()
    calculator = EntropyCalculator(builder)
    
    # Compute current entropy
    H_after = calculator.calculate_module_entropy(
        builder._path_to_module_name(file_path)
    )
    
    # Compute current neighbor entropy
    neighbor_H_after = compute_neighbor_entropy(
        project_root, file_path, builder, calculator
    )
    
    # Estimate H_before from drift in annotation (if available)
    # This is approximate - ideally we'd have historical snapshots
    from .stigma import StigmaReader
    try:
        reader = StigmaReader(file_path)
        annotation = reader.read_annotation()
        if annotation:
            # H_before ≈ H_after - drift
            H_before = H_after - annotation.drift
            neighbor_H_before = neighbor_H_after  # Approximation
        else:
            H_before = H_after
            neighbor_H_before = neighbor_H_after
    except (SyntaxError, FileNotFoundError):
        H_before = H_after
        neighbor_H_before = neighbor_H_after
    
    # Compute sessions elapsed
    intervention_session = intervention_entry.data.get("session_id", "")
    sessions_elapsed = world.session_count  # Approximation
    
    return InterventionRecord(
        session_id=intervention_entry.data.get("session_id", "unknown"),
        timestamp=intervention_entry.timestamp,
        file_path=file_path_str,
        intervention_type=intervention_type,
        H_before=H_before,
        H_after=H_after,
        neighbor_H_before=neighbor_H_before,
        neighbor_H_after=neighbor_H_after,
        sessions_elapsed=sessions_elapsed
    )


def validate_interventions(
    project_root: Path,
    limit: Optional[int] = None
) -> list[InterventionRecord]:
    """Validate all interventions from session history.
    
    Args:
        project_root: Project root
        limit: Limit number of sessions to analyze
        
    Returns:
        List of InterventionRecord objects
    """
    project_root = Path(project_root)
    logger = SessionLogger(project_root)
    world = WorldModel.load(project_root)
    
    # Read all session entries
    entries = logger.read_log_file()
    
    if not entries:
        return []
    
    # Group by session
    sessions = {}
    for entry in entries:
        session_id = entry.data.get("session_id", "unknown")
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(entry)
    
    # Analyze interventions from recent sessions
    session_list = list(sessions.items())
    if limit:
        session_list = session_list[-limit:]
    
    records = []
    for session_id, session_entries in session_list:
        interventions = find_intervention_entries(session_entries)
        for intervention in interventions:
            record = analyze_intervention(project_root, intervention, world)
            if record:
                records.append(record)
    
    return records


def format_validation_table(records: list[InterventionRecord]) -> str:
    """Format validation records as a table.
    
    Args:
        records: List of intervention records
        
    Returns:
        Formatted table string
    """
    if not records:
        return "No interventions found in session history."
    
    lines = [
        "=" * 120,
        "INTERVENTION VALIDATION REPORT",
        "=" * 120,
        "",
        f"{'Session':<20} {'File':<30} {'Type':<20} {'H_before':<10} {'H_after':<10} {'ΔH':<8} {'Neighbor ΔH':<12}",
        "-" * 120,
    ]
    
    for record in records:
        delta_H = record.H_after - record.H_before
        neighbor_delta = record.neighbor_H_after - record.neighbor_H_before
        
        lines.append(
            f"{record.session_id[:19]:<20} "
            f"{record.file_path[:29]:<30} "
            f"{record.intervention_type:<20} "
            f"{record.H_before:<10.3f} "
            f"{record.H_after:<10.3f} "
            f"{delta_H:<+8.3f} "
            f"{neighbor_delta:<+12.3f}"
        )
    
    # Summary statistics
    lines.append("")
    lines.append("-" * 120)
    lines.append("SUMMARY")
    lines.append("-" * 120)
    
    # Group by intervention type
    by_type = {}
    for record in records:
        t = record.intervention_type
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(record)
    
    lines.append(f"{'Intervention Type':<25} {'Count':<8} {'Avg ΔH':<12} {'Improvements':<15}")
    lines.append("-" * 60)
    
    for intervention_type, type_records in sorted(by_type.items()):
        count = len(type_records)
        avg_delta = sum(r.H_after - r.H_before for r in type_records) / count
        improvements = sum(1 for r in type_records if r.H_after < r.H_before)
        
        lines.append(
            f"{intervention_type:<25} "
            f"{count:<8} "
            f"{avg_delta:<+12.3f} "
            f"{improvements}/{count:<10}"
        )
    
    # Overall statistics
    total = len(records)
    overall_avg = sum(r.H_after - r.H_before for r in records) / total if total > 0 else 0
    overall_improvements = sum(1 for r in records if r.H_after < r.H_before)
    
    lines.append("-" * 60)
    lines.append(
        f"{'TOTAL':<25} "
        f"{total:<8} "
        f"{overall_avg:<+12.3f} "
        f"{overall_improvements}/{total:<10}"
    )
    
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  - ΔH < 0: Entropy decreased (improvement for crystallized modules)")
    lines.append("  - ΔH > 0: Entropy increased (may indicate degradation)")
    lines.append("  - Improvements: Count of interventions where entropy moved toward 0.5")
    lines.append("=" * 120)
    
    return "\n".join(lines)


@dataclass
class SessionMetrics:
    """Metrics for a single session."""
    session_id: str
    iterations: int
    tokens: int
    joules: float
    entropy_delta: float
    blocked_count: int = 0

    def efficiency_ratio(self) -> float:
        """Compute efficiency ratio: entropy improvement per joule."""
        if self.joules == 0:
            return 0.0
        # Negative entropy_delta means improvement
        return -self.entropy_delta / self.joules

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "iterations": self.iterations,
            "tokens": self.tokens,
            "joules": self.joules,
            "entropy_delta": self.entropy_delta,
            "blocked_count": self.blocked_count,
            "efficiency_ratio": self.efficiency_ratio(),
        }


def extract_session_metrics(session_entries: list[LogEntry]) -> Optional[SessionMetrics]:
    """Extract metrics from session entries.

    Args:
        session_entries: List of log entries for a session

    Returns:
        SessionMetrics or None if session end not found
    """
    # Find session_end entry
    for entry in session_entries:
        if entry.event_type == "session_end":
            data = entry.data
            return SessionMetrics(
                session_id=data.get("session_id", "unknown"),
                iterations=data.get("iterations", 0),
                tokens=data.get("tokens", 0),
                joules=data.get("joules", 0.0),
                entropy_delta=data.get("entropy_delta", 0.0),
                blocked_count=0  # Would need to count gate_check events with permitted=False
            )
    return None


def compare_last_n_sessions(
    project_root: Path,
    n: int = 20
) -> list[SessionMetrics]:
    """Compare metrics from last N sessions.

    Args:
        project_root: Project root
        n: Number of sessions to compare

    Returns:
        List of SessionMetrics, newest first
    """
    logger = SessionLogger(project_root)
    world = WorldModel.load(project_root)

    # Read all entries
    entries = logger.read_log_file()

    if not entries:
        return []

    # Group by session
    sessions = {}
    for entry in entries:
        session_id = entry.data.get("session_id", "unknown")
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(entry)

    # Extract metrics from each session
    metrics = []
    session_list = list(sessions.items())[-n:]  # Last N sessions

    for session_id, session_entries in session_list:
        session_metrics = extract_session_metrics(session_entries)
        if session_metrics:
            # Count blocked actions
            blocked = sum(
                1 for e in session_entries
                if e.event_type == "gate_check" and not e.data.get("permitted", True)
            )
            session_metrics.blocked_count = blocked
            metrics.append(session_metrics)

    return metrics


def format_session_comparison(metrics: list[SessionMetrics]) -> str:
    """Format session comparison as trend table.

    Args:
        metrics: List of session metrics

    Returns:
        Formatted table string
    """
    if not metrics:
        return "No session data available for comparison."

    lines = [
        "=" * 100,
        "SESSION TREND ANALYSIS (Instrument Panel)",
        "=" * 100,
        "",
        f"Sessions analyzed: {len(metrics)}",
        "",
        f"{'Session':<20} {'Iterations':<12} {'Tokens':<10} {'Joules':<10} {'ΔH':<8} {'Blocked':<10} {'Efficiency':<12}",
        "-" * 100,
    ]

    for m in metrics:
        efficiency = m.efficiency_ratio()
        lines.append(
            f"{m.session_id[:19]:<20} "
            f"{m.iterations:<12} "
            f"{m.tokens:<10} "
            f"{m.joules:<10.2f} "
            f"{m.entropy_delta:<+8.3f} "
            f"{m.blocked_count:<10} "
            f"{efficiency:<+12.4f}"
        )

    # Trend analysis
    lines.append("")
    lines.append("-" * 100)
    lines.append("TREND ANALYSIS")
    lines.append("-" * 100)

    if len(metrics) >= 2:
        # Compute trends (comparing first half to second half)
        mid = len(metrics) // 2
        first_half = metrics[:mid]
        second_half = metrics[mid:]

        # Average metrics for each half
        def avg_metrics(session_list):
            if not session_list:
                return None
            return {
                "iterations": sum(s.iterations for s in session_list) / len(session_list),
                "joules": sum(s.joules for s in session_list) / len(session_list),
                "entropy_delta": sum(s.entropy_delta for s in session_list) / len(session_list),
                "blocked_rate": sum(s.blocked_count for s in session_list) / len(session_list) / max(sum(s.iterations for s in session_list), 1),
                "efficiency": sum(s.efficiency_ratio() for s in session_list) / len(session_list),
            }

        first_avg = avg_metrics(first_half)
        second_avg = avg_metrics(second_half)

        if first_avg and second_avg:
            lines.append("")
            lines.append(f"{'Metric':<20} {'First Half':<15} {'Second Half':<15} {'Change':<15} {'Trend':<15}")
            lines.append("-" * 80)

            # Entropy delta trend (negative is good for crystallized substrate)
            delta_change = second_avg["entropy_delta"] - first_avg["entropy_delta"]
            delta_trend = "↓ Improving" if delta_change < -0.01 else "↑ Worsening" if delta_change > 0.01 else "→ Stable"
            lines.append(
                f"{'Entropy Δ':<20} {first_avg['entropy_delta']:<+15.3f} {second_avg['entropy_delta']:<+15.3f} "
                f"{delta_change:<+15.3f} {delta_trend:<15}"
            )

            # Blocked rate trend (lower is better)
            blocked_change = second_avg["blocked_rate"] - first_avg["blocked_rate"]
            blocked_trend = "↓ Improving" if blocked_change < 0 else "↑ Worsening" if blocked_change > 0 else "→ Stable"
            lines.append(
                f"{'Blocked Rate':<20} {first_avg['blocked_rate']:<+15.4f} {second_avg['blocked_rate']:<+15.4f} "
                f"{blocked_change:<+15.4f} {blocked_trend:<15}"
            )

            # Iterations trend (context dependent)
            iter_change = second_avg["iterations"] - first_avg["iterations"]
            iter_trend = "↑ More work" if iter_change > 0.5 else "↓ Less work" if iter_change < -0.5 else "→ Stable"
            lines.append(
                f"{'Iterations':<20} {first_avg['iterations']:<+15.1f} {second_avg['iterations']:<+15.1f} "
                f"{iter_change:<+15.1f} {iter_trend:<15}"
            )

            # Efficiency trend (higher is better)
            eff_change = second_avg["efficiency"] - first_avg["efficiency"]
            eff_trend = "↑ Improving" if eff_change > 0.01 else "↓ Worsening" if eff_change < -0.01 else "→ Stable"
            lines.append(
                f"{'Efficiency':<20} {first_avg['efficiency']:<+15.4f} {second_avg['efficiency']:<+15.4f} "
                f"{eff_change:<+15.4f} {eff_trend:<15}"
            )

            lines.append("")
            lines.append("Trend Interpretation:")
            lines.append("  - Entropy Δ: Negative = entropy decreased (good for crystallized substrate)")
            lines.append("  - Blocked Rate: Lower = gate permits more actions")
            lines.append("  - Efficiency: Higher = more entropy improvement per joule")
            lines.append("")

            # Overall assessment
            improving_count = sum([
                1 if delta_change < -0.01 else 0,
                1 if blocked_change < 0 else 0,
                1 if eff_change > 0.01 else 0,
            ])

            if improving_count >= 2:
                lines.append("Overall: ✓ System improving (2+ metrics trending correctly)")
            elif improving_count == 1:
                lines.append("Overall: ⚠ Mixed results (1 metric improving)")
            else:
                lines.append("Overall: ✗ System degrading (no metrics improving)")

    lines.append("=" * 100)

    return "\n".join(lines)


def assess_gate_readiness(metrics: list[SessionMetrics], min_sessions: int = 10) -> dict:
    """Assess whether gate threshold relaxation is safe.

    Per roadmap: Step 4 (gate threshold relaxation) should only run after
    min_sessions of efficiency ratio data.

    Args:
        metrics: List of session metrics
        min_sessions: Minimum sessions required (default 10)

    Returns:
        Dict with readiness assessment
    """
    result = {
        "ready": False,
        "sessions_available": len(metrics),
        "min_sessions_required": min_sessions,
        "efficiency_trend": 0.0,
        "blocked_rate_trend": 0.0,
        "recommendation": "",
    }

    if len(metrics) < min_sessions:
        result["recommendation"] = (
            f"Need {min_sessions - len(metrics)} more sessions before gate threshold relaxation."
        )
        return result

    # Calculate trends
    mid = len(metrics) // 2
    first_half = metrics[:mid]
    second_half = metrics[mid:]

    def avg_efficiency(session_list):
        if not session_list:
            return 0.0
        return sum(s.efficiency_ratio() for s in session_list) / len(session_list)

    def avg_blocked_rate(session_list):
        if not session_list:
            return 0.0
        total_iterations = sum(s.iterations for s in session_list)
        if total_iterations == 0:
            return 0.0
        return sum(s.blocked_count for s in session_list) / total_iterations

    eff_first = avg_efficiency(first_half)
    eff_second = avg_efficiency(second_half)
    blocked_first = avg_blocked_rate(first_half)
    blocked_second = avg_blocked_rate(second_half)

    result["efficiency_trend"] = eff_second - eff_first
    result["blocked_rate_trend"] = blocked_second - blocked_first

    # Assess readiness
    # Safe to relax if:
    # 1. Efficiency is stable or improving (trend >= -0.01)
    # 2. Blocked rate is stable or decreasing (trend <= 0.01)
    efficiency_ok = result["efficiency_trend"] >= -0.01
    blocked_ok = result["blocked_rate_trend"] <= 0.01

    if efficiency_ok and blocked_ok:
        result["ready"] = True
        result["recommendation"] = (
            "✓ Gate threshold relaxation is SAFE. "
            f"Efficiency trend: {result['efficiency_trend']:+.4f}, "
            f"Blocked rate trend: {result['blocked_rate_trend']:+.4f}"
        )
    else:
        issues = []
        if not efficiency_ok:
            issues.append(f"efficiency declining ({result['efficiency_trend']:+.4f})")
        if not blocked_ok:
            issues.append(f"blocked rate increasing ({result['blocked_rate_trend']:+.4f})")
        result["recommendation"] = (
            f"✗ Gate threshold relaxation is NOT SAFE. Issues: {', '.join(issues)}. "
            "Continue monitoring."
        )

    return result


def format_gate_readiness(assessment: dict) -> str:
    """Format gate readiness assessment.

    Args:
        assessment: Dict from assess_gate_readiness

    Returns:
        Formatted string
    """
    lines = [
        "=" * 80,
        "GATE THRESHOLD RELAXATION READINESS (Step 4)",
        "=" * 80,
        "",
        f"Sessions available: {assessment['sessions_available']}",
        f"Sessions required: {assessment['min_sessions_required']}",
        "",
    ]

    if assessment["ready"]:
        lines.append(f"Status: ✓ READY for threshold relaxation")
        lines.append("")
        lines.append(f"Efficiency trend: {assessment['efficiency_trend']:+.4f} (higher = better)")
        lines.append(f"Blocked rate trend: {assessment['blocked_rate_trend']:+.4f} (lower = better)")
    else:
        lines.append(f"Status: ✗ NOT READY")
        lines.append("")

    lines.append("")
    lines.append(f"Recommendation: {assessment['recommendation']}")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
