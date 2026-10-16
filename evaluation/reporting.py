"""Format baseline and RG-LNS results for terminal and text summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def format_status(status: str) -> str:
    """Use descriptive status text while preserving stored status codes."""
    return status.replace("no_strict_e1_improvement", "no_strict_objective_improvement")


def format_metric_summary(
    baseline: Mapping[str, float],
    selected: Mapping[str, float],
    initial_solver_name: str,
    metric_keys: Sequence[str],
) -> list[str]:
    """Keep the existing metric labels and precision in two result groups."""
    labels = {"horizontal_SRA": "Horizontal SRA", "vertical_SRA": "Vertical SRA"}
    lines = []
    for title, metrics in (
        (f"ViT-T + {initial_solver_name} baseline", baseline),
        ("RG-LNS", selected),
    ):
        lines.append(f"{title}:")
        lines.extend(f"  {labels.get(key, key)}={metrics[key]:.4%}" for key in metric_keys)
        lines.append("")
    return lines
