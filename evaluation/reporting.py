"""Format baseline and RG-LNS results for terminal and text summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path


def format_run_configuration(
    *,
    dataset: str,
    split: str,
    samples: int,
    initial_solver_name: str,
    device: str,
    data_root: str | Path,
    checkpoint: str | Path,
    task: str | None = None,
) -> list[str]:
    """Describe the resolved inputs and device before evaluation starts."""
    fields = [("Dataset", dataset)]
    if task is not None:
        fields.append(("Task", task))
    fields.extend(
        [
            ("Split", split),
            ("Samples", str(samples)),
            ("Baseline", f"ViT-T + {initial_solver_name}"),
            ("Device", device),
            ("Data root", str(Path(data_root).expanduser().resolve())),
            ("Checkpoint", str(Path(checkpoint).expanduser().resolve())),
        ]
    )
    width = max(len(label) for label, _ in fields)
    return ["RG-LNS evaluation", *(f"  {label:<{width}} : {value}" for label, value in fields), ""]


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
    label_width = max((len(labels.get(key, key)) for key in metric_keys), default=0)
    lines = []
    for title, metrics in (
        (f"Baseline (ViT-T + {initial_solver_name})", baseline),
        ("Baseline + RG-LNS", selected),
    ):
        lines.append(f"{title}:")
        lines.extend(
            f"  {labels.get(key, key):<{label_width}} = {metrics[key]:.4%}"
            for key in metric_keys
        )
        lines.append("")
    return lines
