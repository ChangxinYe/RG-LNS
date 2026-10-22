"""Format baseline and RG-LNS results for terminal and text summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from evaluation.runtime_info import format_runtime_environment


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


def _format_fields(title: str, fields: Sequence[tuple[str, object]]) -> list[str]:
    """Align named values without exposing Python dictionary formatting."""
    width = max((len(label) for label, _ in fields), default=0)
    return [title, *(f"  {label:<{width}} : {value}" for label, value in fields), ""]


def format_evaluation_summary(
    run_configuration: Sequence[str],
    metric_summary: Sequence[str],
    payload: Mapping,
    output_dir: str | Path,
    *,
    loss_type: str | None = None,
) -> list[str]:
    """Keep two method results alongside the context needed to reproduce them."""
    configuration = payload["configuration"]
    start = int(payload["start_index"])
    end = int(payload.get("end_index_exclusive", start + payload["evaluated_samples"]))
    # Reuse the resolved startup inputs, with a separate heading for the saved report.
    lines = ["Run Configuration", *run_configuration[1:]]
    fields = [("Sample range", f"[{start}, {end}) (zero-based, end exclusive)")]
    if loss_type is not None:
        fields.append(("Loss type", loss_type))
    for key, label in (
        ("completion_beam_width", "Completion beam width"),
        ("partial_completion_beam_width", "Partial-layout completion beam width"),
        ("mutual_top_k", "Mutual Top-K"),
        ("translation_mode", "Translation mode"),
        ("component_index", "Component index"),
        ("min_component_size", "Minimum component size"),
        ("refinement_rounds", "Maximum refinement rounds"),
        ("postprocess", "Score postprocessing"),
        ("score_batch_size", "Score batch size"),
        ("runtime_warmup_samples", "Configured warmup samples"),
        ("save_visuals", "Save visualizations"),
        ("visual_limit", "Visualization limit"),
        ("save_predictions", "Save predictions"),
        ("save_completion_details", "Save search diagnostics"),
    ):
        if key in configuration:
            fields.append((label, configuration[key]))
    evaluation_settings = _format_fields("Evaluation Settings", fields)
    solver_config = configuration.get("initial_solver", {})
    if solver_config:
        lines.extend(_format_fields("Initial Solver Settings", [
            (key, value) for key, value in solver_config.items() if key != "implementation"
        ]))
    lines.extend(metric_summary)

    comparison = payload["comparison"]
    baseline, selected = payload["baseline"], payload["s1a4"]
    fields = [
        (f"{key} gain", f"{100.0 * (selected[key] - baseline[key]):+.4f} percentage points")
        for key in ("PA", "AA", "SRA")
    ]
    fields.extend((label, comparison[key]) for key, label in (
        ("improved_puzzles", "Puzzles with higher AA"),
        ("damaged_puzzles", "Puzzles with lower AA"),
        ("unchanged_puzzles", "Puzzles with unchanged AA"),
        ("changed_predictions", "Changed layouts"),
        ("repaired_to_perfect", "Failed to perfect layouts"),
        ("broken_perfect", "Perfect to failed layouts"),
    ))
    lines.extend(_format_fields("Improvement Summary (relative to Baseline)", fields))

    retrieval = payload.get("neighbor_retrieval")
    if retrieval is not None:
        fields = [
            ("True-Neighbor MRR", f"{retrieval['neighbor_mrr']:.6f}"),
            ("Mean True-Neighbor Rank", f"{retrieval['mean_true_neighbor_rank']:.3f}"),
        ]
        recall_prefix = "neighbor_recall_at_"
        recall_keys = sorted(
            (key for key in retrieval if key.startswith(recall_prefix)),
            key=lambda key: int(key.removeprefix(recall_prefix)),
        )
        fields.extend(
            (f"True-Neighbor Recall@{key.removeprefix(recall_prefix)}", f"{retrieval[key]:.4%}")
            for key in recall_keys
        )
        lines.extend(_format_fields("Neighbor Retrieval (raw ViT-T compatibility)", fields))

    runtime = payload["runtime"]
    lines.extend([
        "Runtime",
        f"  Unit: seconds per puzzle | warmup={runtime['warmup_samples']} | "
        f"measured={runtime['measured_samples']}",
    ])
    for key, label in (
        ("compatibility", "Compatibility scoring"),
        ("initial_solver", "Initial solver"),
        ("refinement_total", "RG-LNS refinement"),
        ("inference_total", "Total inference"),
    ):
        stats = runtime["stages"][key]
        lines.append(
            f"  {label:<21}: mean={stats['mean']:.6f} | median={stats['median']:.6f} | "
            f"std={stats['std']:.6f} | p95={stats['p95']:.6f} | total={stats['total']:.6f}"
        )
    lines.extend([
        "  RG-LNS refinement includes automatic partial-layout completion.",
        f"  Evaluation elapsed: {payload['elapsed_seconds']:.6f} seconds (includes loop overhead).",
        "",
    ])

    output_dir = Path(output_dir).expanduser().resolve()
    lines.extend(["Output Files", f"  Directory: {output_dir}", "  summary.txt"])
    for name in (
        "metrics.json", "sample_results.csv", "predictions.npz", "visuals",
        "pa_repaired_images.txt", "pa_broken_images.txt", "round_results.csv",
        "candidates.csv", "pose_results.csv", "completion_results.csv",
        "pa_repaired_vs_s1a3_images.txt", "pa_broken_vs_s1a3_images.txt",
    ):
        path = output_dir / name
        if not path.exists():
            continue
        description = name + ("/" if path.is_dir() else "")
        if name in {"pa_repaired_images.txt", "pa_broken_images.txt"}:
            key = "repaired_images" if name == "pa_repaired_images.txt" else "broken_images"
            description += f" ({payload['pa_case_lists'][key]} samples)"
        lines.append(f"  {description}")
    lines.extend(["", *evaluation_settings, *format_runtime_environment(payload["runtime_environment"]), ""])
    return lines
