"""
@file: metric_logger.py
@description: Puzzle-level 度量学习拼图 baseline 的日志、CSV、曲线和 summary 保存工具。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.6
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

try:
    from .experiment_runtime_info import runtime_environment_lines
except ImportError:
    from experiment_runtime_info import runtime_environment_lines


SCALAR_TYPES = (int, float, str, bool, type(None))
EVAL_SPLITS = ["train", "valid", "test"]
TRIPLET_METRIC_KEYS = [
    "loss",
    "triplet_loss",
    "triplet_acc",
    "pos_dist",
    "hard_neg_dist",
    "margin_gap",
    "negatives_per_anchor",
    "hard_neg_index",
    "embedding_std",
    "embedding_norm",
]
SEMI_HARD_METRIC_KEYS = [
    "loss",
    "triplet_loss",
    "triplet_acc",
    "pos_dist",
    "hard_neg_dist",
    "selected_neg_dist",
    "margin_gap",
    "selected_margin_gap",
    "semi_hard_fraction",
    "fallback_fraction",
    "active_triplet_fraction",
    "negatives_per_anchor",
    "selected_neg_index",
    "embedding_std",
    "embedding_norm",
]
SAMPLED_TRIPLET_METRIC_KEYS = [
    "loss",
    "triplet_loss",
    "triplet_acc",
    "pos_dist",
    "hard_neg_dist",
    "selected_neg_dist",
    "margin_gap",
    "selected_margin_gap",
    "hard_sample_fraction",
    "semi_hard_sample_fraction",
    "easy_sample_fraction",
    "active_triplet_fraction",
    "eligible_candidate_fraction",
    "sampling_entropy",
    "hard_negative_count",
    "selected_negative_rank",
    "negatives_per_anchor",
    "selected_neg_index",
    "embedding_std",
    "embedding_norm",
]
INFONCE_METRIC_KEYS = [
    "loss",
    "contrastive_loss",
    "candidate_top1_acc",
    "positive_rank",
    "mrr",
    "positive_similarity",
    "hardest_negative_similarity",
    "similarity_gap",
    "negatives_per_anchor",
    "temperature",
    "embedding_std",
    "embedding_norm",
]
ASSIGNMENT_METRIC_KEYS = [
    "loss",
    "hard_triplet_loss",
    "assignment_loss",
    "row_assignment_loss",
    "column_assignment_loss",
    "triplet_acc",
    "row_top1_acc",
    "column_top1_acc",
    "bidirectional_top1_acc",
    "mutual_top1_acc",
    "positive_rank",
    "mrr",
    "pos_dist",
    "hard_neg_dist",
    "margin_gap",
    "matched_edges",
    "candidates_per_edge",
    "negatives_per_anchor",
    "temperature",
    "assignment_weight",
    "hard_triplet_weight",
    "embedding_std",
    "embedding_norm",
]
METRIC_KEYS = list(
    dict.fromkeys(
        [
            *TRIPLET_METRIC_KEYS,
            *SEMI_HARD_METRIC_KEYS,
            *SAMPLED_TRIPLET_METRIC_KEYS,
            *INFONCE_METRIC_KEYS,
            *ASSIGNMENT_METRIC_KEYS,
        ]
    )
)
ASSEMBLY_METRIC_KEYS = ["pa", "aa", "perfect_puzzles", "correct_pieces", "total_pieces", "samples", "elapsed_seconds"]


def normalize_loss_type(loss_type: str | None) -> str:
    normalized = str(loss_type).lower()
    if normalized in {
        "hard_triplet",
        "semi_hard_triplet",
        "random_triplet",
        "distance_weighted_triplet",
        "infonce",
        "bidirectional_assignment",
    }:
        return normalized
    return "hard_triplet"


def loss_type_from_args(args) -> str:
    return normalize_loss_type(getattr(args, "loss_type", "hard_triplet"))


def loss_title(loss_type: str) -> str:
    titles = {
        "hard_triplet": "Hard Triplet",
        "semi_hard_triplet": "Semi-Hard Triplet",
        "random_triplet": "Random Triplet",
        "distance_weighted_triplet": "Distance-Weighted Triplet",
        "infonce": "InfoNCE",
        "bidirectional_assignment": "Bidirectional Assignment",
    }
    return titles[normalize_loss_type(loss_type)]


def selection_metric_for_loss(loss_type: str) -> str:
    normalized = normalize_loss_type(loss_type)
    if normalized == "infonce":
        return "candidate_top1_acc"
    if normalized == "bidirectional_assignment":
        return "mutual_top1_acc"
    return "triplet_acc"


def metric_keys_for_loss(loss_type: str) -> list[str]:
    normalized = normalize_loss_type(loss_type)
    if normalized == "infonce":
        return INFONCE_METRIC_KEYS
    if normalized == "bidirectional_assignment":
        return ASSIGNMENT_METRIC_KEYS
    if normalized == "semi_hard_triplet":
        return SEMI_HARD_METRIC_KEYS
    if normalized in {"random_triplet", "distance_weighted_triplet"}:
        return SAMPLED_TRIPLET_METRIC_KEYS
    return TRIPLET_METRIC_KEYS


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_value(value, digits: int = 6) -> str:
    if is_missing(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def args_to_dict(args) -> dict:
    output = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, SCALAR_TYPES):
            output[key] = value
        else:
            output[key] = str(value)
    return output


def write_training_args(args, path: Path) -> None:
    lines = [f"Puzzle-Level {loss_title(loss_type_from_args(args))} Training Arguments", "=" * 80]
    for key, value in sorted(args_to_dict(args).items()):
        lines.append(f"{key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rows(rows: list[dict], fieldnames: list[str], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_epoch_log(record: dict, path: Path) -> None:
    lines = [f"epoch {record['epoch']:03d}"]
    train_opt = record.get("train_optimization", {})
    if train_opt:
        lines.append(format_metric_line("train_optimization", train_opt))
    for split_name in EVAL_SPLITS:
        metrics = record.get(split_name, {})
        if metrics:
            lines.append(format_metric_line(split_name, metrics))

    assembly = record.get("assembly", {})
    if isinstance(assembly, dict):
        for split_name in EVAL_SPLITS:
            metrics = assembly.get(split_name, {})
            if metrics:
                lines.append(
                    f"assembly_{split_name}: "
                    f"PA={metrics['pa']:.6f} "
                    f"AA={metrics['aa']:.6f} "
                    f"perfect={metrics['perfect_puzzles']}/{metrics['samples']} "
                    f"correct={metrics['correct_pieces']}/{metrics['total_pieces']} "
                    f"elapsed={format_seconds(metrics['elapsed_seconds'])}"
                )
    if "learning_rate" in record:
        lines.append(f"learning_rate: {record['learning_rate']:.8g}")
    if "backbone_learning_rate" in record:
        lines.append(f"backbone_learning_rate: {record['backbone_learning_rate']:.8g}")
    if "head_learning_rate" in record:
        lines.append(f"head_learning_rate: {record['head_learning_rate']:.8g}")
    if "elapsed_seconds" in record:
        lines.append(f"elapsed: {format_seconds(record['elapsed_seconds'])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def format_metric_line(name: str, metrics: dict) -> str:
    fields = [("loss", "loss")]
    if "assignment_loss" in metrics:
        fields.extend(
            [
                ("hard_triplet_loss", "hard_triplet_loss"),
                ("assignment_loss", "assignment_loss"),
                ("triplet_acc", "triplet_acc"),
                ("row_top1_acc", "row_top1"),
                ("column_top1_acc", "column_top1"),
                ("mutual_top1_acc", "mutual_top1"),
                ("positive_rank", "positive_rank"),
                ("mrr", "mrr"),
                ("pos_dist", "pos_dist"),
                ("hard_neg_dist", "hard_neg_dist"),
            ]
        )
    elif "contrastive_loss" in metrics:
        fields.extend(
            [
                ("contrastive_loss", "infonce_loss"),
                ("candidate_top1_acc", "candidate_top1_acc"),
                ("positive_rank", "positive_rank"),
                ("mrr", "mrr"),
                ("positive_similarity", "pos_sim"),
                ("hardest_negative_similarity", "hard_neg_sim"),
                ("similarity_gap", "similarity_gap"),
            ]
        )
    elif "hard_sample_fraction" in metrics:
        fields.extend(
            [
                ("triplet_loss", "triplet_loss"),
                ("triplet_acc", "candidate_top1_acc"),
                ("pos_dist", "pos_dist"),
                ("hard_neg_dist", "hard_neg_dist"),
                ("selected_neg_dist", "selected_neg_dist"),
                ("selected_negative_rank", "selected_rank"),
                ("hard_sample_fraction", "hard_fraction"),
                ("semi_hard_sample_fraction", "semi_hard_fraction"),
                ("easy_sample_fraction", "easy_fraction"),
                ("active_triplet_fraction", "active_fraction"),
                ("sampling_entropy", "sampling_entropy"),
            ]
        )
    elif "selected_neg_dist" in metrics:
        fields.extend(
            [
                ("triplet_loss", "triplet_loss"),
                ("triplet_acc", "candidate_top1_acc"),
                ("pos_dist", "pos_dist"),
                ("hard_neg_dist", "hard_neg_dist"),
                ("selected_neg_dist", "selected_neg_dist"),
                ("selected_margin_gap", "selected_gap"),
                ("semi_hard_fraction", "semi_hard_fraction"),
                ("fallback_fraction", "fallback_fraction"),
                ("active_triplet_fraction", "active_fraction"),
            ]
        )
    else:
        fields.extend(
            [
                ("triplet_loss", "triplet_loss"),
                ("triplet_acc", "triplet_acc"),
                ("pos_dist", "pos_dist"),
                ("hard_neg_dist", "hard_neg_dist"),
                ("margin_gap", "margin_gap"),
            ]
        )
    fields.extend(
        [
            ("embedding_std", "embedding_std"),
            ("embedding_norm", "embedding_norm"),
            ("negatives_per_anchor", "negatives"),
        ]
    )
    parts = []
    for key, label in fields:
        digits = 2 if key == "negatives_per_anchor" else 6
        parts.append(f"{label}={format_value(metrics.get(key), digits=digits)}")
    return f"{name}: " + " ".join(parts)


def write_metric_outputs(history: list[dict], output_dir: Path, loss_type: str = "hard_triplet") -> None:
    loss_type = normalize_loss_type(loss_type)
    active_metric_keys = metric_keys_for_loss(loss_type)
    csv_dir = output_dir / "csvs"
    curves_dir = output_dir / "curves"
    rows = []
    for record in history:
        row = {
            "epoch": record["epoch"],
            "learning_rate": record.get("learning_rate"),
            "backbone_learning_rate": record.get("backbone_learning_rate"),
            "head_learning_rate": record.get("head_learning_rate"),
            "elapsed_seconds": record.get("elapsed_seconds"),
        }
        train_opt = record.get("train_optimization", {})
        for key in active_metric_keys:
            row[f"train_optimization_{key}"] = train_opt.get(key)
        for split_name in EVAL_SPLITS:
            metrics = record.get(split_name, {})
            for key in active_metric_keys:
                row[f"{split_name}_{key}"] = metrics.get(key)
        assembly = record.get("assembly", {})
        if isinstance(assembly, dict):
            for split_name in EVAL_SPLITS:
                metrics = assembly.get(split_name, {})
                for key in ASSEMBLY_METRIC_KEYS:
                    row[f"assembly_{split_name}_{key}"] = metrics.get(key)
        rows.append(row)

    fieldnames = [
        "epoch",
        "learning_rate",
        "backbone_learning_rate",
        "head_learning_rate",
        "elapsed_seconds",
    ]
    fieldnames.extend(f"train_optimization_{key}" for key in active_metric_keys)
    for split_name in EVAL_SPLITS:
        fieldnames.extend(f"{split_name}_{key}" for key in active_metric_keys)
    for split_name in EVAL_SPLITS:
        fieldnames.extend(f"assembly_{split_name}_{key}" for key in ASSEMBLY_METRIC_KEYS)
    write_rows(rows, fieldnames, csv_dir / "metrics.csv")
    write_rows(
        rows,
        ["epoch", "learning_rate", "backbone_learning_rate", "head_learning_rate"],
        csv_dir / "learning_rate.csv",
    )
    write_rows(rows, ["epoch", "train_optimization_loss", "train_loss", "valid_loss", "test_loss"], csv_dir / "loss.csv")
    if loss_type == "bidirectional_assignment":
        write_rows(
            rows,
            [
                "epoch",
                "train_optimization_mutual_top1_acc",
                "valid_mutual_top1_acc",
                "train_optimization_row_top1_acc",
                "valid_row_top1_acc",
                "train_optimization_column_top1_acc",
                "valid_column_top1_acc",
                "train_optimization_bidirectional_top1_acc",
                "valid_bidirectional_top1_acc",
                "train_optimization_mrr",
                "valid_mrr",
                "train_optimization_positive_rank",
                "valid_positive_rank",
            ],
            csv_dir / "assignment_accuracy.csv",
        )
        write_rows(
            rows,
            [
                "epoch",
                "train_optimization_hard_triplet_loss",
                "valid_hard_triplet_loss",
                "train_optimization_assignment_loss",
                "valid_assignment_loss",
                "train_optimization_row_assignment_loss",
                "valid_row_assignment_loss",
                "train_optimization_column_assignment_loss",
                "valid_column_assignment_loss",
            ],
            csv_dir / "assignment_loss.csv",
        )
    elif loss_type == "infonce":
        write_rows(
            rows,
            [
                "epoch",
                "train_optimization_candidate_top1_acc",
                "train_candidate_top1_acc",
                "valid_candidate_top1_acc",
                "train_mrr",
                "valid_mrr",
                "train_positive_rank",
                "valid_positive_rank",
            ],
            csv_dir / "infonce_accuracy.csv",
        )
        write_rows(
            rows,
            [
                "epoch",
                "train_positive_similarity",
                "train_hardest_negative_similarity",
                "train_similarity_gap",
                "valid_positive_similarity",
                "valid_hardest_negative_similarity",
                "valid_similarity_gap",
            ],
            csv_dir / "infonce_similarity.csv",
        )
    else:
        write_rows(
            rows,
            ["epoch", "train_optimization_triplet_acc", "train_triplet_acc", "valid_triplet_acc", "test_triplet_acc"],
            csv_dir / "triplet_accuracy.csv",
        )
        write_rows(
            rows,
            [
                "epoch",
                "train_pos_dist",
                "train_hard_neg_dist",
                "train_margin_gap",
                "valid_pos_dist",
                "valid_hard_neg_dist",
                "valid_margin_gap",
                "test_pos_dist",
                "test_hard_neg_dist",
                "test_margin_gap",
            ],
            csv_dir / "triplet_distance.csv",
        )
        if loss_type == "semi_hard_triplet":
            write_rows(
                rows,
                [
                    "epoch",
                    "train_optimization_semi_hard_fraction",
                    "train_semi_hard_fraction",
                    "valid_semi_hard_fraction",
                    "train_fallback_fraction",
                    "valid_fallback_fraction",
                    "train_active_triplet_fraction",
                    "valid_active_triplet_fraction",
                ],
                csv_dir / "semi_hard_mining.csv",
            )
        if loss_type in {"random_triplet", "distance_weighted_triplet"}:
            write_rows(
                rows,
                [
                    "epoch",
                    "train_optimization_hard_sample_fraction",
                    "train_optimization_semi_hard_sample_fraction",
                    "train_optimization_easy_sample_fraction",
                    "train_optimization_active_triplet_fraction",
                    "train_optimization_selected_negative_rank",
                    "train_optimization_sampling_entropy",
                    "valid_hard_sample_fraction",
                    "valid_semi_hard_sample_fraction",
                    "valid_easy_sample_fraction",
                    "valid_active_triplet_fraction",
                    "valid_selected_negative_rank",
                    "valid_sampling_entropy",
                    "valid_hard_negative_count",
                    "valid_eligible_candidate_fraction",
                ],
                csv_dir / "negative_sampling.csv",
            )
    write_rows(
        rows,
        [
            "epoch",
            "assembly_train_pa",
            "assembly_valid_pa",
            "assembly_test_pa",
            "assembly_train_aa",
            "assembly_valid_aa",
            "assembly_test_aa",
        ],
        csv_dir / "assembly_accuracy.csv",
    )
    plot_lines(
        history,
        curves_dir,
        "loss.png",
        f"Puzzle-Level {loss_title(loss_type)} Loss",
        [("train_optimization", "loss", "train_optimization"), ("train", "loss", "train"), ("valid", "loss", "valid"), ("test", "loss", "test")],
    )
    if loss_type == "bidirectional_assignment":
        plot_lines(
            history,
            curves_dir,
            "assignment_top1_acc.png",
            "Puzzle Bidirectional Assignment Accuracy",
            [
                ("train", "row_top1_acc", "train_row"),
                ("valid", "row_top1_acc", "valid_row"),
                ("train", "column_top1_acc", "train_column"),
                ("valid", "column_top1_acc", "valid_column"),
                ("train", "mutual_top1_acc", "train_mutual"),
                ("valid", "mutual_top1_acc", "valid_mutual"),
            ],
        )
        plot_lines(
            history,
            curves_dir,
            "assignment_components.png",
            "Bidirectional Assignment Loss Components",
            [
                ("train", "hard_triplet_loss", "train_hard_triplet"),
                ("valid", "hard_triplet_loss", "valid_hard_triplet"),
                ("train", "assignment_loss", "train_assignment"),
                ("valid", "assignment_loss", "valid_assignment"),
            ],
        )
        plot_lines(
            history,
            curves_dir,
            "assignment_mrr.png",
            "Bidirectional Assignment Mean Reciprocal Rank",
            [("train", "mrr", "train"), ("valid", "mrr", "valid")],
        )
    elif loss_type == "infonce":
        plot_lines(
            history,
            curves_dir,
            "candidate_top1_acc.png",
            "Sampled-Candidate Top-1 Accuracy",
            [("train", "candidate_top1_acc", "train"), ("valid", "candidate_top1_acc", "valid")],
        )
        plot_lines(
            history,
            curves_dir,
            "similarity_gap.png",
            "Positive vs. Hardest-Negative Cosine Similarity Gap",
            [("train", "similarity_gap", "train"), ("valid", "similarity_gap", "valid")],
        )
        plot_lines(
            history,
            curves_dir,
            "mrr.png",
            "Sampled-Candidate Mean Reciprocal Rank",
            [("train", "mrr", "train"), ("valid", "mrr", "valid")],
        )
    else:
        plot_lines(
            history,
            curves_dir,
            "triplet_acc.png",
            f"Puzzle-Level {loss_title(loss_type)} Candidate Top-1 Accuracy",
            [("train", "triplet_acc", "train"), ("valid", "triplet_acc", "valid"), ("test", "triplet_acc", "test")],
        )
        plot_lines(
            history,
            curves_dir,
            "margin_gap.png",
            f"Puzzle-Level {loss_title(loss_type)} Hardest-Negative Margin Gap",
            [("train", "margin_gap", "train"), ("valid", "margin_gap", "valid"), ("test", "margin_gap", "test")],
        )
        if loss_type == "semi_hard_triplet":
            plot_lines(
                history,
                curves_dir,
                "semi_hard_mining.png",
                "Semi-Hard Mining Fractions",
                [
                    ("train", "semi_hard_fraction", "train_semi_hard"),
                    ("valid", "semi_hard_fraction", "valid_semi_hard"),
                    ("train", "fallback_fraction", "train_fallback"),
                    ("valid", "fallback_fraction", "valid_fallback"),
                    ("train", "active_triplet_fraction", "train_active"),
                    ("valid", "active_triplet_fraction", "valid_active"),
                ],
            )
        if loss_type in {"random_triplet", "distance_weighted_triplet"}:
            plot_lines(
                history,
                curves_dir,
                "negative_sampling.png",
                f"{loss_title(loss_type)} Sampling Fractions",
                [
                    ("train", "hard_sample_fraction", "train_hard"),
                    ("train", "semi_hard_sample_fraction", "train_semi_hard"),
                    ("train", "easy_sample_fraction", "train_easy"),
                    ("valid", "hard_sample_fraction", "valid_hard"),
                    ("valid", "semi_hard_sample_fraction", "valid_semi_hard"),
                    ("valid", "easy_sample_fraction", "valid_easy"),
                ],
            )
            plot_lines(
                history,
                curves_dir,
                "selected_negative_rank.png",
                f"{loss_title(loss_type)} Selected Negative Rank",
                [
                    ("train", "selected_negative_rank", "train"),
                    ("valid", "selected_negative_rank", "valid"),
                ],
            )
    plot_lines(
        history,
        curves_dir,
        "embedding_std.png",
        "Embedding Standard Deviation (Collapse Monitor)",
        [("train", "embedding_std", "train"), ("valid", "embedding_std", "valid"), ("test", "embedding_std", "test")],
    )
    plot_lines(
        history,
        curves_dir,
        "learning_rate.png",
        "Learning Rate",
        [("", "backbone_learning_rate", "backbone"), ("", "head_learning_rate", "projection_head")],
    )
    plot_assembly(history, curves_dir, method_title=loss_title(loss_type))


def _lookup(record: dict, section: str, key: str):
    if section:
        value = record.get(section, {})
        return value.get(key) if isinstance(value, dict) else None
    return record.get(key)


def plot_lines(history: list[dict], curves_dir: Path, filename: str, title: str, lines: list[tuple[str, str, str]]) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in history]
    plt.figure(figsize=(9, 5.2))
    plotted = False
    for section, key, label in lines:
        values = [_lookup(record, section, key) for record in history]
        if all(is_missing(value) for value in values):
            continue
        plt.plot(epochs, values, marker="o", linewidth=1.8, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(title)
    plt.xlabel("Epoch")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    curves_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(curves_dir / filename, dpi=200)
    plt.close()


def plot_assembly(history: list[dict], curves_dir: Path, method_title: str = "Hard Triplet") -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = []
    series = {f"{split}_{metric}": [] for split in EVAL_SPLITS for metric in ("pa", "aa")}
    for record in history:
        assembly = record.get("assembly", {})
        if not isinstance(assembly, dict):
            continue
        epochs.append(record["epoch"])
        for split in EVAL_SPLITS:
            metrics = assembly.get(split, {})
            for metric in ("pa", "aa"):
                series[f"{split}_{metric}"].append(metrics.get(metric))
    if not epochs:
        return
    plt.figure(figsize=(9.5, 5.4))
    plotted = False
    for name, values in series.items():
        if all(is_missing(value) for value in values):
            continue
        plt.plot(epochs, values, marker="o", linewidth=1.6, label=name)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(f"Puzzle-Level {method_title} Assembly PA/AA")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(ncol=2)
    plt.tight_layout()
    curves_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(curves_dir / "assembly_pa_aa.png", dpi=200)
    plt.close()


def write_summary(args, history: list[dict], total_time: float, path: Path) -> None:
    dataset_name = getattr(args, "dataset", "ImageNet-LSEJ" if hasattr(args, "task") else "unknown")
    loss_type = loss_type_from_args(args)
    selection_metric = selection_metric_for_loss(loss_type)
    lines = [
        f"Puzzle-Level {loss_title(loss_type)} Training Summary",
        "=" * 80,
        f"dataset: {dataset_name}",
        f"output_dir: {args.output_dir}",
        f"total_time: {format_seconds(total_time)}",
        "",
    ]
    if hasattr(args, "task"):
        lines.insert(3, f"task: {args.task}")
    add_best_block(
        lines,
        f"Best valid {selection_metric}",
        best_record(history, "valid", selection_metric, "max"),
        "valid",
    )
    add_best_block(lines, "Best valid loss", best_record(history, "valid", "loss", "min"), "valid")
    add_best_assembly_block(lines, "Best valid assembly AA", best_assembly_record(history, "valid", "aa", "max"), "valid")
    add_best_assembly_block(lines, "Best test assembly AA", best_assembly_record(history, "test", "aa", "max"), "test")
    lines.extend(["Arguments", "-" * 80])
    for key, value in sorted(args_to_dict(args).items()):
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            *runtime_environment_lines(
                requested_device=getattr(args, "device", None),
                gpu_id=getattr(args, "gpu_id", None),
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_record(history: list[dict], split: str, key: str, mode: str):
    records = []
    for record in history:
        metrics = record.get(split)
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(key)
        if is_missing(value):
            continue
        records.append(record)
    if not records:
        return None
    reverse = mode == "max"
    return sorted(records, key=lambda record: record[split][key], reverse=reverse)[0]


def best_assembly_record(history: list[dict], split: str, key: str, mode: str):
    records = []
    for record in history:
        assembly = record.get("assembly", {})
        metrics = assembly.get(split) if isinstance(assembly, dict) else None
        if isinstance(metrics, dict) and not is_missing(metrics.get(key)):
            records.append(record)
    if not records:
        return None
    reverse = mode == "max"
    return sorted(records, key=lambda record: record["assembly"][split][key], reverse=reverse)[0]


def add_best_block(lines: list[str], title: str, record: dict | None, split: str) -> None:
    lines.append(title)
    if record is None:
        lines.extend(["not available", ""])
        return
    lines.append(f"epoch: {record['epoch']}")
    for key in METRIC_KEYS:
        if key in record[split]:
            lines.append(f"{split}_{key}: {format_value(record[split][key])}")
    test = record.get("test", {})
    if isinstance(test, dict):
        for key in METRIC_KEYS:
            if key in test:
                lines.append(f"test_{key}: {format_value(test[key])}")
    lines.append("")


def add_best_assembly_block(lines: list[str], title: str, record: dict | None, split: str) -> None:
    lines.append(title)
    if record is None:
        lines.extend(["not available", ""])
        return
    metrics = record["assembly"][split]
    lines.append(f"epoch: {record['epoch']}")
    lines.append(f"{split}_PA: {metrics['pa']:.6f}")
    lines.append(f"{split}_AA: {metrics['aa']:.6f}")
    lines.append(f"{split}_perfect_puzzles: {metrics['perfect_puzzles']}/{metrics['samples']}")
    lines.append(f"{split}_correct_pieces: {metrics['correct_pieces']}/{metrics['total_pieces']}")
    test = record["assembly"].get("test", {})
    if isinstance(test, dict):
        lines.append(f"test_PA: {test['pa']:.6f}")
        lines.append(f"test_AA: {test['aa']:.6f}")
        lines.append(f"test_perfect_puzzles: {test['perfect_puzzles']}/{test['samples']}")
        lines.append(f"test_correct_pieces: {test['correct_pieces']}/{test['total_pieces']}")
    lines.append("")
