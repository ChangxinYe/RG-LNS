"""Shared ImageNet-LSEJ metrics; importing this module never starts an evaluation."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from evaluation.lsej import compute_relationship_counts


def _float_metric(value: float) -> str:
    return f"{float(value):.10g}"


def _ground_truth_metrics(prediction: np.ndarray, target: np.ndarray, grid: int) -> dict[str, float | int]:
    relationships = compute_relationship_counts(prediction, target, grid)
    correct_pieces = int(np.count_nonzero(prediction == target))
    horizontal_correct = int(relationships["horizontal_correct"])
    horizontal_total = int(relationships["horizontal_total"])
    vertical_correct = int(relationships["vertical_correct"])
    vertical_total = int(relationships["vertical_total"])
    return {
        "correct_pieces": correct_pieces,
        "total_pieces": grid * grid,
        "PA": int(correct_pieces == grid * grid),
        "AA": correct_pieces / (grid * grid),
        "horizontal_correct": horizontal_correct,
        "horizontal_total": horizontal_total,
        "horizontal_SRA": horizontal_correct / horizontal_total,
        "vertical_correct": vertical_correct,
        "vertical_total": vertical_total,
        "vertical_SRA": vertical_correct / vertical_total,
        "SRA": (horizontal_correct + vertical_correct) / (horizontal_total + vertical_total),
    }


def _aggregate_metrics(rows: list[dict]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list")
    samples = len(rows)
    horizontal_correct = sum(int(row["horizontal_correct"]) for row in rows)
    horizontal_total = sum(int(row["horizontal_total"]) for row in rows)
    vertical_correct = sum(int(row["vertical_correct"]) for row in rows)
    vertical_total = sum(int(row["vertical_total"]) for row in rows)
    return {
        "samples": samples,
        "perfect_puzzles": sum(int(row["PA"]) for row in rows),
        "correct_pieces": sum(int(row["correct_pieces"]) for row in rows),
        "total_pieces": sum(int(row["total_pieces"]) for row in rows),
        "PA": sum(int(row["PA"]) for row in rows) / samples,
        "AA": sum(float(row["AA"]) for row in rows) / samples,
        "horizontal_SRA": horizontal_correct / horizontal_total,
        "vertical_SRA": vertical_correct / vertical_total,
        "SRA": (horizontal_correct + vertical_correct) / (horizontal_total + vertical_total),
    }


def _write_pa_case_lists(output_dir: Path, rows: list[dict]) -> tuple[list[str], list[str]]:
    repaired = [
        f"sample_{int(row['sample_index']):06d}_{row['lsej_id']}.png"
        for row in rows
        if int(row["baseline_PA"]) == 0 and int(row["selected_PA"]) == 1
    ]
    broken = [
        f"sample_{int(row['sample_index']):06d}_{row['lsej_id']}.png"
        for row in rows
        if int(row["baseline_PA"]) == 1 and int(row["selected_PA"]) == 0
    ]
    for filename, values in (("pa_repaired_images.txt", repaired), ("pa_broken_images.txt", broken)):
        text = "\n".join(values)
        (output_dir / filename).write_text(text + ("\n" if text else ""), encoding="utf-8")
    return repaired, broken
