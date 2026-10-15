"""
@file: partial_layout_completion.py
@description: Complete incomplete layouts produced by Gallagher/LP and other solvers without ground-truth labels using frozen E1 scores.
              Treat placed pieces as a rigid scaffold, enumerate legal global translations, then use a profiled beam
              to fill all unplaced pieces and deterministically select the final candidate by the full-layout E1 objective.
@author: Changxin Ye
@created: 2026-08-05
@version: 1.0
"""

from __future__ import annotations

import numpy as np

from evaluation.lsej import compute_relationship_counts
from rg_lns.beam_search import seeded_profiled_beam_fill
from rg_lns.component_search import e1_layout_selection_key, legal_component_translations


def is_complete_layout(positions: np.ndarray, grid: int) -> bool:
    positions = np.asarray(positions)
    expected = np.arange(grid * grid)
    return positions.shape == expected.shape and np.array_equal(np.sort(positions), expected)


def validate_partial_layout(positions: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Validate a partial piece-to-position permutation and return the int32 layout and placed piece ids."""

    positions = np.asarray(positions, dtype=np.int32)
    count = grid * grid
    if positions.shape != (count,):
        raise ValueError(f"partial layout must have shape {(count,)}, got {positions.shape}")
    if np.any((positions < -1) | (positions >= count)):
        raise ValueError("partial layout positions must be -1 or in 0..N-1")
    placed_pieces = np.flatnonzero(positions >= 0).astype(np.int32)
    placed_slots = positions[placed_pieces]
    if len(np.unique(placed_slots)) != len(placed_slots):
        raise ValueError("partial layout contains duplicate grid positions and cannot be used as a rigid scaffold")
    if len(placed_pieces) == 0:
        raise ValueError("partial layout contains no placed pieces; E1 scaffold completion cannot start")
    return positions, placed_pieces


def partial_layout_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    grid: int,
) -> dict[str, float]:
    """Compute PA/AA/SRA for a layout containing -1; unplaced pieces count as errors."""

    prediction = np.asarray(prediction, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)
    count = grid * grid
    if prediction.shape != (count,) or target.shape != (count,):
        raise ValueError("prediction and target lengths must equal grid²")
    correct = int(np.count_nonzero(prediction == target))
    relationships = compute_relationship_counts(prediction, target, grid)
    relation_correct = relationships["horizontal_correct"] + relationships["vertical_correct"]
    relation_total = relationships["horizontal_total"] + relationships["vertical_total"]
    return {
        "PA": float(correct == count),
        "AA": correct / count,
        "SRA": relation_correct / relation_total if relation_total else 0.0,
    }


def complete_partial_layout(
    scores: np.ndarray,
    positions: np.ndarray,
    grid: int,
    *,
    completion_beam_width: int = 2,
) -> dict:
    """Complete a partial permutation; candidate generation and selection never read the ground-truth layout."""

    if completion_beam_width <= 0:
        raise ValueError("completion_beam_width must be positive")
    positions = np.asarray(positions, dtype=np.int32)
    count = grid * grid
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (count, count, 4):
        raise ValueError(f"E1 scores must have shape {(count, count, 4)}, got {scores.shape}")
    if is_complete_layout(positions, grid):
        return {
            "prediction": positions.copy(),
            "status": "already_complete",
            "applied": False,
            "placed_pieces": count,
            "missing_pieces": 0,
            "legal_translation_count": 1,
            "candidate_count": 1,
            "selected_row_shift": 0,
            "selected_column_shift": 0,
            "selected_completion_origin": "current",
            "selected_completion_index": -1,
            "candidates": [],
        }

    positions, component = validate_partial_layout(positions, grid)
    translations = legal_component_translations(component, positions, grid)
    candidates: list[dict] = []
    seen_predictions: set[bytes] = set()
    for row_shift, column_shift in translations:
        result = seeded_profiled_beam_fill(
            component,
            positions,
            scores,
            grid,
            row_shift,
            column_shift,
            beam_width=completion_beam_width,
        )
        for completion in result["completions"]:
            prediction = np.asarray(completion["prediction"], dtype=np.int32)
            key = prediction.tobytes()
            if key in seen_predictions:
                continue
            seen_predictions.add(key)
            candidates.append(
                {
                    "label": (
                        f"partial_shift_{row_shift}_{column_shift}_"
                        f"{completion['origin']}_{completion['completion_index']}"
                    ),
                    "row_shift": int(row_shift),
                    "column_shift": int(column_shift),
                    "completion_origin": str(completion["origin"]),
                    "completion_index": int(completion["completion_index"]),
                    "prediction": prediction,
                    "mean_adjacency_score": float(completion["mean_adjacency_score"]),
                    "mutual_top1_edges": int(completion["mutual_top1_edges"]),
                }
            )
    if not candidates:
        raise RuntimeError("partial-layout completion produced no complete candidate")

    def selection_key(candidate: dict):
        return (
            *e1_layout_selection_key(candidate),
            int(candidate["completion_index"]),
            str(candidate["completion_origin"]),
        )

    selected = min(candidates, key=selection_key)
    return {
        "prediction": selected["prediction"].copy(),
        "status": "partial_layout_completed",
        "applied": True,
        "placed_pieces": int(len(component)),
        "missing_pieces": int(count - len(component)),
        "legal_translation_count": len(translations),
        "candidate_count": len(candidates),
        "selected_row_shift": int(selected["row_shift"]),
        "selected_column_shift": int(selected["column_shift"]),
        "selected_completion_origin": str(selected["completion_origin"]),
        "selected_completion_index": int(selected["completion_index"]),
        "selected_mean_adjacency_score": float(selected["mean_adjacency_score"]),
        "selected_mutual_top1_edges": int(selected["mutual_top1_edges"]),
        "candidates": candidates,
    }


__all__ = [
    "complete_partial_layout",
    "is_complete_layout",
    "partial_layout_metrics",
    "validate_partial_layout",
]
