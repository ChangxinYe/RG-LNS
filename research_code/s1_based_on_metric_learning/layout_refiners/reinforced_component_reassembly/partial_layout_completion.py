"""
@file: partial_layout_completion.py
@description: 使用冻结 E1 对 Gallagher/LP 等求解器产生的不完整布局进行无真实标签补全。
              已放置 piece 作为刚性 scaffold，枚举其合法整体平移，再以 profiled beam
              回填所有未放置 piece，并按完整布局 E1 目标确定性选择最终候选。
@author: Changxin Ye
@created: 2026-08-05
@version: 1.0
"""

from __future__ import annotations

import numpy as np

try:
    from ...assembly_solvers.our_iterative_component_reassembly.profiled_completion_e1 import (
        seeded_profiled_beam_fill,
    )
    from ...assembly_solvers.our_iterative_component_reassembly.single_component_e1 import (
        e1_layout_selection_key,
        legal_component_translations,
    )
    from ...metric_lsej_evaluator import compute_relationship_counts
except ImportError:
    from assembly_solvers.our_iterative_component_reassembly.profiled_completion_e1 import (
        seeded_profiled_beam_fill,
    )
    from assembly_solvers.our_iterative_component_reassembly.single_component_e1 import (
        e1_layout_selection_key,
        legal_component_translations,
    )
    from metric_lsej_evaluator import compute_relationship_counts


def is_complete_layout(positions: np.ndarray, grid: int) -> bool:
    positions = np.asarray(positions)
    expected = np.arange(grid * grid)
    return positions.shape == expected.shape and np.array_equal(np.sort(positions), expected)


def validate_partial_layout(positions: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """验证 piece-to-position 部分排列，返回 int32 布局和已放置 piece id。"""

    positions = np.asarray(positions, dtype=np.int32)
    count = grid * grid
    if positions.shape != (count,):
        raise ValueError(f"部分布局应为 {(count,)}，实际为 {positions.shape}")
    if np.any((positions < -1) | (positions >= count)):
        raise ValueError("部分布局的位置只能是 -1 或 0..N-1")
    placed_pieces = np.flatnonzero(positions >= 0).astype(np.int32)
    placed_slots = positions[placed_pieces]
    if len(np.unique(placed_slots)) != len(placed_slots):
        raise ValueError("部分布局包含重复占用的网格位置，无法作为刚性 scaffold")
    if len(placed_pieces) == 0:
        raise ValueError("部分布局没有任何已放置 piece，E1 scaffold 补全无法启动")
    return positions, placed_pieces


def partial_layout_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    grid: int,
) -> dict[str, float]:
    """对含 -1 的布局计算 PA/AA/SRA；未放置 piece 自然计为错误。"""

    prediction = np.asarray(prediction, dtype=np.int32)
    target = np.asarray(target, dtype=np.int32)
    count = grid * grid
    if prediction.shape != (count,) or target.shape != (count,):
        raise ValueError("prediction 与 target 的长度必须等于 grid²")
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
    """把部分排列补为完整排列；候选生成与选择全过程不读取真实布局。"""

    if completion_beam_width <= 0:
        raise ValueError("completion_beam_width 必须为正")
    positions = np.asarray(positions, dtype=np.int32)
    count = grid * grid
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (count, count, 4):
        raise ValueError(f"E1 scores 应为 {(count, count, 4)}，实际为 {scores.shape}")
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
        raise RuntimeError("部分布局补全没有产生任何完整候选")

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
