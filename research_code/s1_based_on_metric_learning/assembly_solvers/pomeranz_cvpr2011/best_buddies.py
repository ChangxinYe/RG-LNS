"""
@file: best_buddies.py
@description: 从 S1A E1 方向距离构造 Pomeranz Best-Buddies 关系及无真值布局评分。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

import numpy as np

try:
    from ...metric_geometry import BOTTOM, LEFT, OPPOSITE, RIGHT, TOP
except ImportError:
    from metric_geometry import BOTTOM, LEFT, OPPOSITE, RIGHT, TOP


def validate_scores(scores: np.ndarray, num_pieces: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    expected = (num_pieces, num_pieces, 4)
    if scores.shape != expected:
        raise ValueError(f"E1 分数形状应为 {expected}，实际为 {scores.shape}")
    if np.any(np.isnan(scores)) or np.any(np.isneginf(scores)):
        raise ValueError("E1 分数中不能包含 NaN 或负无穷值")
    candidates = scores.copy()
    piece_ids = np.arange(num_pieces)
    candidates[piece_ids, piece_ids, :] = np.inf
    if np.any(~np.any(np.isfinite(candidates), axis=1)):
        raise ValueError("至少有一个 piece/方向不存在有限的非自身候选")
    return scores


def compute_best_neighbors(scores: np.ndarray) -> np.ndarray:
    """返回 ``best_neighbors[piece, direction]``，距离越小排名越靠前。"""

    num_pieces = int(np.asarray(scores).shape[0])
    scores = validate_scores(scores, num_pieces).copy()
    piece_ids = np.arange(num_pieces)
    scores[piece_ids, piece_ids, :] = np.inf
    return np.argmin(scores, axis=1).astype(np.int32, copy=False)


def compute_mutual_best_buddies(best_neighbors: np.ndarray) -> np.ndarray:
    """返回有方向的 mutual Top-1 布尔矩阵，形状为 ``[N, 4]``。"""

    best_neighbors = np.asarray(best_neighbors, dtype=np.int64)
    if best_neighbors.ndim != 2 or best_neighbors.shape[1] != 4:
        raise ValueError("best_neighbors 应为 [N, 4]")
    num_pieces = best_neighbors.shape[0]
    if np.any(best_neighbors < 0) or np.any(best_neighbors >= num_pieces):
        raise ValueError("best_neighbors 中存在越界 piece 编号")

    piece_ids = np.arange(num_pieces, dtype=np.int64)[:, None]
    directions = np.arange(4, dtype=np.int64)[None, :]
    candidates = best_neighbors
    reverse = best_neighbors[candidates, OPPOSITE[directions]]
    return reverse == piece_ids


def prediction_to_layout(prediction: np.ndarray, grid: int) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.int64)
    num_pieces = grid * grid
    if prediction.shape != (num_pieces,):
        raise ValueError(f"prediction 应为 ({num_pieces},)，实际为 {prediction.shape}")
    if not np.array_equal(np.sort(prediction), np.arange(num_pieces)):
        raise ValueError("prediction 必须是完整的 piece-to-position 排列")
    layout = np.empty((grid, grid), dtype=np.int32)
    for piece, position in enumerate(prediction):
        layout.flat[int(position)] = int(piece)
    return layout


def best_buddies_layout_score(
    prediction: np.ndarray,
    best_neighbors: np.ndarray,
    mutual_best_buddies: np.ndarray,
    grid: int,
) -> dict[str, float | int]:
    """仅统计当前布局真实相邻边上的 Best-Buddies 比例。"""

    layout = prediction_to_layout(prediction, grid)
    best_neighbors = np.asarray(best_neighbors, dtype=np.int64)
    mutual_best_buddies = np.asarray(mutual_best_buddies, dtype=bool)
    buddy_edges = 0
    total_edges = 2 * grid * (grid - 1)
    for row in range(grid):
        for column in range(grid):
            piece = int(layout[row, column])
            if column + 1 < grid:
                candidate = int(layout[row, column + 1])
                buddy_edges += int(
                    mutual_best_buddies[piece, RIGHT]
                    and int(best_neighbors[piece, RIGHT]) == candidate
                )
            if row + 1 < grid:
                candidate = int(layout[row + 1, column])
                buddy_edges += int(
                    mutual_best_buddies[piece, BOTTOM]
                    and int(best_neighbors[piece, BOTTOM]) == candidate
                )
    return {
        "best_buddy_edges": int(buddy_edges),
        "total_edges": int(total_edges),
        "best_buddies_metric": float(buddy_edges / total_edges),
    }


__all__ = [
    "best_buddies_layout_score",
    "compute_best_neighbors",
    "compute_mutual_best_buddies",
    "prediction_to_layout",
    "validate_scores",
]
