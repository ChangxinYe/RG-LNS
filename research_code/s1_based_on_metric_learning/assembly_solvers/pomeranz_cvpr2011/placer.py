"""
@file: placer.py
@description: 使用 S1A E1 距离实现 Pomeranz 的浮动画布贪心 Placer。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

try:
    from ...metric_geometry import BOTTOM, LEFT, RIGHT, TOP
except ImportError:
    from metric_geometry import BOTTOM, LEFT, RIGHT, TOP


_NEIGHBOR_STEPS = (
    (-1, 0, TOP),
    (0, 1, RIGHT),
    (1, 0, BOTTOM),
    (0, -1, LEFT),
)


def _bounding_box_fits(
    occupied: Mapping[tuple[int, int], int],
    candidate: tuple[int, int],
    grid: int,
) -> bool:
    rows = [row for row, _ in occupied]
    columns = [column for _, column in occupied]
    rows.append(int(candidate[0]))
    columns.append(int(candidate[1]))
    return max(rows) - min(rows) + 1 <= grid and max(columns) - min(columns) + 1 <= grid


def _candidate_slots(
    occupied: Mapping[tuple[int, int], int],
    grid: int,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    proposed: set[tuple[int, int]] = set()
    for row, column in occupied:
        for row_step, column_step, _ in _NEIGHBOR_STEPS:
            coordinate = (row + row_step, column + column_step)
            if coordinate not in occupied and _bounding_box_fits(occupied, coordinate, grid):
                proposed.add(coordinate)

    candidates = []
    for row, column in proposed:
        neighbors = []
        for row_step, column_step, direction in _NEIGHBOR_STEPS:
            coordinate = (row + row_step, column + column_step)
            if coordinate in occupied:
                neighbors.append((int(occupied[coordinate]), int(direction)))
        if neighbors:
            candidates.append((row, column, neighbors))
    if not candidates:
        return []
    maximum_neighbors = max(len(item[2]) for item in candidates)
    return sorted(
        (item for item in candidates if len(item[2]) == maximum_neighbors),
        key=lambda item: (item[0], item[1]),
    )


def _select_piece_and_slot(
    candidates: list[tuple[int, int, list[tuple[int, int]]]],
    remaining: set[int],
    scores: np.ndarray,
    best_neighbors: np.ndarray,
    mutual_best_buddies: np.ndarray,
) -> tuple[int, int, int]:
    remaining_ids = np.asarray(sorted(remaining), dtype=np.int64)
    unique_buddy_matches: list[tuple[int, int, int]] = []
    cost_choices: list[tuple[float, int, int, int]] = []

    for row, column, neighbors in candidates:
        neighbor_ids = np.asarray([piece for piece, _ in neighbors], dtype=np.int64)
        directions = np.asarray([direction for _, direction in neighbors], dtype=np.int64)
        costs = scores[remaining_ids[:, None], neighbor_ids[None, :], directions[None, :]]
        mean_costs = np.mean(costs, axis=1)

        buddy_matches = np.ones(len(remaining_ids), dtype=bool)
        for neighbor, direction in neighbors:
            buddy_matches &= mutual_best_buddies[remaining_ids, direction]
            buddy_matches &= best_neighbors[remaining_ids, direction] == int(neighbor)
        for piece in remaining_ids[buddy_matches]:
            unique_buddy_matches.append((int(piece), int(row), int(column)))

        best_index = min(
            range(len(remaining_ids)),
            key=lambda index: (float(mean_costs[index]), int(remaining_ids[index])),
        )
        cost_choices.append(
            (
                float(mean_costs[best_index]),
                int(remaining_ids[best_index]),
                int(row),
                int(column),
            )
        )

    if len(unique_buddy_matches) == 1:
        piece, row, column = unique_buddy_matches[0]
        return piece, row, column

    _, piece, row, column = min(cost_choices)
    return piece, row, column


def _positions_to_prediction(occupied: Mapping[tuple[int, int], int], grid: int) -> np.ndarray:
    num_pieces = grid * grid
    if len(occupied) != num_pieces:
        raise RuntimeError(f"Placer 只放置了 {len(occupied)}/{num_pieces} 个 piece")
    rows = [row for row, _ in occupied]
    columns = [column for _, column in occupied]
    min_row, max_row = min(rows), max(rows)
    min_column, max_column = min(columns), max(columns)
    if max_row - min_row + 1 != grid or max_column - min_column + 1 != grid:
        raise RuntimeError("完整布局的包围盒不是目标网格大小")

    prediction = np.full(num_pieces, -1, dtype=np.int32)
    for (row, column), piece in occupied.items():
        position = (row - min_row) * grid + (column - min_column)
        if prediction[int(piece)] >= 0:
            raise RuntimeError("同一个 piece 被重复放置")
        prediction[int(piece)] = int(position)
    if not np.array_equal(np.sort(prediction), np.arange(num_pieces)):
        raise RuntimeError("Placer 输出不是完整排列")
    return prediction


def greedy_place(
    scores: np.ndarray,
    best_neighbors: np.ndarray,
    mutual_best_buddies: np.ndarray,
    grid: int,
    seed_positions: Mapping[int, tuple[int, int]],
) -> np.ndarray:
    """从单 piece 或刚性组件种子开始，在浮动画布上放置全部剩余 piece。"""

    num_pieces = grid * grid
    if not seed_positions:
        raise ValueError("seed_positions 不能为空")
    occupied: dict[tuple[int, int], int] = {}
    for piece, coordinate in seed_positions.items():
        piece = int(piece)
        coordinate = (int(coordinate[0]), int(coordinate[1]))
        if not 0 <= piece < num_pieces:
            raise ValueError(f"种子 piece 编号越界：{piece}")
        if coordinate in occupied:
            raise ValueError("种子组件中存在坐标冲突")
        occupied[coordinate] = piece
    if len(set(occupied.values())) != len(occupied):
        raise ValueError("种子组件中存在重复 piece")

    remaining = set(range(num_pieces)) - set(occupied.values())
    while remaining:
        candidates = _candidate_slots(occupied, grid)
        if not candidates:
            raise RuntimeError("Pomeranz Placer 找不到可行空位")
        piece, row, column = _select_piece_and_slot(
            candidates,
            remaining,
            scores,
            best_neighbors,
            mutual_best_buddies,
        )
        occupied[(row, column)] = piece
        remaining.remove(piece)
    return _positions_to_prediction(occupied, grid)


__all__ = ["greedy_place"]
