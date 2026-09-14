"""
@file: segmenter.py
@description: 根据当前布局中的 Best-Buddies 邻接边提取 Pomeranz 高置信组件。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

import numpy as np

try:
    from ...metric_geometry import BOTTOM, LEFT, RIGHT, TOP
except ImportError:
    from metric_geometry import BOTTOM, LEFT, RIGHT, TOP

from .best_buddies import prediction_to_layout


def _are_best_buddies(
    piece: int,
    neighbor: int,
    direction: int,
    best_neighbors: np.ndarray,
    mutual_best_buddies: np.ndarray,
) -> bool:
    return bool(
        mutual_best_buddies[int(piece), int(direction)]
        and int(best_neighbors[int(piece), int(direction)]) == int(neighbor)
    )


def best_buddy_components(
    prediction: np.ndarray,
    best_neighbors: np.ndarray,
    mutual_best_buddies: np.ndarray,
    grid: int,
    *,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """使用 Best-Buddies 谓词执行论文描述的随机种子 region growing。"""

    layout = prediction_to_layout(prediction, grid)
    num_pieces = grid * grid
    rng = rng or np.random.default_rng(0)
    labels = np.full((grid, grid), -1, dtype=np.int32)
    components: list[np.ndarray] = []

    neighbor_steps = (
        (-1, 0, TOP),
        (0, 1, RIGHT),
        (1, 0, BOTTOM),
        (0, -1, LEFT),
    )
    while np.any(labels < 0):
        unassigned = np.argwhere(labels < 0)
        seed_row, seed_column = unassigned[int(rng.integers(len(unassigned)))]
        segment_index = len(components)
        labels[int(seed_row), int(seed_column)] = segment_index
        stack = [(int(seed_row), int(seed_column))]
        component: list[int] = []
        while stack:
            row, column = stack.pop()
            piece = int(layout[row, column])
            component.append(piece)
            for row_step, column_step, _ in neighbor_steps:
                candidate_row = row + row_step
                candidate_column = column + column_step
                if not (0 <= candidate_row < grid and 0 <= candidate_column < grid):
                    continue
                if labels[candidate_row, candidate_column] >= 0:
                    continue

                candidate_piece = int(layout[candidate_row, candidate_column])
                compatible_with_segment = True
                for check_row_step, check_column_step, direction in neighbor_steps:
                    check_row = candidate_row + check_row_step
                    check_column = candidate_column + check_column_step
                    if not (0 <= check_row < grid and 0 <= check_column < grid):
                        continue
                    if labels[check_row, check_column] != segment_index:
                        continue
                    neighbor_piece = int(layout[check_row, check_column])
                    if not _are_best_buddies(
                        candidate_piece,
                        neighbor_piece,
                        direction,
                        best_neighbors,
                        mutual_best_buddies,
                    ):
                        compatible_with_segment = False
                        break
                if compatible_with_segment:
                    labels[candidate_row, candidate_column] = segment_index
                    stack.append((candidate_row, candidate_column))
        components.append(np.asarray(sorted(component), dtype=np.int32))
    components.sort(key=lambda component: (-len(component), int(component[0])))
    return components


__all__ = ["best_buddy_components"]
