"""
@file: shifter.py
@description: 将 Pomeranz 最大 Best-Buddies 组件转换为下一轮 Placer 的刚性种子。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def component_seed_positions(
    prediction: np.ndarray,
    component: np.ndarray,
    grid: int,
) -> Mapping[int, tuple[int, int]]:
    """保留组件内部相对位置；整体平移由浮动画布 Placer 隐式决定。"""

    prediction = np.asarray(prediction, dtype=np.int64)
    component = np.asarray(component, dtype=np.int64)
    if component.ndim != 1 or len(component) == 0:
        raise ValueError("component 必须是一维非空数组")
    coordinates = {
        int(piece): divmod(int(prediction[int(piece)]), grid)
        for piece in component
    }
    min_row = min(row for row, _ in coordinates.values())
    min_column = min(column for _, column in coordinates.values())
    return {
        piece: (row - min_row, column - min_column)
        for piece, (row, column) in coordinates.items()
    }


__all__ = ["component_seed_positions"]
