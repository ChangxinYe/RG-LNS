"""
@file: shifter.py
@description: Convert the largest Pomeranz Best-Buddies component into a rigid seed for the next Placer round.
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
    """Preserve internal component coordinates; global translation is handled implicitly by the floating-canvas Placer."""

    prediction = np.asarray(prediction, dtype=np.int64)
    component = np.asarray(component, dtype=np.int64)
    if component.ndim != 1 or len(component) == 0:
        raise ValueError("component must be a non-empty one-dimensional array")
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
