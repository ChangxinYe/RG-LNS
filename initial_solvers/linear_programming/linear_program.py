"""
@file: linear_program.py
@description: Faithful port of the sparse weighted L1 linear program and rigid-component constraints from lp_from_patch_and_weight_sp.m.
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .matching import MatchInfo


# MATLAB coordinates use x=row and y=column; configurations 1..4 are up, right, down, and left.
DIRECTION_OFFSETS = np.asarray(
    [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]],
    dtype=np.float64,
)


def find_best_fix_piece(info: MatchInfo, num_pieces: int) -> int:
    """Reproduce ``find_best_match.m`` by selecting the most reliable piece with non-unit matches in all four directions."""

    counts = np.zeros(num_pieces, dtype=np.int32)
    sums = np.zeros(num_pieces, dtype=np.float64)
    for direction in range(4):
        candidates = np.flatnonzero(
            (info.directions == direction) & (info.weights != 1.0)
        )
        seen: set[int] = set()
        for index in candidates:
            anchor = int(info.anchors[index])
            if anchor in seen:
                continue
            seen.add(anchor)
            counts[anchor] += 1
            sums[anchor] += float(info.weights[index])
    sums[counts != 4] = 0.0
    return int(np.argmax(sums))


def solve_weighted_l1_lp(
    num_pieces: int,
    info: MatchInfo,
    *,
    fix_piece: int,
    fix_value: float = -100000.0,
    previous_labels: np.ndarray | None = None,
    previous_coordinates: np.ndarray | None = None,
    rigid_components: bool = False,
    time_limit: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Solve the author's equality-form LP; rigid components preserve their internal shape through shared translation variables."""

    if len(info) == 0:
        raise RuntimeError("LP candidate set is empty")
    m = len(info)
    labels = None if previous_labels is None else np.asarray(previous_labels, dtype=np.int32)
    coordinates = (
        None if previous_coordinates is None else np.asarray(previous_coordinates, dtype=np.float64)
    )
    component_ids: list[int] = []
    if rigid_components:
        if labels is None or coordinates is None:
            raise ValueError("rigid-component constraints require previous_labels and previous_coordinates")
        component_ids = [int(x) for x in np.unique(labels) if int(x) >= 0]

    coordinate_variables = 2 * num_pieces
    auxiliary_variables = 4 * m
    translation_variables = 2 * len(component_ids)
    variable_count = coordinate_variables + auxiliary_variables + translation_variables

    objective = np.zeros(variable_count, dtype=np.float64)
    # MATLAB order: [horizontal observation weights; vertical observation weights], then g and h respectively.
    observation_weights = np.concatenate([info.weights, info.weights])
    objective[coordinate_variables : coordinate_variables + 2 * m] = observation_weights
    objective[coordinate_variables + 2 * m : coordinate_variables + 4 * m] = observation_weights

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs: list[float] = []

    def add_equality(entries: list[tuple[int, float]], target: float) -> None:
        row = len(rhs)
        for column, value in entries:
            rows.append(row)
            columns.append(column)
            values.append(value)
        rhs.append(float(target))

    # Exactly matches A1, A2, and B in lp_from_patch_and_weight_sp.m.
    for observation, (anchor, neighbor, direction) in enumerate(
        zip(info.anchors, info.neighbors, info.directions)
    ):
        anchor = int(anchor)
        neighbor = int(neighbor)
        direction = int(direction)
        primary_axis = 1 if direction in (1, 3) else 0
        secondary_axis = 1 - primary_axis
        sign = -1.0 if direction in (1, 2) else 1.0
        primary_row = observation
        secondary_row = m + observation
        g_primary = coordinate_variables + primary_row
        g_secondary = coordinate_variables + secondary_row
        h_primary = coordinate_variables + 2 * m + primary_row
        h_secondary = coordinate_variables + 2 * m + secondary_row
        add_equality(
            [
                (primary_axis * num_pieces + anchor, sign),
                (primary_axis * num_pieces + neighbor, -sign),
                (g_primary, -1.0),
                (h_primary, 1.0),
            ],
            1.0,
        )
        add_equality(
            [
                (secondary_axis * num_pieces + anchor, sign),
                (secondary_axis * num_pieces + neighbor, -sign),
                (g_secondary, -1.0),
                (h_secondary, 1.0),
            ],
            0.0,
        )

    # connected_comp(+fix_point): each component has only two shared translation degrees of freedom.
    translation_start = coordinate_variables + auxiliary_variables
    for component_index, component_id in enumerate(component_ids):
        members = np.flatnonzero(labels == component_id)
        for piece in members:
            for axis in range(2):
                translation = translation_start + 2 * component_index + axis
                add_equality(
                    [(axis * num_pieces + int(piece), 1.0), (translation, 1.0)],
                    float(coordinates[int(piece), axis]),
                )

    add_equality([(fix_piece, 1.0)], fix_value)
    add_equality([(num_pieces + fix_piece, 1.0)], fix_value)

    a_eq = sparse.coo_matrix(
        (values, (rows, columns)), shape=(len(rhs), variable_count), dtype=np.float64
    ).tocsr()
    bounds = (
        [(None, None)] * coordinate_variables
        + [(0.0, None)] * auxiliary_variables
        + [(None, None)] * translation_variables
    )
    options: dict[str, float] = {}
    if time_limit > 0:
        options["time_limit"] = float(time_limit)
    result = linprog(
        objective,
        A_eq=a_eq,
        b_eq=np.asarray(rhs, dtype=np.float64),
        bounds=bounds,
        method="highs",
        options=options,
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"HiGHS LP failed: status={result.status}, message={result.message}"
        )
    solved = np.column_stack(
        [result.x[:num_pieces], result.x[num_pieces : 2 * num_pieces]]
    )
    return solved, {
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "iterations": int(getattr(result, "nit", 0)),
        "candidates": m,
        "rigid_components": len(component_ids),
    }


__all__ = ["DIRECTION_OFFSETS", "find_best_fix_piece", "solve_weighted_l1_lp"]
