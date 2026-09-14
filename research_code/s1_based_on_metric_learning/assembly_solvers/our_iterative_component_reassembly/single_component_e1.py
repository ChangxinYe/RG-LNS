"""
@file: single_component_e1.py
@description: 我们的单组件 E1 迭代重组算法；适用于任意 Type-1 方形拼图数据集。
@author: Changxin Ye
@created: 2026-07-26
@version: 1.1
"""

from __future__ import annotations

import numpy as np

try:
    from ...metric_geometry import BOTTOM, LEFT, RIGHT, TOP
    from ...metric_lsej_macro_assisted import (
        candidate_rank,
        iter_macro_grid_relations,
        macro_relation_rank_penalty,
        positions_to_layout,
    )
except ImportError:
    from metric_geometry import BOTTOM, LEFT, RIGHT, TOP
    from metric_lsej_macro_assisted import (
        candidate_rank,
        iter_macro_grid_relations,
        macro_relation_rank_penalty,
        positions_to_layout,
    )


def _edge_key(first: int, second: int) -> tuple[int, int]:
    return tuple(sorted((int(first), int(second))))


def reliable_components(
    scores: np.ndarray,
    prediction: np.ndarray,
    grid: int,
    mutual_top_k: int = 3,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Build reliable components from cycle-supported reciprocal Top-K edges.

    A layout edge is retained only when both directed ranks are within Top-K
    and the edge participates in at least one complete 2x2 cycle.  The cycle
    filter suppresses isolated bridge matches before connected components are
    extracted.
    """

    if mutual_top_k <= 0:
        raise ValueError("mutual_top_k must be positive")
    layout = positions_to_layout(prediction, grid)
    mutual_edges: set[tuple[int, int]] = set()

    for row in range(grid):
        for column in range(grid):
            anchor = int(layout[row, column])
            if column + 1 < grid:
                candidate = int(layout[row, column + 1])
                if (
                    candidate_rank(scores, anchor, candidate, RIGHT) <= mutual_top_k
                    and candidate_rank(scores, candidate, anchor, LEFT) <= mutual_top_k
                ):
                    mutual_edges.add(_edge_key(anchor, candidate))
            if row + 1 < grid:
                candidate = int(layout[row + 1, column])
                if (
                    candidate_rank(scores, anchor, candidate, BOTTOM) <= mutual_top_k
                    and candidate_rank(scores, candidate, anchor, TOP) <= mutual_top_k
                ):
                    mutual_edges.add(_edge_key(anchor, candidate))

    supported_edges: set[tuple[int, int]] = set()
    cycle_count = 0
    for row in range(grid - 1):
        for column in range(grid - 1):
            top_left = int(layout[row, column])
            top_right = int(layout[row, column + 1])
            bottom_left = int(layout[row + 1, column])
            bottom_right = int(layout[row + 1, column + 1])
            perimeter = {
                _edge_key(top_left, top_right),
                _edge_key(top_left, bottom_left),
                _edge_key(top_right, bottom_right),
                _edge_key(bottom_left, bottom_right),
            }
            if perimeter <= mutual_edges:
                cycle_count += 1
                supported_edges.update(perimeter)

    num_pieces = grid * grid
    parents = list(range(num_pieces))

    def find(piece: int) -> int:
        while parents[piece] != piece:
            parents[piece] = parents[parents[piece]]
            piece = parents[piece]
        return piece

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first, second in supported_edges:
        union(first, second)

    grouped: dict[int, list[int]] = {}
    for piece in range(num_pieces):
        grouped.setdefault(find(piece), []).append(piece)
    components = [np.asarray(group, dtype=np.int32) for group in grouped.values()]
    components.sort(key=lambda group: (-len(group), int(group.min())))
    return components, {
        "mutual_edges": len(mutual_edges),
        "complete_cycles": cycle_count,
        "cycle_supported_edges": len(supported_edges),
        "components": len(components),
    }


def cycle_supported_components(
    scores: np.ndarray,
    prediction: np.ndarray,
    grid: int,
    mutual_top_k: int = 3,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Backward-compatible cycle-supported component extractor."""

    return reliable_components(
        scores,
        prediction,
        grid,
        mutual_top_k,
    )


def legal_component_translations(
    component: np.ndarray,
    prediction: np.ndarray,
    grid: int,
    *,
    translation_mode: str = "all",
) -> list[tuple[int, int]]:
    if translation_mode not in {"fixed", "all"}:
        raise ValueError("translation_mode must be 'fixed' or 'all'")
    component = np.asarray(component, dtype=np.int64)
    if component.ndim != 1 or len(component) == 0:
        raise ValueError("component must be a non-empty vector of piece indices")
    if translation_mode == "fixed":
        return [(0, 0)]
    coordinates = [divmod(int(prediction[piece]), grid) for piece in component]
    min_row = min(row for row, _ in coordinates)
    max_row = max(row for row, _ in coordinates)
    min_column = min(column for _, column in coordinates)
    max_column = max(column for _, column in coordinates)
    return [
        (row_shift, column_shift)
        for row_shift in range(-min_row, grid - max_row)
        for column_shift in range(-min_column, grid - max_column)
    ]


def seeded_greedy_fill(
    component: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
    grid: int,
    row_shift: int,
    column_shift: int,
) -> np.ndarray:
    """Translate a rigid component and greedily place all remaining pieces.

    Candidate slots with the most already placed neighbors are filled first.
    E1 compatibility to all occupied neighbors is averaged, and deterministic
    piece/row/column tie breaks make the diagnostic exactly reproducible.
    """

    component = np.asarray(component, dtype=np.int64)
    num_pieces = grid * grid
    if scores.shape != (num_pieces, num_pieces, 4):
        raise ValueError(f"Unexpected score shape: {scores.shape}")
    canvas = np.full((grid, grid), -1, dtype=np.int32)
    component_set = {int(piece) for piece in component}
    for piece in component_set:
        row, column = divmod(int(prediction[piece]), grid)
        target_row = row + int(row_shift)
        target_column = column + int(column_shift)
        if not (0 <= target_row < grid and 0 <= target_column < grid):
            raise ValueError("Component translation leaves the puzzle canvas")
        if canvas[target_row, target_column] >= 0:
            raise RuntimeError("Translated component contains a coordinate collision")
        canvas[target_row, target_column] = piece

    remaining = set(range(num_pieces)) - component_set
    while remaining:
        candidate_slots = []
        for row, column in zip(*np.where(canvas < 0)):
            neighbors = []
            for neighbor_row, neighbor_column, direction in (
                (row - 1, column, TOP),
                (row, column + 1, RIGHT),
                (row + 1, column, BOTTOM),
                (row, column - 1, LEFT),
            ):
                if (
                    0 <= neighbor_row < grid
                    and 0 <= neighbor_column < grid
                    and canvas[neighbor_row, neighbor_column] >= 0
                ):
                    neighbors.append((int(canvas[neighbor_row, neighbor_column]), direction))
            if neighbors:
                candidate_slots.append((len(neighbors), int(row), int(column), neighbors))
        if not candidate_slots:
            raise RuntimeError("No fillable slot remains around the translated component")

        maximum_neighbors = max(item[0] for item in candidate_slots)
        best = None
        for _, row, column, neighbors in candidate_slots:
            if len(neighbors) != maximum_neighbors:
                continue
            for piece in remaining:
                mean_cost = float(
                    np.mean([scores[piece, neighbor, direction] for neighbor, direction in neighbors])
                )
                key = (mean_cost, int(piece), row, column)
                if best is None or key < best[0]:
                    best = (key, int(piece), row, column)
        if best is None:
            raise RuntimeError("Could not choose a piece for a fillable slot")
        _, piece, row, column = best
        canvas[row, column] = piece
        remaining.remove(piece)

    result = np.empty(num_pieces, dtype=np.int32)
    result[canvas.ravel()] = np.arange(num_pieces, dtype=np.int32)
    return result


def fine_layout_score(
    scores: np.ndarray,
    prediction: np.ndarray,
    grid: int,
) -> dict[str, float | int]:
    layout = positions_to_layout(prediction, grid)
    adjacency_scores = []
    mutual_top1 = 0
    for row in range(grid):
        for column in range(grid):
            anchor = int(layout[row, column])
            if column + 1 < grid:
                candidate = int(layout[row, column + 1])
                adjacency_scores.append(float(scores[anchor, candidate, RIGHT]))
                mutual_top1 += int(
                    candidate_rank(scores, anchor, candidate, RIGHT) == 1
                    and candidate_rank(scores, candidate, anchor, LEFT) == 1
                )
            if row + 1 < grid:
                candidate = int(layout[row + 1, column])
                adjacency_scores.append(float(scores[anchor, candidate, BOTTOM]))
                mutual_top1 += int(
                    candidate_rank(scores, anchor, candidate, BOTTOM) == 1
                    and candidate_rank(scores, candidate, anchor, TOP) == 1
                )
    return {
        "mean_adjacency_score": float(np.mean(adjacency_scores)),
        "mutual_top1_edges": int(mutual_top1),
    }


def macro_layout_rank_score(macro_scores: np.ndarray, macro_grid: int) -> dict[str, float | int]:
    penalties = [
        macro_relation_rank_penalty(macro_scores, anchor, candidate, direction)[2]
        for anchor, candidate, direction in iter_macro_grid_relations(macro_grid)
    ]
    return {
        "mean_rank_penalty": float(np.mean(penalties)),
        "rank1_boundaries": int(sum(penalty == 0.0 for penalty in penalties)),
        "boundaries": len(penalties),
    }


def e1_layout_selection_key(candidate: dict) -> tuple[float, int, int, int, int]:
    """Return the deterministic, ground-truth-free E1 candidate order."""

    row_shift = int(candidate["row_shift"])
    column_shift = int(candidate["column_shift"])
    mean_score = candidate.get("e1_mean_adjacency_score", candidate.get("mean_adjacency_score"))
    mutual_top1 = candidate.get("e1_mutual_top1_edges", candidate.get("mutual_top1_edges"))
    if mean_score is None or mutual_top1 is None:
        raise KeyError("candidate is missing E1 layout statistics")
    return (
        float(mean_score),
        -int(mutual_top1),
        abs(row_shift) + abs(column_shift),
        row_shift,
        column_shift,
    )


def iterative_component_reassembly(
    scores: np.ndarray,
    initial_prediction: np.ndarray,
    grid: int,
    *,
    mutual_top_k: int = 3,
    component_index: int = 0,
    min_component_size: int = 4,
    max_rounds: int = 1,
    translation_mode: str = "all",
) -> dict:
    """Iteratively rebuild a layout around one reliable E1 component.

    Every round keeps the current layout as a candidate, enumerates all legal
    translations of the selected component (or only its current pose in
    ``fixed`` mode), greedily fills the remaining pieces using the frozen E1
    matrix, and accepts a new layout only when its E1 objective strictly
    improves. Ground-truth positions are never used.
    """

    if mutual_top_k <= 0:
        raise ValueError("mutual_top_k must be positive")
    if component_index < 0:
        raise ValueError("component_index must be non-negative")
    if min_component_size <= 0:
        raise ValueError("min_component_size must be positive")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if translation_mode not in {"fixed", "all"}:
        raise ValueError("translation_mode must be 'fixed' or 'all'")

    num_pieces = grid * grid
    current = np.asarray(initial_prediction, dtype=np.int32).copy()
    if current.shape != (num_pieces,):
        raise ValueError(f"Expected prediction shape {(num_pieces,)}, got {current.shape}")
    if sorted(int(position) for position in current) != list(range(num_pieces)):
        raise ValueError("initial_prediction must be a complete piece-to-position permutation")

    rounds: list[dict] = []
    accepted_rounds = 0
    last_accepted_component = np.empty(0, dtype=np.int32)
    last_accepted_seed_prediction = current.copy()
    seen_layouts = {current.tobytes()}
    stop_reason = "max_rounds"

    for round_index in range(1, max_rounds + 1):
        components, graph_statistics = reliable_components(
            scores,
            current,
            grid,
            mutual_top_k=mutual_top_k,
        )
        current_score = fine_layout_score(scores, current, grid)
        round_record = {
            "round": round_index,
            "seed_prediction": current.copy(),
            "component": np.empty(0, dtype=np.int32),
            "graph_statistics": graph_statistics,
            "translation_mode": translation_mode,
            "status": "evaluated",
            "accepted": False,
            "selected_label": "current",
            "selected_row_shift": 0,
            "selected_column_shift": 0,
            "candidates": [],
        }

        if component_index >= len(components):
            round_record["status"] = "missing_component"
            rounds.append(round_record)
            stop_reason = "missing_component"
            break

        component = components[component_index]
        round_record["component"] = component.copy()
        if len(component) < min_component_size:
            round_record["status"] = "small_component"
            rounds.append(round_record)
            stop_reason = "small_component"
            break

        current_candidate = {
            "label": "baseline" if round_index == 1 else "current",
            "row_shift": 0,
            "column_shift": 0,
            "prediction": current.copy(),
            **current_score,
        }
        candidates = [current_candidate]
        round_layouts = {current.tobytes()}
        for row_shift, column_shift in legal_component_translations(
            component,
            current,
            grid,
            translation_mode=translation_mode,
        ):
            prediction = seeded_greedy_fill(
                component,
                current,
                scores,
                grid,
                row_shift,
                column_shift,
            )
            layout_key = prediction.tobytes()
            if layout_key in round_layouts:
                continue
            round_layouts.add(layout_key)
            candidates.append(
                {
                    "label": f"component_shift_{row_shift}_{column_shift}",
                    "row_shift": int(row_shift),
                    "column_shift": int(column_shift),
                    "prediction": prediction,
                    **fine_layout_score(scores, prediction, grid),
                }
            )

        selected = min(candidates, key=e1_layout_selection_key)
        for candidate in candidates:
            candidate["selected"] = candidate is selected
        round_record["candidates"] = candidates
        round_record["selected_label"] = selected["label"]
        round_record["selected_row_shift"] = selected["row_shift"]
        round_record["selected_column_shift"] = selected["column_shift"]

        current_objective = (
            float(current_candidate["mean_adjacency_score"]),
            -int(current_candidate["mutual_top1_edges"]),
        )
        selected_objective = (
            float(selected["mean_adjacency_score"]),
            -int(selected["mutual_top1_edges"]),
        )
        if selected_objective >= current_objective:
            round_record["status"] = "no_strict_e1_improvement"
            rounds.append(round_record)
            stop_reason = "no_strict_e1_improvement"
            break

        selected_layout_key = selected["prediction"].tobytes()
        if selected_layout_key in seen_layouts:
            round_record["status"] = "repeated_layout"
            rounds.append(round_record)
            stop_reason = "repeated_layout"
            break

        round_record["accepted"] = True
        rounds.append(round_record)
        accepted_rounds += 1
        last_accepted_component = component.copy()
        last_accepted_seed_prediction = current.copy()
        current = selected["prediction"].copy()
        seen_layouts.add(selected_layout_key)

    final_score = fine_layout_score(scores, current, grid)
    return {
        "prediction": current,
        "initial_score": fine_layout_score(scores, initial_prediction, grid),
        "final_score": final_score,
        "rounds": rounds,
        "rounds_attempted": len(rounds),
        "rounds_accepted": accepted_rounds,
        "stop_reason": stop_reason,
        "translation_mode": translation_mode,
        "last_accepted_component": last_accepted_component,
        "last_accepted_seed_prediction": last_accepted_seed_prediction,
    }


__all__ = [
    "cycle_supported_components",
    "reliable_components",
    "fine_layout_score",
    "e1_layout_selection_key",
    "iterative_component_reassembly",
    "legal_component_translations",
    "macro_layout_rank_score",
    "seeded_greedy_fill",
]
