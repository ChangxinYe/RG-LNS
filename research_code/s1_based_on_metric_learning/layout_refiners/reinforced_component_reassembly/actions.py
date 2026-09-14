"""
@file: actions.py
@description: 为 S8A 构造不依赖真实排列的空间状态，以及“可靠组件刚性平移+E1 回填”宏动作。
@author: Changxin Ye
@created: 2026-08-04
@version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ...assembly_solvers.our_iterative_component_reassembly.single_component_e1 import (
        cycle_supported_components,
        fine_layout_score,
        legal_component_translations,
    )
    from ...assembly_solvers.our_iterative_component_reassembly.profiled_completion_e1 import seeded_profiled_beam_fill
    from ...metric_geometry import BOTTOM, LEFT, RIGHT, TOP
    from ...metric_lsej_macro_assisted import candidate_rank, positions_to_layout
except ImportError:
    from assembly_solvers.our_iterative_component_reassembly.single_component_e1 import (
        cycle_supported_components,
        fine_layout_score,
        legal_component_translations,
    )
    from assembly_solvers.our_iterative_component_reassembly.profiled_completion_e1 import seeded_profiled_beam_fill
    from metric_geometry import BOTTOM, LEFT, RIGHT, TOP
    from metric_lsej_macro_assisted import candidate_rank, positions_to_layout


STATE_CHANNELS = 16
ACTION_FEATURE_DIM = 16
_NEIGHBORS = ((-1, 0, TOP, BOTTOM), (0, 1, RIGHT, LEFT), (1, 0, BOTTOM, TOP), (0, -1, LEFT, RIGHT))


@dataclass
class ReassemblyAction:
    """一个候选宏动作；prediction 是执行动作并完成回填后的完整布局。"""

    label: str
    prediction: np.ndarray
    features: np.ndarray
    component: np.ndarray
    row_shift: int = 0
    column_shift: int = 0
    is_stop: bool = False
    mean_e1: float = 0.0
    mutual_top1_edges: int = 0
    completion_origin: str = "current"


def _validate(scores: np.ndarray, prediction: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.int32)
    count = grid * grid
    if scores.shape != (count, count, 4):
        raise ValueError(f"scores 应为 {(count, count, 4)}，实际为 {scores.shape}")
    if prediction.shape != (count,) or not np.array_equal(np.sort(prediction), np.arange(count)):
        raise ValueError("prediction 必须是完整的 piece-to-position 排列")
    return scores, prediction


def _score_scale(scores: np.ndarray) -> float:
    finite = scores[np.isfinite(scores)]
    return float(np.std(finite)) if finite.size and float(np.std(finite)) > 1e-8 else 1.0


def build_spatial_state(
    scores: np.ndarray,
    prediction: np.ndarray,
    grid: int,
    *,
    mutual_top_k: int,
) -> np.ndarray:
    """把当前布局变成 H×W×C 状态图；所有通道都只依赖 E1 和当前布局。"""

    scores, prediction = _validate(scores, prediction, grid)
    layout = positions_to_layout(prediction, grid)
    state = np.zeros((grid, grid, STATE_CHANNELS), dtype=np.float32)
    scale = _score_scale(scores)
    denominator = max(grid - 1, 1)
    for row in range(grid):
        for column in range(grid):
            piece = int(layout[row, column])
            supported = 0
            for index, (dr, dc, direction, reverse) in enumerate(_NEIGHBORS):
                nr, nc = row + dr, column + dc
                if not (0 <= nr < grid and 0 <= nc < grid):
                    continue
                neighbor = int(layout[nr, nc])
                forward_rank = candidate_rank(scores, piece, neighbor, direction)
                reverse_rank = candidate_rank(scores, neighbor, piece, reverse)
                # 距离越小越好，tanh 将不同 checkpoint 的数值尺度压到稳定范围。
                state[row, column, index] = float(np.tanh(scores[piece, neighbor, direction] / scale))
                state[row, column, index + 4] = 1.0 / max(forward_rank, 1)
                mutual = forward_rank <= mutual_top_k and reverse_rank <= mutual_top_k
                state[row, column, index + 8] = float(mutual)
                supported += int(mutual)
            state[row, column, 12] = row / denominator
            state[row, column, 13] = column / denominator
            state[row, column, 14] = supported / 4.0
            state[row, column, 15] = 1.0
    return state


def _component_statistics(
    scores: np.ndarray,
    prediction: np.ndarray,
    component: np.ndarray,
    grid: int,
    mutual_top_k: int,
) -> tuple[float, float, float, float]:
    layout = positions_to_layout(prediction, grid)
    members = {int(piece) for piece in component}
    internal_scores: list[float] = []
    internal_mutual = 0
    boundary_scores: list[float] = []
    for piece in component:
        row, column = divmod(int(prediction[int(piece)]), grid)
        for dr, dc, direction, reverse in _NEIGHBORS:
            nr, nc = row + dr, column + dc
            if not (0 <= nr < grid and 0 <= nc < grid):
                continue
            neighbor = int(layout[nr, nc])
            value = float(scores[int(piece), neighbor, direction])
            if neighbor in members:
                internal_scores.append(value)
                internal_mutual += int(
                    candidate_rank(scores, int(piece), neighbor, direction) <= mutual_top_k
                    and candidate_rank(scores, neighbor, int(piece), reverse) <= mutual_top_k
                )
            else:
                boundary_scores.append(value)
    internal_mean = float(np.mean(internal_scores)) if internal_scores else 0.0
    internal_ratio = internal_mutual / len(internal_scores) if internal_scores else 0.0
    boundary_mean = float(np.mean(boundary_scores)) if boundary_scores else 0.0
    rows = [divmod(int(prediction[int(piece)]), grid)[0] for piece in component]
    columns = [divmod(int(prediction[int(piece)]), grid)[1] for piece in component]
    compactness = len(component) / ((max(rows) - min(rows) + 1) * (max(columns) - min(columns) + 1))
    return internal_mean, internal_ratio, boundary_mean, float(compactness)


def _translation_proxy(
    scores: np.ndarray,
    prediction: np.ndarray,
    component: np.ndarray,
    grid: int,
    shift: tuple[int, int],
) -> tuple[float, int, int, int]:
    """用目标边界上当前 piece 的 E1 代价预筛平移，避免每轮回填全部合法位置。"""

    row_shift, column_shift = shift
    original_layout = positions_to_layout(prediction, grid)
    component_set = {int(piece) for piece in component}
    moved = {}
    for piece in component:
        row, column = divmod(int(prediction[int(piece)]), grid)
        moved[(row + row_shift, column + column_shift)] = int(piece)
    costs = []
    for (row, column), piece in moved.items():
        for dr, dc, direction, _ in _NEIGHBORS:
            nr, nc = row + dr, column + dc
            if not (0 <= nr < grid and 0 <= nc < grid) or (nr, nc) in moved:
                continue
            neighbor = int(original_layout[nr, nc])
            if neighbor not in component_set:
                costs.append(float(scores[piece, neighbor, direction]))
    proxy = float(np.mean(costs)) if costs else float("inf")
    return proxy, abs(row_shift) + abs(column_shift), row_shift, column_shift


def build_action_set(
    scores: np.ndarray,
    prediction: np.ndarray,
    grid: int,
    *,
    mutual_top_k: int = 3,
    min_component_size: int = 4,
    max_components: int = 2,
    translations_per_component: int = 6,
    completion_beam_width: int = 2,
) -> tuple[np.ndarray, list[ReassemblyAction], dict[str, int]]:
    """构造当前状态和可变长动作集合；动作 0 永远是 STOP。"""

    scores, prediction = _validate(scores, prediction, grid)
    if completion_beam_width <= 0:
        raise ValueError("completion_beam_width 必须为正")
    state = build_spatial_state(scores, prediction, grid, mutual_top_k=mutual_top_k)
    current_score = fine_layout_score(scores, prediction, grid)
    scale = _score_scale(scores)
    edge_total = max(2 * grid * (grid - 1), 1)
    stop_features = np.zeros(ACTION_FEATURE_DIM, dtype=np.float32)
    stop_features[0] = 1.0
    stop_features[12] = float(current_score["mean_adjacency_score"] / scale)
    stop_features[13] = float(current_score["mutual_top1_edges"] / edge_total)
    actions = [
        ReassemblyAction(
            label="stop",
            prediction=prediction.copy(),
            features=stop_features,
            component=np.empty(0, dtype=np.int32),
            is_stop=True,
            mean_e1=float(current_score["mean_adjacency_score"]),
            mutual_top1_edges=int(current_score["mutual_top1_edges"]),
        )
    ]
    components, graph_stats = cycle_supported_components(
        scores, prediction, grid, mutual_top_k=mutual_top_k
    )
    components = [group for group in components if len(group) >= min_component_size][:max_components]
    seen = {prediction.tobytes()}
    count = grid * grid
    for component_index, component in enumerate(components):
        statistics = _component_statistics(
            scores, prediction, component, grid, mutual_top_k
        )
        translations = legal_component_translations(component, prediction, grid)
        translations.sort(
            key=lambda shift: _translation_proxy(scores, prediction, component, grid, shift)
        )
        # 0,0 对应“保留核心但重新回填外围”，它本身也是有意义的宏动作。
        selected = translations[:translations_per_component]
        if (0, 0) in translations and (0, 0) not in selected:
            selected = selected[:-1] + [(0, 0)] if selected else [(0, 0)]
        for row_shift, column_shift in selected:
            profiled = seeded_profiled_beam_fill(
                component,
                prediction,
                scores,
                grid,
                row_shift,
                column_shift,
                beam_width=completion_beam_width,
            )
            for completion in profiled["completions"]:
                candidate = np.asarray(completion["prediction"], dtype=np.int32)
                key = candidate.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidate_score = fine_layout_score(scores, candidate, grid)
                changed = float(np.mean(candidate != prediction))
                internal_mean, internal_ratio, boundary_mean, compactness = statistics
                rows = [divmod(int(prediction[int(piece)]), grid)[0] for piece in component]
                columns = [divmod(int(prediction[int(piece)]), grid)[1] for piece in component]
                completion_index = int(completion["completion_index"])
                completion_origin = str(completion["origin"])
                features = np.asarray(
                    [
                        0.0,
                        len(component) / count,
                        internal_mean / scale,
                        internal_ratio,
                        boundary_mean / scale,
                        compactness,
                        row_shift / max(grid - 1, 1),
                        column_shift / max(grid - 1, 1),
                        (abs(row_shift) + abs(column_shift)) / max(2 * (grid - 1), 1),
                        changed,
                        (max(rows) - min(rows) + 1) / grid,
                        (max(columns) - min(columns) + 1) / grid,
                        float(candidate_score["mean_adjacency_score"] / scale),
                        float(candidate_score["mutual_top1_edges"] / edge_total),
                        float(
                            (current_score["mean_adjacency_score"] - candidate_score["mean_adjacency_score"])
                            / scale
                        ),
                        completion_index / max(completion_beam_width - 1, 1),
                    ],
                    dtype=np.float32,
                )
                suffix = "greedy" if completion_origin == "s1a3_greedy" else f"beam{completion_index}"
                actions.append(
                    ReassemblyAction(
                        label=f"component_{component_index}_shift_{row_shift}_{column_shift}_{suffix}",
                        prediction=candidate,
                        features=features,
                        component=component.copy(),
                        row_shift=int(row_shift),
                        column_shift=int(column_shift),
                        mean_e1=float(candidate_score["mean_adjacency_score"]),
                        mutual_top1_edges=int(candidate_score["mutual_top1_edges"]),
                        completion_origin=completion_origin,
                    )
                )
    return state, actions, graph_stats


__all__ = [
    "ACTION_FEATURE_DIM",
    "STATE_CHANNELS",
    "ReassemblyAction",
    "build_action_set",
    "build_spatial_state",
]
