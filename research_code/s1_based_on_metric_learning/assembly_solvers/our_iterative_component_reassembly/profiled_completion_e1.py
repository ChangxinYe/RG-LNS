"""
@file: profiled_completion_e1.py
@description: 面向 S1A4 的 E1 profiled large-neighborhood reassembly 核心实现。
              算法保留 S1A3 的可靠组件提取、合法刚性平移与严格 E1 改善门控，
              但在每个组件位姿下使用确定性 beam completion 保留多条条件回填路径，
              再以完整布局的 E1 目标选择该位姿的代表解。beam_width=1 时直接复用
              S1A3 正式求解器，以保证历史结果、并列规则与停止条件严格一致。
@author: Changxin Ye
@created: 2026-07-31
@version: 1.0
"""

from __future__ import annotations

import heapq

import numpy as np

try:
    from ...metric_geometry import BOTTOM, LEFT, RIGHT, TOP
except ImportError:
    from metric_geometry import BOTTOM, LEFT, RIGHT, TOP

from .single_component_e1 import (
    e1_layout_selection_key,
    fine_layout_score,
    iterative_component_reassembly,
    legal_component_translations,
    reliable_components,
    seeded_greedy_fill,
)


def _validate_seed(
    component: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
    grid: int,
    row_shift: int,
    column_shift: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """构造刚性平移后的 seed canvas，并返回尚未放置的 piece。"""

    component = np.asarray(component, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int32)
    num_pieces = grid * grid
    if component.ndim != 1 or len(component) == 0:
        raise ValueError("component must be a non-empty vector of piece indices")
    if prediction.shape != (num_pieces,):
        raise ValueError(f"Expected prediction shape {(num_pieces,)}, got {prediction.shape}")
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

    remaining = tuple(piece for piece in range(num_pieces) if piece not in component_set)
    return canvas, remaining


def _canonical_new_edge_statistics(
    canvas: np.ndarray,
    scores: np.ndarray,
    row: int,
    column: int,
    piece: int,
) -> tuple[float, int]:
    """计算把 piece 放到指定位置后新闭合边的规范方向 E1 代价。

    `fine_layout_score` 始终按照 left->right 和 top->bottom 统计完整布局。
    这里采用完全相同的方向，因此 partial beam score 与最终 E1 目标一致；
    它不依赖 compatibility 是否做过 opposite-transpose 对称化。
    """

    grid = int(canvas.shape[0])
    edge_sum = 0.0
    edge_count = 0
    if row > 0 and canvas[row - 1, column] >= 0:
        edge_sum += float(scores[int(canvas[row - 1, column]), piece, BOTTOM])
        edge_count += 1
    if column + 1 < grid and canvas[row, column + 1] >= 0:
        edge_sum += float(scores[piece, int(canvas[row, column + 1]), RIGHT])
        edge_count += 1
    if row + 1 < grid and canvas[row + 1, column] >= 0:
        edge_sum += float(scores[piece, int(canvas[row + 1, column]), BOTTOM])
        edge_count += 1
    if column > 0 and canvas[row, column - 1] >= 0:
        edge_sum += float(scores[int(canvas[row, column - 1]), piece, RIGHT])
        edge_count += 1
    return edge_sum, edge_count


def _initial_closed_edge_statistics(
    canvas: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, int]:
    """统计 seed 组件内部已经闭合的 E1 网格边。"""

    grid = int(canvas.shape[0])
    edge_sum = 0.0
    edge_count = 0
    for row in range(grid):
        for column in range(grid):
            anchor = int(canvas[row, column])
            if anchor < 0:
                continue
            if column + 1 < grid and canvas[row, column + 1] >= 0:
                edge_sum += float(scores[anchor, int(canvas[row, column + 1]), RIGHT])
                edge_count += 1
            if row + 1 < grid and canvas[row + 1, column] >= 0:
                edge_sum += float(scores[anchor, int(canvas[row + 1, column]), BOTTOM])
                edge_count += 1
    return edge_sum, edge_count


def _candidate_fill_moves(
    canvas: np.ndarray,
    remaining: tuple[int, ...],
    scores: np.ndarray,
) -> list[tuple[float, int, int, int, float, int]]:
    """枚举当前 state 中所有合法的 `(piece, slot)` 动作。

    与 S1A3 一样，只考虑“已放置邻居数最多”的空槽。动作首先按新闭合边
    的规范 E1 均值排序，再使用 piece/row/column 做确定性并列裁决。
    返回项依次为：排序代价、piece、row、column、新边代价和、新边数量。
    """

    grid = int(canvas.shape[0])
    candidate_slots: list[tuple[int, int, int]] = []
    maximum_neighbors = 0
    for row, column in zip(*np.where(canvas < 0)):
        occupied_neighbors = 0
        for neighbor_row, neighbor_column in (
            (row - 1, column),
            (row, column + 1),
            (row + 1, column),
            (row, column - 1),
        ):
            if (
                0 <= neighbor_row < grid
                and 0 <= neighbor_column < grid
                and canvas[neighbor_row, neighbor_column] >= 0
            ):
                occupied_neighbors += 1
        if occupied_neighbors > 0:
            maximum_neighbors = max(maximum_neighbors, occupied_neighbors)
            candidate_slots.append((occupied_neighbors, int(row), int(column)))

    if maximum_neighbors == 0:
        raise RuntimeError("No fillable slot remains around the translated component")

    moves: list[tuple[float, int, int, int, float, int]] = []
    for occupied_neighbors, row, column in candidate_slots:
        if occupied_neighbors != maximum_neighbors:
            continue
        for piece in remaining:
            added_sum, added_count = _canonical_new_edge_statistics(
                canvas,
                scores,
                row,
                column,
                int(piece),
            )
            if added_count <= 0:
                continue
            mean_cost = added_sum / added_count
            moves.append(
                (mean_cost, int(piece), row, column, float(added_sum), int(added_count))
            )

    if not moves:
        raise RuntimeError("Could not generate a piece assignment for a fillable slot")
    return moves


def _canvas_to_prediction(canvas: np.ndarray) -> np.ndarray:
    """把 position->piece canvas 转换为 piece->position 完整排列。"""

    flat = np.asarray(canvas, dtype=np.int32).ravel()
    num_pieces = int(flat.size)
    if sorted(int(piece) for piece in flat) != list(range(num_pieces)):
        raise RuntimeError("Beam completion did not produce a complete piece permutation")
    prediction = np.empty(num_pieces, dtype=np.int32)
    prediction[flat] = np.arange(num_pieces, dtype=np.int32)
    return prediction


def _completion_selection_key(completion: dict) -> tuple[float, int, int]:
    return (
        float(completion["mean_adjacency_score"]),
        -int(completion["mutual_top1_edges"]),
        int(completion["completion_index"]),
    )


def seeded_profiled_beam_fill(
    component: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
    grid: int,
    row_shift: int,
    column_shift: int,
    *,
    beam_width: int,
) -> dict:
    """在固定组件位姿下生成多条条件完成，并返回完整 E1 最优解。

    beam_width 表示每个位姿最终保留的完成数量。第 0 条始终是原 S1A3
    `seeded_greedy_fill` 路径；其余路径由 canonical-E1 beam search 产生。
    因此扩大 beam 后的候选集合始终包含原贪心解，profiled E1 目标不会劣于
    同一位姿的 S1A3 单路径结果，但这并不等价于 PA 必然提升。
    """

    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    greedy_prediction = seeded_greedy_fill(
        component,
        prediction,
        scores,
        grid,
        row_shift,
        column_shift,
    )
    greedy_score = fine_layout_score(scores, greedy_prediction, grid)
    greedy_completion = {
        "completion_index": 0,
        "origin": "s1a3_greedy",
        "prediction": greedy_prediction,
        "beam_partial_mean_score": None,
        "beam_closed_edges": None,
        **greedy_score,
    }
    if beam_width == 1:
        greedy_completion["profile_rank"] = 1
        greedy_completion["selected_within_pose"] = True
        return {
            "prediction": greedy_prediction.copy(),
            "selected_completion_index": 0,
            "selected_completion_origin": "s1a3_greedy",
            "completions": [greedy_completion],
            "completion_count": 1,
            "beam_final_states": 0,
            "greedy_score": greedy_score,
            "profiled_score": greedy_score,
            "profile_mean_gain_vs_greedy": 0.0,
            "profile_mutual_top1_gain_vs_greedy": 0,
        }

    seed_canvas, remaining = _validate_seed(
        component,
        prediction,
        scores,
        grid,
        row_shift,
        column_shift,
    )
    initial_sum, initial_count = _initial_closed_edge_statistics(seed_canvas, scores)
    beam = [
        {
            "canvas": seed_canvas,
            "remaining": remaining,
            "closed_score_sum": float(initial_sum),
            "closed_edge_count": int(initial_count),
            "path_key": (),
        }
    ]

    # 每一层放置一个 piece；同一层的 state 已放置 piece 数量相同。
    # 先把所有 parent 的动作放入一个全局最小堆，再持续弹出动作，直到得到
    # B 个互不相同的 child canvas 或当前动作堆耗尽。不能先把每个 parent 截成 top-B 再去重，
    # 否则不同落子顺序汇合到同一 canvas 时会让 beam 静默缩水。
    for _ in range(len(remaining)):
        child_descriptors = []
        for state_index, state in enumerate(beam):
            moves = _candidate_fill_moves(
                state["canvas"],
                state["remaining"],
                scores,
            )
            for _, piece, row, column, added_sum, added_count in moves:
                child_sum = float(state["closed_score_sum"] + added_sum)
                child_count = int(state["closed_edge_count"] + added_count)
                child_descriptors.append(
                    (
                        child_sum / child_count,
                        -child_count,
                        state["path_key"],
                        piece,
                        row,
                        column,
                        state_index,
                        added_sum,
                        added_count,
                    )
                )

        heapq.heapify(child_descriptors)
        children_by_canvas: dict[bytes, dict] = {}
        while child_descriptors and len(children_by_canvas) < beam_width:
            (
                _,
                _,
                _,
                piece,
                row,
                column,
                state_index,
                added_sum,
                added_count,
            ) = heapq.heappop(child_descriptors)
            state = beam[state_index]
            child_canvas = state["canvas"].copy()
            child_canvas[row, column] = piece
            canvas_key = child_canvas.tobytes()
            if canvas_key in children_by_canvas:
                continue
            child_remaining = tuple(
                remaining_piece
                for remaining_piece in state["remaining"]
                if remaining_piece != piece
            )
            children_by_canvas[canvas_key] = {
                "canvas": child_canvas,
                "remaining": child_remaining,
                "closed_score_sum": float(state["closed_score_sum"] + added_sum),
                "closed_edge_count": int(state["closed_edge_count"] + added_count),
                "path_key": state["path_key"] + ((piece, row, column),),
            }

        if not children_by_canvas:
            raise RuntimeError("Beam completion produced no valid child state")
        beam = sorted(
            children_by_canvas.values(),
            key=lambda state: (
                state["closed_score_sum"] / state["closed_edge_count"],
                -state["closed_edge_count"],
                state["path_key"],
            ),
        )[:beam_width]

    beam_completions: list[dict] = []
    seen_predictions = {greedy_prediction.tobytes()}
    for state in beam:
        beam_prediction = _canvas_to_prediction(state["canvas"])
        prediction_key = beam_prediction.tobytes()
        if prediction_key in seen_predictions:
            continue
        seen_predictions.add(prediction_key)
        beam_completions.append(
            {
                "origin": "profiled_beam",
                "prediction": beam_prediction,
                "beam_partial_mean_score": (
                    float(state["closed_score_sum"]) / int(state["closed_edge_count"])
                ),
                "beam_closed_edges": int(state["closed_edge_count"]),
                "path_key": state["path_key"],
                **fine_layout_score(scores, beam_prediction, grid),
            }
        )

    beam_completions.sort(
        key=lambda completion: (
            float(completion["mean_adjacency_score"]),
            -int(completion["mutual_top1_edges"]),
            float(completion["beam_partial_mean_score"]),
            completion["path_key"],
        )
    )
    retained = [greedy_completion, *beam_completions[: beam_width - 1]]
    for completion_index, completion in enumerate(retained):
        completion["completion_index"] = completion_index
        completion.pop("path_key", None)

    ranked = sorted(retained, key=_completion_selection_key)
    for profile_rank, completion in enumerate(ranked, start=1):
        completion["profile_rank"] = profile_rank
        completion["selected_within_pose"] = profile_rank == 1
    selected = ranked[0]
    selected_score = {
        "mean_adjacency_score": float(selected["mean_adjacency_score"]),
        "mutual_top1_edges": int(selected["mutual_top1_edges"]),
    }
    return {
        "prediction": selected["prediction"].copy(),
        "selected_completion_index": int(selected["completion_index"]),
        "selected_completion_origin": str(selected["origin"]),
        "completions": retained,
        "completion_count": len(retained),
        "beam_final_states": len(beam),
        "greedy_score": greedy_score,
        "profiled_score": selected_score,
        "profile_mean_gain_vs_greedy": float(
            greedy_score["mean_adjacency_score"] - selected_score["mean_adjacency_score"]
        ),
        "profile_mutual_top1_gain_vs_greedy": int(
            selected_score["mutual_top1_edges"] - greedy_score["mutual_top1_edges"]
        ),
    }


def _decorate_single_path_result(
    result: dict,
    grid: int,
    *,
    translation_mode: str,
) -> dict:
    """为 S1A3 的严格等价结果补齐 S1A4 诊断字段。"""

    last_accepted_row_shift = 0
    last_accepted_column_shift = 0
    last_accepted_completion_index = -1
    last_accepted_completion_origin = "current"
    for round_record in result["rounds"]:
        poses = []
        pose_index = 0
        for candidate in round_record["candidates"]:
            is_current = candidate["label"] in {"baseline", "current"}
            candidate["completion_count"] = 0 if is_current else 1
            candidate["selected_completion_index"] = -1 if is_current else 0
            candidate["selected_completion_origin"] = "current" if is_current else "s1a3_greedy"
            candidate["profile_mean_gain_vs_greedy"] = 0.0
            candidate["profile_mutual_top1_gain_vs_greedy"] = 0
            candidate["greedy_mean_adjacency_score"] = candidate["mean_adjacency_score"]
            candidate["greedy_mutual_top1_edges"] = candidate["mutual_top1_edges"]
            candidate["completions"] = []
            if not is_current:
                candidate["pose_index"] = pose_index
                candidate["beam_final_states"] = 0
                candidate["deduplicated_from_global_candidates"] = False
                pose_index += 1
                completion = {
                    "completion_index": 0,
                    "origin": "s1a3_greedy",
                    "prediction": candidate["prediction"].copy(),
                    "beam_partial_mean_score": None,
                    "beam_closed_edges": None,
                    "mean_adjacency_score": candidate["mean_adjacency_score"],
                    "mutual_top1_edges": candidate["mutual_top1_edges"],
                    "profile_rank": 1,
                    "selected_within_pose": True,
                    "selected_in_round": bool(candidate["selected"]),
                    "accepted_global": bool(
                        candidate["selected"] and round_record["accepted"]
                    ),
                }
                candidate["completions"] = [completion]
                candidate["selected_pose"] = bool(candidate["selected"])
                candidate["accepted_pose"] = bool(
                    candidate["selected"] and round_record["accepted"]
                )
                poses.append(candidate)

        selected_candidate = next(
            (candidate for candidate in round_record["candidates"] if candidate["selected"]),
            None,
        )
        round_record["poses"] = poses
        evaluated_pose_count = 0
        if round_record["status"] not in {"missing_component", "small_component"}:
            evaluated_pose_count = len(
                legal_component_translations(
                    round_record["component"],
                    round_record["seed_prediction"],
                    grid,
                    translation_mode=translation_mode,
                )
            )
        # Legacy S1A3 在记录 candidates 前会按完整 prediction 去重，因此 poses
        # 可能少于实际评估的合法平移。两个口径都显式保存，避免 B=1/B>1 误比。
        round_record["pose_count"] = evaluated_pose_count
        round_record["recorded_pose_count"] = len(poses)
        round_record["completion_candidate_count"] = evaluated_pose_count
        round_record["recorded_completion_count"] = len(poses)
        round_record["pose_diagnostics_complete"] = len(poses) == evaluated_pose_count
        round_record["completion_beam_width"] = 1
        round_record["selected_completion_index"] = (
            -1 if selected_candidate is None else selected_candidate["selected_completion_index"]
        )
        round_record["selected_completion_origin"] = (
            "current" if selected_candidate is None else selected_candidate["selected_completion_origin"]
        )
        round_record["single_path_selected_label"] = (
            "current" if selected_candidate is None else selected_candidate["label"]
        )
        round_record["single_path_selected_row_shift"] = (
            0 if selected_candidate is None else int(selected_candidate["row_shift"])
        )
        round_record["single_path_selected_column_shift"] = (
            0 if selected_candidate is None else int(selected_candidate["column_shift"])
        )
        round_record["single_path_prediction"] = (
            round_record["seed_prediction"].copy()
            if selected_candidate is None
            else selected_candidate["prediction"].copy()
        )
        if round_record["accepted"] and selected_candidate is not None:
            last_accepted_row_shift = int(selected_candidate["row_shift"])
            last_accepted_column_shift = int(selected_candidate["column_shift"])
            last_accepted_completion_index = int(selected_candidate["selected_completion_index"])
            last_accepted_completion_origin = str(selected_candidate["selected_completion_origin"])

    result["completion_beam_width"] = 1
    result["last_accepted_row_shift"] = last_accepted_row_shift
    result["last_accepted_column_shift"] = last_accepted_column_shift
    result["last_accepted_completion_index"] = last_accepted_completion_index
    result["last_accepted_completion_origin"] = last_accepted_completion_origin
    result["single_path_reference_prediction"] = result["prediction"].copy()
    return result


def profiled_large_neighborhood_reassembly(
    scores: np.ndarray,
    initial_prediction: np.ndarray,
    grid: int,
    *,
    mutual_top_k: int = 3,
    component_index: int = 0,
    min_component_size: int = 4,
    max_rounds: int = 1,
    completion_beam_width: int = 4,
    translation_mode: str = "all",
) -> dict:
    """执行关系保持、位置释放和多路径条件完成的 S1A4 重组。

    每轮从当前完整布局提取由双向 Top-K 邻接与 2x2 环支持确定的可靠组件，
    再按 ``translation_mode`` 使用原位或枚举全部合法刚性平移。对每个位姿保留
    至多 B 条完整回填，并把其中真实 E1 最优者作为该位姿的 profiled
    representative。跨位姿仍使用 S1A3 的 E1 排序和严格改善门控。
    """

    if completion_beam_width <= 0:
        raise ValueError("completion_beam_width must be positive")
    if translation_mode not in {"fixed", "all"}:
        raise ValueError("translation_mode must be 'fixed' or 'all'")
    if completion_beam_width == 1:
        legacy_result = iterative_component_reassembly(
            scores,
            initial_prediction,
            grid,
            mutual_top_k=mutual_top_k,
            component_index=component_index,
            min_component_size=min_component_size,
            max_rounds=max_rounds,
            translation_mode=translation_mode,
        )
        return _decorate_single_path_result(
            legacy_result,
            grid,
            translation_mode=translation_mode,
        )

    if mutual_top_k <= 0:
        raise ValueError("mutual_top_k must be positive")
    if component_index < 0:
        raise ValueError("component_index must be non-negative")
    if min_component_size <= 0:
        raise ValueError("min_component_size must be positive")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")

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
    last_accepted_row_shift = 0
    last_accepted_column_shift = 0
    last_accepted_completion_index = -1
    last_accepted_completion_origin = "current"
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
            "selected_completion_index": -1,
            "selected_completion_origin": "current",
            "single_path_selected_label": "current",
            "single_path_selected_row_shift": 0,
            "single_path_selected_column_shift": 0,
            "single_path_prediction": current.copy(),
            "completion_beam_width": completion_beam_width,
            "pose_count": 0,
            "recorded_pose_count": 0,
            "completion_candidate_count": 0,
            "recorded_completion_count": 0,
            "pose_diagnostics_complete": True,
            "poses": [],
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
            "completion_count": 0,
            "selected_completion_index": -1,
            "selected_completion_origin": "current",
            "profile_mean_gain_vs_greedy": 0.0,
            "profile_mutual_top1_gain_vs_greedy": 0,
            "greedy_mean_adjacency_score": current_score["mean_adjacency_score"],
            "greedy_mutual_top1_edges": current_score["mutual_top1_edges"],
            "completions": [],
            **current_score,
        }
        candidates = [current_candidate]
        single_path_candidates = [current_candidate]
        single_path_layouts = {current.tobytes()}
        poses = []
        round_layouts = {current.tobytes()}
        for pose_index, (row_shift, column_shift) in enumerate(
            legal_component_translations(
                component,
                current,
                grid,
                translation_mode=translation_mode,
            )
        ):
            profiled = seeded_profiled_beam_fill(
                component,
                current,
                scores,
                grid,
                row_shift,
                column_shift,
                beam_width=completion_beam_width,
            )
            pose = {
                "pose_index": pose_index,
                "label": f"component_shift_{row_shift}_{column_shift}",
                "row_shift": int(row_shift),
                "column_shift": int(column_shift),
                "prediction": profiled["prediction"],
                "completion_count": int(profiled["completion_count"]),
                "beam_final_states": int(profiled["beam_final_states"]),
                "selected_completion_index": int(profiled["selected_completion_index"]),
                "selected_completion_origin": str(profiled["selected_completion_origin"]),
                "profile_mean_gain_vs_greedy": float(profiled["profile_mean_gain_vs_greedy"]),
                "profile_mutual_top1_gain_vs_greedy": int(
                    profiled["profile_mutual_top1_gain_vs_greedy"]
                ),
                "greedy_mean_adjacency_score": float(
                    profiled["greedy_score"]["mean_adjacency_score"]
                ),
                "greedy_mutual_top1_edges": int(
                    profiled["greedy_score"]["mutual_top1_edges"]
                ),
                "completions": profiled["completions"],
                "mean_adjacency_score": float(
                    profiled["profiled_score"]["mean_adjacency_score"]
                ),
                "mutual_top1_edges": int(profiled["profiled_score"]["mutual_top1_edges"]),
                "selected_pose": False,
                "accepted_pose": False,
                "deduplicated_from_global_candidates": False,
            }
            poses.append(pose)

            # completion_index=0 永远是原 S1A3 seeded_greedy_fill 路径。
            # 单独维护这一组候选，使默认单轮实验能够在同一次 E1 计算中得到
            # 严格可比的 S1A3 reference，而不必重复运行全部贪心回填。
            greedy_completion = pose["completions"][0]
            greedy_layout_key = greedy_completion["prediction"].tobytes()
            if greedy_layout_key not in single_path_layouts:
                single_path_layouts.add(greedy_layout_key)
                single_path_candidates.append(
                    {
                        "label": pose["label"],
                        "row_shift": pose["row_shift"],
                        "column_shift": pose["column_shift"],
                        "prediction": greedy_completion["prediction"],
                        "mean_adjacency_score": greedy_completion[
                            "mean_adjacency_score"
                        ],
                        "mutual_top1_edges": greedy_completion["mutual_top1_edges"],
                    }
                )

            layout_key = pose["prediction"].tobytes()
            if layout_key in round_layouts:
                pose["deduplicated_from_global_candidates"] = True
                continue
            round_layouts.add(layout_key)
            candidates.append(pose)

        selected = min(candidates, key=e1_layout_selection_key)
        single_path_selected = min(single_path_candidates, key=e1_layout_selection_key)
        for candidate in candidates:
            candidate["selected"] = candidate is selected
        if selected is not current_candidate:
            selected["selected_pose"] = True
        for pose in poses:
            for completion in pose["completions"]:
                completion["selected_in_round"] = bool(
                    pose is selected and completion["selected_within_pose"]
                )
                completion["accepted_global"] = False

        round_record["poses"] = poses
        round_record["pose_count"] = len(poses)
        round_record["recorded_pose_count"] = len(poses)
        round_record["completion_candidate_count"] = sum(
            len(pose["completions"]) for pose in poses
        )
        round_record["recorded_completion_count"] = round_record[
            "completion_candidate_count"
        ]
        round_record["pose_diagnostics_complete"] = True
        round_record["candidates"] = candidates
        round_record["selected_label"] = selected["label"]
        round_record["selected_row_shift"] = int(selected["row_shift"])
        round_record["selected_column_shift"] = int(selected["column_shift"])
        round_record["selected_completion_index"] = int(selected["selected_completion_index"])
        round_record["selected_completion_origin"] = str(selected["selected_completion_origin"])
        round_record["single_path_selected_label"] = single_path_selected["label"]
        round_record["single_path_selected_row_shift"] = int(
            single_path_selected["row_shift"]
        )
        round_record["single_path_selected_column_shift"] = int(
            single_path_selected["column_shift"]
        )
        round_record["single_path_prediction"] = single_path_selected["prediction"].copy()

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
        if selected is not current_candidate:
            selected["accepted_pose"] = True
            for completion in selected["completions"]:
                completion["accepted_global"] = bool(
                    completion["selected_within_pose"]
                )
        rounds.append(round_record)
        accepted_rounds += 1
        last_accepted_component = component.copy()
        last_accepted_seed_prediction = current.copy()
        last_accepted_row_shift = int(selected["row_shift"])
        last_accepted_column_shift = int(selected["column_shift"])
        last_accepted_completion_index = int(selected["selected_completion_index"])
        last_accepted_completion_origin = str(selected["selected_completion_origin"])
        current = selected["prediction"].copy()
        seen_layouts.add(selected_layout_key)

    single_path_reference_prediction = None
    if max_rounds == 1:
        single_path_reference_prediction = (
            rounds[0]["single_path_prediction"].copy() if rounds else initial_prediction.copy()
        )

    return {
        "prediction": current,
        "initial_score": fine_layout_score(scores, initial_prediction, grid),
        "final_score": fine_layout_score(scores, current, grid),
        "rounds": rounds,
        "rounds_attempted": len(rounds),
        "rounds_accepted": accepted_rounds,
        "stop_reason": stop_reason,
        "completion_beam_width": completion_beam_width,
        "translation_mode": translation_mode,
        "last_accepted_component": last_accepted_component,
        "last_accepted_seed_prediction": last_accepted_seed_prediction,
        "last_accepted_row_shift": last_accepted_row_shift,
        "last_accepted_column_shift": last_accepted_column_shift,
        "last_accepted_completion_index": last_accepted_completion_index,
        "last_accepted_completion_origin": last_accepted_completion_origin,
        "single_path_reference_prediction": single_path_reference_prediction,
    }


__all__ = [
    "profiled_large_neighborhood_reassembly",
    "seeded_profiled_beam_fill",
]
