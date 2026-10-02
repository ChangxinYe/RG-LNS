"""
@file: solver.py
@description: 逐步复现 PuzzleMain_test.m 的 Type-1 LP、多轮刚性组件优化和 Gallagher 后处理主流程。
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .components import (
    blocks_info_cut_type1,
    get_connected_from_solution,
    recover_connected_blocks,
    sco_update,
)
from .config import LPSolverConfig
from .linear_program import find_best_fix_piece, solve_weighted_l1_lp
from .matching import get_candidate_and_weight, info_filter
from .postprocess import trim_and_fill


def _grid_to_prediction(block: np.ndarray, grid: int) -> np.ndarray:
    prediction = np.full(grid * grid, -1, dtype=np.int32)
    for row in range(min(grid, block.shape[0])):
        for column in range(min(grid, block.shape[1])):
            piece_id = int(block[row, column])
            if 1 <= piece_id <= grid * grid:
                prediction[piece_id - 1] = row * grid + column
    return prediction


def solve_with_lp(
    pieces,
    scores: np.ndarray,
    grid: int,
    *,
    verbose: bool = False,
    config: LPSolverConfig | None = None,
    return_diagnostics: bool = False,
):
    """不依赖 MATLAB，执行作者 Type-1 算法的纯 Python 等价流程。"""

    del pieces  # Type-1 布局只使用兼容分数；图像仅用于 MATLAB 的中间显示。
    config = config or LPSolverConfig()
    grid = int(grid)
    num_pieces = grid * grid
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (num_pieces, num_pieces, 4):
        raise ValueError(
            f"兼容分数应为 ({num_pieces}, {num_pieces}, 4)，实际为 {scores.shape}"
        )
    if np.isnan(scores).any():
        raise ValueError("兼容分数中不能含 NaN")
    config.validate(num_pieces)

    info = get_candidate_and_weight(
        scores,
        config.top_k,
        config.probability_lambda,
        config.minimum_weight,
    )
    if len(info) == 0:
        raise RuntimeError("初始候选集合为空")
    fix_piece = find_best_fix_piece(info, num_pieces)
    coordinates, lp_record = solve_weighted_l1_lp(
        num_pieces,
        info,
        fix_piece=fix_piece,
        time_limit=config.highs_time_limit,
    )
    labels, info = get_connected_from_solution(
        coordinates,
        info,
        num_pieces,
        buddy_check=config.buddy_check,
        loop_check=config.loop_check,
        tolerance=config.match_tolerance,
    )
    labels, blocks_info = recover_connected_blocks(labels, coordinates, num_pieces)
    rounds = [
        {
            "round": 0,
            **lp_record,
            "accepted_matches": len(info),
            "components": len(blocks_info.blocks),
        }
    ]
    updated_scores = scores.copy()

    for iteration in range(1, config.refinement_iterations + 1):
        updated_scores = sco_update(
            updated_scores, blocks_info, labels, remove_boundary=True
        )
        proposed = get_candidate_and_weight(
            updated_scores,
            config.top_k,
            config.probability_lambda,
            config.minimum_weight,
        )
        if config.buddy_check:
            proposed = info_filter(proposed)
        blocks_info, broken_pairs = blocks_info_cut_type1(blocks_info, num_pieces)
        if broken_pairs.size:
            raise AssertionError("Type-1 不应产生旋转复制冲突切边")
        if len(proposed) == 0:
            rounds.append({"round": iteration, "stop_reason": "no_candidates"})
            break

        previous_labels = labels.copy()
        previous_info = info
        coordinates, lp_record = solve_weighted_l1_lp(
            num_pieces,
            proposed,
            fix_piece=fix_piece,
            previous_labels=labels,
            previous_coordinates=coordinates,
            rigid_components=config.rigid_components,
            time_limit=config.highs_time_limit,
        )
        labels, info = get_connected_from_solution(
            coordinates,
            proposed,
            num_pieces,
            previous=previous_info,
            buddy_check=config.buddy_check,
            loop_check=config.loop_check,
            tolerance=config.match_tolerance,
        )
        labels, blocks_info = recover_connected_blocks(labels, coordinates, num_pieces)
        rounds.append(
            {
                "round": iteration,
                **lp_record,
                "accepted_matches": len(info),
                "components": len(blocks_info.blocks),
            }
        )
        if verbose:
            print(
                f"LP 第 {iteration} 轮：候选={len(proposed)}，保留={len(info)}，"
                f"刚性组件={len(blocks_info.blocks)}，目标值={lp_record['objective']:.6g}"
            )
        if np.array_equal(previous_labels, labels):
            rounds[-1]["stop_reason"] = "labels_unchanged"
            break

    blocks_info, broken_pairs = blocks_info_cut_type1(blocks_info, num_pieces)
    if broken_pairs.size:
        raise AssertionError("Type-1 不应产生旋转复制冲突切边")
    final_block, postprocess_diagnostics = trim_and_fill(
        scores,
        blocks_info,
        labels,
        grid,
        stop_threshold=config.greedy_stop_threshold,
    )
    prediction = _grid_to_prediction(final_block, grid)
    diagnostics = {
        "implementation": "faithful_bmvc2016_type1_python_port",
        "matlab_entry": "PuzzleMain_test.m",
        "config": asdict(config),
        "rounds": rounds,
        "rounds_completed": len(rounds),
        "fix_piece_zero_based": fix_piece,
        "postprocess": postprocess_diagnostics,
        "final_block": final_block.copy(),
    }
    if return_diagnostics:
        return prediction, diagnostics
    return prediction


__all__ = ["LPSolverConfig", "solve_with_lp"]
