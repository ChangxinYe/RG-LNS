"""
@file: solver.py
@description: 基于论文流程组织 Pomeranz Placer、Segmenter 与 Shifter，并接入 S1A E1 距离。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

import numpy as np

from .best_buddies import (
    best_buddies_layout_score,
    compute_best_neighbors,
    compute_mutual_best_buddies,
    validate_scores,
)
from .config import PomeranzSolverConfig
from .placer import greedy_place
from .segmenter import best_buddy_components
from .shifter import component_seed_positions


def _restart_seed_ids(num_pieces: int, config: PomeranzSolverConfig) -> np.ndarray:
    rng = np.random.default_rng(config.random_seed)
    if config.restarts <= num_pieces:
        return rng.choice(num_pieces, size=config.restarts, replace=False).astype(np.int32)
    initial = rng.permutation(num_pieces).astype(np.int32)
    extra = rng.choice(num_pieces, size=config.restarts - num_pieces, replace=True).astype(np.int32)
    return np.concatenate([initial, extra])


def _run_restart(
    scores: np.ndarray,
    best_neighbors: np.ndarray,
    mutual_best_buddies: np.ndarray,
    grid: int,
    seed_piece: int,
    config: PomeranzSolverConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    current = greedy_place(
        scores,
        best_neighbors,
        mutual_best_buddies,
        grid,
        {int(seed_piece): (0, 0)},
    )
    current_score = best_buddies_layout_score(current, best_neighbors, mutual_best_buddies, grid)
    history = [{"round": 0, "seed_piece": int(seed_piece), **current_score}]
    seen_layouts = {current.tobytes()}
    automatic_round_limit = 2 * grid * (grid - 1) + 1
    round_limit = config.max_refinement_rounds or automatic_round_limit
    stop_reason = (
        "max_refinement_rounds"
        if config.max_refinement_rounds
        else "theoretical_bbm_round_bound"
    )

    for round_index in range(1, round_limit + 1):
        components = best_buddy_components(
            current,
            best_neighbors,
            mutual_best_buddies,
            grid,
            rng=rng,
        )
        largest = components[0]
        if len(largest) == grid * grid:
            stop_reason = "all_pieces_in_largest_component"
            break
        candidate = greedy_place(
            scores,
            best_neighbors,
            mutual_best_buddies,
            grid,
            component_seed_positions(current, largest, grid),
        )
        candidate_score = best_buddies_layout_score(candidate, best_neighbors, mutual_best_buddies, grid)
        record = {
            "round": int(round_index),
            "largest_component_size": int(len(largest)),
            **candidate_score,
        }
        if int(candidate_score["best_buddy_edges"]) <= int(current_score["best_buddy_edges"]):
            record["accepted"] = False
            history.append(record)
            stop_reason = "no_strict_bbm_improvement"
            break
        candidate_key = candidate.tobytes()
        if candidate_key in seen_layouts:
            record["accepted"] = False
            history.append(record)
            stop_reason = "repeated_layout"
            break
        record["accepted"] = True
        history.append(record)
        current = candidate
        current_score = candidate_score
        seen_layouts.add(candidate_key)

    return current, {
        "seed_piece": int(seed_piece),
        "rounds_attempted": max(0, len(history) - 1),
        "rounds_accepted": sum(int(row.get("accepted", False)) for row in history),
        "stop_reason": stop_reason,
        "history": history,
        **current_score,
    }


def solve_with_pomeranz(
    pieces,
    scores: np.ndarray,
    grid: int,
    *,
    config: PomeranzSolverConfig | None = None,
    verbose: bool = False,
    return_diagnostics: bool = False,
):
    """使用统一 S1A E1 矩阵执行多种子 Pomeranz CVPR 2011 求解流程。"""

    del pieces  # 与其他求解器保持统一调用签名；Pomeranz 仅使用预计算 E1 分数。
    config = config or PomeranzSolverConfig()
    num_pieces = grid * grid
    config.validate(num_pieces)
    scores = validate_scores(scores, num_pieces)
    best_neighbors = compute_best_neighbors(scores)
    mutual_best_buddies = compute_mutual_best_buddies(best_neighbors)

    successful: list[tuple[np.ndarray, dict]] = []
    failures: list[dict] = []
    for restart_index, seed_piece in enumerate(_restart_seed_ids(num_pieces, config)):
        try:
            prediction, diagnostics = _run_restart(
                scores,
                best_neighbors,
                mutual_best_buddies,
                grid,
                int(seed_piece),
                config,
                np.random.default_rng(
                    config.random_seed + 1_000_003 * (int(restart_index) + 1)
                ),
            )
            diagnostics["restart_index"] = int(restart_index)
            successful.append((prediction, diagnostics))
            if verbose:
                print(
                    f"  Pomeranz restart={restart_index + 1}/{config.restarts}, "
                    f"seed={int(seed_piece)}, BBM={diagnostics['best_buddies_metric']:.4f}, "
                    f"rounds={diagnostics['rounds_accepted']}/{diagnostics['rounds_attempted']}, "
                    f"stop={diagnostics['stop_reason']}"
                )
        except (RuntimeError, ValueError) as error:
            failures.append(
                {
                    "restart_index": int(restart_index),
                    "seed_piece": int(seed_piece),
                    "error": str(error),
                }
            )
            if verbose:
                print(f"  Pomeranz restart={restart_index + 1}/{config.restarts} failed: {error}")

    if not successful:
        prediction = np.full(num_pieces, -1, dtype=np.int32)
        diagnostics = {
            "status": "all_restarts_failed",
            "failures": failures,
            "restarts_attempted": int(config.restarts),
            "restarts_succeeded": 0,
        }
    else:
        best_index = max(
            range(len(successful)),
            key=lambda index: (
                int(successful[index][1]["best_buddy_edges"]),
                -int(successful[index][1]["restart_index"]),
            ),
        )
        prediction, best_restart = successful[best_index]
        diagnostics = {
            "status": "complete",
            "selected_restart": int(best_restart["restart_index"]),
            "selected_seed_piece": int(best_restart["seed_piece"]),
            "best_buddy_edges": int(best_restart["best_buddy_edges"]),
            "total_edges": int(best_restart["total_edges"]),
            "best_buddies_metric": float(best_restart["best_buddies_metric"]),
            "rounds_attempted": int(best_restart["rounds_attempted"]),
            "rounds_accepted": int(best_restart["rounds_accepted"]),
            "stop_reason": best_restart["stop_reason"],
            "history": best_restart["history"],
            "restart_summaries": [item[1] for item in successful],
            "failures": failures,
            "restarts_attempted": int(config.restarts),
            "restarts_succeeded": len(successful),
        }
    if return_diagnostics:
        return prediction, diagnostics
    return prediction


__all__ = ["solve_with_pomeranz"]
