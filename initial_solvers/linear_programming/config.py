"""
@file: config.py
@description: 作者 MATLAB Type-1 LP 求解器参数及一致性检查。
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LPSolverConfig:
    """对应 ``PuzzleParametersSet.m`` 的 Type-1 参数。"""

    top_k: int = 1
    refinement_iterations: int = 5
    probability_lambda: float = 5.0
    minimum_weight: float = 1e-15
    match_tolerance: float = 1e-4
    rigid_components: bool = True
    buddy_check: bool = True
    loop_check: bool = True
    greedy_stop_threshold: float = 1.25
    highs_time_limit: float = 0.0

    def validate(self, num_pieces: int) -> None:
        if num_pieces < 4:
            raise ValueError("LP 求解至少需要 4 个 piece")
        if not 1 <= self.top_k < num_pieces:
            raise ValueError(f"top_k 必须位于 [1, {num_pieces - 1}]")
        if self.refinement_iterations <= 0:
            raise ValueError("refinement_iterations 必须为正整数")
        if self.probability_lambda <= 0:
            raise ValueError("probability_lambda 必须为正数")
        if not 0 < self.minimum_weight < 1:
            raise ValueError("minimum_weight 必须位于 (0, 1)")
        if self.match_tolerance <= 0:
            raise ValueError("match_tolerance 必须为正数")
        if self.greedy_stop_threshold <= 0:
            raise ValueError("greedy_stop_threshold 必须为正数")
        if self.highs_time_limit < 0:
            raise ValueError("highs_time_limit 不能为负数")
