"""
@file: config.py
@description: Pomeranz CVPR 2011 贪心拼图求解器的可复现实验参数。
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PomeranzSolverConfig:
    """使用统一 S1A E1 距离时的 Pomeranz 求解参数。"""

    restarts: int = 10
    max_refinement_rounds: int = 0
    random_seed: int = 0

    def validate(self, num_pieces: int) -> None:
        if num_pieces < 4:
            raise ValueError("Pomeranz 求解器至少需要 4 个 piece")
        if self.restarts <= 0:
            raise ValueError("restarts 必须为正整数")
        if self.max_refinement_rounds < 0:
            raise ValueError("max_refinement_rounds 不能为负数；0 表示直到 BBM 不再提升")
        if self.random_seed < 0:
            raise ValueError("random_seed 不能为负数")


__all__ = ["PomeranzSolverConfig"]
