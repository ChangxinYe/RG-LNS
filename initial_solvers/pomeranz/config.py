"""
@file: config.py
@description: Reproducible experiment parameters for the Pomeranz CVPR 2011 greedy jigsaw solver.
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PomeranzSolverConfig:
    """Pomeranz solver parameters when using the unified learned compatibility distance."""

    restarts: int = 10
    max_refinement_rounds: int = 0
    random_seed: int = 0

    def validate(self, num_pieces: int) -> None:
        if num_pieces < 4:
            raise ValueError("Pomeranz solver requires at least 4 pieces")
        if self.restarts <= 0:
            raise ValueError("restarts must be a positive integer")
        if self.max_refinement_rounds < 0:
            raise ValueError("max_refinement_rounds cannot be negative; 0 means until BBM no longer improves")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")


__all__ = ["PomeranzSolverConfig"]
