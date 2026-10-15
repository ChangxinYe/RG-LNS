"""
@file: __init__.py
@description: Entry point for the pure-Python equivalent of Yu et al.'s BMVC 2016 Type-1 linear-programming jigsaw algorithm.
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from .config import LPSolverConfig
from .solver import solve_with_lp

__all__ = ["LPSolverConfig", "solve_with_lp"]
