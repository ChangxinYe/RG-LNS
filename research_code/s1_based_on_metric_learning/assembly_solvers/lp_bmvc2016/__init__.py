"""
@file: __init__.py
@description: Yu 等 BMVC 2016 Type-1 线性规划拼图算法的纯 Python 等价移植入口。
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from .config import LPSolverConfig
from .solver import solve_with_lp

__all__ = ["LPSolverConfig", "solve_with_lp"]
