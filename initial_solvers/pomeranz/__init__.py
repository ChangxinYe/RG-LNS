"""
@file: __init__.py
@description: Entry point for the pure-Python Pomeranz CVPR 2011 solver using the unified E1 compatibility distance.
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from .config import PomeranzSolverConfig
from .solver import solve_with_pomeranz

__all__ = ["PomeranzSolverConfig", "solve_with_pomeranz"]
