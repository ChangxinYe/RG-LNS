"""Geometry helpers for right-edge canonical metric embeddings.

Direction ids follow the Gallagher solver convention:
0 = top, 1 = right, 2 = bottom, 3 = left.

New metric-learning runs use one fixed transform per physical edge, independent
of whether a piece is the anchor or candidate. This produces four embeddings
per piece. The role-aware helpers are retained only for old checkpoints.
"""

from __future__ import annotations

import numpy as np


TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3
DIRECTIONS = (TOP, RIGHT, BOTTOM, LEFT)
OPPOSITE = np.asarray([BOTTOM, LEFT, TOP, RIGHT], dtype=np.int64)
CANONICAL_EDGE = "right"
GEOMETRY_MODE = "right_edge_role_independent"
LEGACY_LEFT_MODE = "left_edge_role_aware"
LEGACY_RIGHT_MODE = "right_edge_role_aware"


def opposite_direction(direction: int) -> int:
    return int(OPPOSITE[int(direction)])


def relation_candidate_edge(direction: int) -> int:
    return opposite_direction(direction)


def _validate_canonical_edge(canonical_edge: str) -> str:
    canonical_edge = str(canonical_edge).lower()
    if canonical_edge not in {"left", "right"}:
        raise ValueError("canonical_edge must be 'left' or 'right'")
    return canonical_edge


def _rot_k_to_make_edge_right(direction: int) -> int:
    return (int(direction) - RIGHT) % 4


def _rot_k_to_make_edge_left(direction: int) -> int:
    return (int(direction) - LEFT) % 4


def canonical_piece(piece: np.ndarray, edge: int) -> np.ndarray:
    """Map one physical edge to the right using a role-independent transform.

    The paired transforms preserve the same boundary traversal order:

    - right: identity
    - left: horizontal flip
    - top: 90-degree clockwise rotation
    - bottom: 90-degree clockwise rotation followed by a horizontal flip
    """
    edge = int(edge)
    if edge == RIGHT:
        transformed = piece
    elif edge == LEFT:
        transformed = np.fliplr(piece)
    elif edge == TOP:
        transformed = np.rot90(piece, k=3)
    elif edge == BOTTOM:
        transformed = np.fliplr(np.rot90(piece, k=3))
    else:
        raise ValueError(f"Unknown edge direction: {edge}")
    return np.ascontiguousarray(transformed)


def canonical_anchor_piece(
    piece: np.ndarray,
    direction: int,
    canonical_edge: str = CANONICAL_EDGE,
) -> np.ndarray:
    """Put the anchor's selected edge on the requested canonical side."""
    canonical_edge = _validate_canonical_edge(canonical_edge)
    rotated = np.rot90(piece, k=_rot_k_to_make_edge_right(direction))
    if canonical_edge == "left":
        rotated = np.fliplr(rotated)
    return np.ascontiguousarray(rotated)


def canonical_candidate_piece(
    piece: np.ndarray,
    edge: int,
    canonical_edge: str = CANONICAL_EDGE,
) -> np.ndarray:
    """Put the candidate's touching edge on the requested canonical side."""
    canonical_edge = _validate_canonical_edge(canonical_edge)
    rotated = np.rot90(piece, k=_rot_k_to_make_edge_left(edge))
    if canonical_edge == "right":
        rotated = np.fliplr(rotated)
    return np.ascontiguousarray(rotated)


__all__ = [
    "CANONICAL_EDGE",
    "GEOMETRY_MODE",
    "LEGACY_LEFT_MODE",
    "LEGACY_RIGHT_MODE",
    "TOP",
    "RIGHT",
    "BOTTOM",
    "LEFT",
    "DIRECTIONS",
    "OPPOSITE",
    "canonical_piece",
    "canonical_anchor_piece",
    "canonical_candidate_piece",
    "relation_candidate_edge",
]
