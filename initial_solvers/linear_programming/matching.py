"""
@file: matching.py
@description: Faithful port of get_normSCO, get_candidate_and_weight, info_filter, and info_filter_loop.
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MIRROR = np.asarray([2, 3, 0, 1], dtype=np.int8)
LOOP_MATRIX = np.asarray(
    [
        [0, 1, 2, 3],
        [0, 3, 2, 1],
        [2, 1, 0, 3],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [3, 2, 1, 0],
        [1, 0, 3, 2],
        [1, 2, 3, 0],
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class MatchInfo:
    """Zero-based, column-wise representation of the MATLAB ``info`` matrix."""

    anchors: np.ndarray
    neighbors: np.ndarray
    weights: np.ndarray
    directions: np.ndarray

    def __len__(self) -> int:
        return int(self.anchors.size)

    def subset(self, mask: np.ndarray) -> "MatchInfo":
        mask = np.asarray(mask, dtype=bool)
        return MatchInfo(
            self.anchors[mask],
            self.neighbors[mask],
            self.weights[mask],
            self.directions[mask],
        )

    def rows(self, include_weight: bool = True) -> np.ndarray:
        columns = [self.anchors, self.neighbors]
        if include_weight:
            columns.append(self.weights)
        columns.append(self.directions)
        return np.column_stack(columns) if len(self) else np.empty((0, len(columns)))


def empty_info() -> MatchInfo:
    return MatchInfo(
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.int8),
    )


def unique_info(info: MatchInfo) -> MatchInfo:
    """Equivalent to MATLAB ``unique(info, 'rows')`` including the weight column."""

    if len(info) == 0:
        return info
    rows = info.rows()
    unique_rows = np.unique(rows, axis=0)
    return MatchInfo(
        unique_rows[:, 0].astype(np.int32),
        unique_rows[:, 1].astype(np.int32),
        unique_rows[:, 2].astype(np.float64),
        unique_rows[:, 3].astype(np.int8),
    )


def concatenate_info(*items: MatchInfo) -> MatchInfo:
    present = [item for item in items if len(item)]
    if not present:
        return empty_info()
    return MatchInfo(
        np.concatenate([item.anchors for item in present]),
        np.concatenate([item.neighbors for item in present]),
        np.concatenate([item.weights for item in present]),
        np.concatenate([item.directions for item in present]),
    )


def normalize_scores(scores: np.ndarray, tiny: float = 1e-15) -> np.ndarray:
    """Reproduce the row/column second-competitor normalization of ``get_normSCO.m`` per direction."""

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[0] != scores.shape[1] or scores.shape[2] != 4:
        raise ValueError(f"compatibility scores must have shape (N, N, 4), got {scores.shape}")
    if np.isnan(scores).any():
        raise ValueError("compatibility scores cannot contain NaN; use inf for pruned candidates")

    normalized = scores.copy()
    n = scores.shape[0]
    for direction in range(4):
        layer = scores[:, :, direction]
        row_order = np.argsort(layer, axis=1, kind="stable")
        row_sorted = np.take_along_axis(layer, row_order, axis=1)
        col_order = np.argsort(layer, axis=0, kind="stable")
        col_sorted = np.take_along_axis(layer, col_order, axis=0)
        row_minimum = row_sorted[:, 0]
        row_second = row_sorted[:, 1]
        row_argmin = row_order[:, 0]
        col_minimum = col_sorted[0, :]
        col_second = col_sorted[1, :]
        col_argmin = col_order[0, :]

        for anchor in range(n):
            values = layer[anchor, :]
            row_competitor = np.full(n, row_minimum[anchor], dtype=np.float64)
            row_competitor[row_argmin[anchor]] = row_second[anchor]
            col_competitor = col_minimum.copy()
            mask = col_argmin == anchor
            col_competitor[mask] = col_second[mask]
            with np.errstate(invalid="ignore", divide="ignore"):
                normalized[anchor, :, direction] = (values + tiny) / (
                    np.minimum(row_competitor, col_competitor) + tiny
                )
    return normalized


def get_candidate_and_weight(
    scores: np.ndarray,
    top_k: int,
    probability_lambda: float = 5.0,
    minimum_weight: float = 1e-15,
) -> MatchInfo:
    """Reproduce probability-weighted ``method=2`` by retaining Top-K candidates per piece and direction."""

    normalized = normalize_scores(scores)
    n = normalized.shape[0]
    anchors: list[int] = []
    neighbors: list[int] = []
    weights: list[float] = []
    directions: list[int] = []
    for direction in range(4):
        order = np.argsort(normalized[:, :, direction], axis=1, kind="stable")
        for rank in range(top_k):
            selected = order[:, rank]
            ratios = normalized[np.arange(n), selected, direction]
            candidate_weights = np.maximum(
                np.exp(-probability_lambda * ratios * ratios), minimum_weight
            )
            # MATLAB then removes candidates whose weight is exactly t.
            keep = np.isfinite(ratios) & (candidate_weights != minimum_weight)
            for anchor in np.flatnonzero(keep):
                anchors.append(int(anchor))
                neighbors.append(int(selected[anchor]))
                weights.append(float(candidate_weights[anchor]))
                directions.append(direction)
    return MatchInfo(
        np.asarray(anchors, dtype=np.int32),
        np.asarray(neighbors, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
        np.asarray(directions, dtype=np.int8),
    )


def info_filter(info: MatchInfo) -> MatchInfo:
    """Retain only candidates that also have a reverse-mirror record."""

    keys = {
        (int(a), int(b), int(d))
        for a, b, d in zip(info.anchors, info.neighbors, info.directions)
    }
    keep = np.asarray(
        [
            (int(b), int(a), int(MIRROR[int(d)])) in keys
            for a, b, d in zip(info.anchors, info.neighbors, info.directions)
        ],
        dtype=bool,
    )
    return info.subset(keep)


def info_filter_loop(info: MatchInfo) -> MatchInfo:
    """Reproduce the author's eight 2D four-step cycle checks, then run the buddy check again."""

    if len(info) == 0:
        return info
    delete = np.zeros(len(info), dtype=bool)
    for loop in LOOP_MATRIX:
        first_indices = np.flatnonzero(info.directions == loop[0])
        for first_index in first_indices:
            start = int(info.anchors[first_index])
            current = int(info.neighbors[first_index])
            complete = True
            for direction in loop[1:]:
                candidates = np.flatnonzero(
                    (info.directions == direction) & (info.anchors == current)
                )
                if candidates.size == 0:
                    complete = False
                    break
                # MATLAB ismember returns the first matching item.
                current = int(info.neighbors[int(candidates[0])])
            if complete and current != start:
                delete[first_index] = True
    return info_filter(info.subset(~delete))


__all__ = [
    "MIRROR",
    "MatchInfo",
    "concatenate_info",
    "empty_info",
    "get_candidate_and_weight",
    "info_filter",
    "info_filter_loop",
    "normalize_scores",
    "unique_info",
]
