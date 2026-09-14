"""
@file: metric_lsej_macro_assisted.py
@description: S4B 宏观辅助重排的布局合成、可靠性门控、排名惩罚与诊断工具。
@author: Changxin Ye
@created: 2026-07-23
@version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .metric_geometry import BOTTOM, LEFT, OPPOSITE, RIGHT, TOP
except ImportError:
    from metric_geometry import BOTTOM, LEFT, OPPOSITE, RIGHT, TOP


@dataclass(frozen=True)
class MacroRefinementConfig:
    """One macro scale used to diagnose or refine a current grid10 layout."""

    scale: int
    weight: float
    min_rank_penalty: float = 0.0
    internal_top_k: int = 3
    internal_confidence_threshold: float = 1.0

    def validate(self, grid: int) -> None:
        if self.scale <= 1 or grid % self.scale != 0:
            raise ValueError(f"Macro scale {self.scale} must divide grid {grid}")
        if self.weight < 0:
            raise ValueError("Macro refinement weight must be non-negative")
        if not 0.0 <= self.min_rank_penalty <= 1.0:
            raise ValueError("min_rank_penalty must be in [0, 1]")
        if self.internal_top_k <= 0:
            raise ValueError("internal_top_k must be positive")
        if not 0.0 <= self.internal_confidence_threshold <= 1.0:
            raise ValueError("internal_confidence_threshold must be in [0, 1]")


def positions_to_layout(prediction: np.ndarray, grid: int) -> np.ndarray:
    """Convert piece->position predictions to a position->piece grid."""

    prediction = np.asarray(prediction, dtype=np.int64)
    num_pieces = int(grid) ** 2
    if prediction.shape != (num_pieces,):
        raise ValueError(f"Expected prediction shape ({num_pieces},), got {prediction.shape}")
    if not np.array_equal(np.sort(prediction), np.arange(num_pieces)):
        raise ValueError("Prediction must be a complete permutation of grid positions")
    layout = np.empty(num_pieces, dtype=np.int64)
    layout[prediction] = np.arange(num_pieces, dtype=np.int64)
    return layout.reshape(grid, grid)


def compose_predicted_macro_pieces(
    pieces: list[np.ndarray],
    prediction: np.ndarray,
    grid: int,
    scale: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Compose non-overlapping macro pieces from the current predicted layout.

    The returned membership tensor has shape ``[macro_count, scale, scale]``
    and stores the original shuffled-piece indices occupying each macro block.
    """

    if scale <= 1 or grid % scale != 0:
        raise ValueError(f"Macro scale {scale} must divide grid {grid}")
    if len(pieces) != grid * grid:
        raise ValueError(f"Expected {grid * grid} pieces, got {len(pieces)}")
    layout = positions_to_layout(prediction, grid)
    macro_grid = grid // scale
    macro_pieces: list[np.ndarray] = []
    memberships = []

    for macro_row in range(macro_grid):
        for macro_column in range(macro_grid):
            members = layout[
                macro_row * scale : (macro_row + 1) * scale,
                macro_column * scale : (macro_column + 1) * scale,
            ].copy()
            rows = [
                np.concatenate([pieces[int(piece)] for piece in member_row], axis=1)
                for member_row in members
            ]
            macro_pieces.append(np.ascontiguousarray(np.concatenate(rows, axis=0)))
            memberships.append(members)

    return macro_pieces, np.stack(memberships, axis=0)


def candidate_rank(scores: np.ndarray, anchor: int, candidate: int, direction: int) -> int:
    """Return the candidate's worst rank within an exact score tie."""

    scores = np.asarray(scores)
    if scores.ndim != 3 or scores.shape[0] != scores.shape[1] or scores.shape[2] != 4:
        raise ValueError(f"Expected square score tensor [N,N,4], got {scores.shape}")
    num_pieces = scores.shape[0]
    if not (0 <= anchor < num_pieces and 0 <= candidate < num_pieces and anchor != candidate):
        raise ValueError("anchor and candidate must be distinct valid indices")
    candidate_score = float(scores[anchor, candidate, int(direction)])
    if not np.isfinite(candidate_score):
        raise ValueError("Candidate relation has a non-finite compatibility score")
    mask = np.ones(num_pieces, dtype=bool)
    mask[anchor] = False
    rank = int(np.count_nonzero(mask & (scores[anchor, :, int(direction)] <= candidate_score)))
    if not 1 <= rank <= num_pieces - 1:
        raise RuntimeError(f"Resolved invalid candidate rank: {rank}")
    return rank


def _directed_internal_relations(members: np.ndarray):
    scale = int(members.shape[0])
    for row in range(scale):
        for column in range(scale):
            anchor = int(members[row, column])
            if column + 1 < scale:
                candidate = int(members[row, column + 1])
                yield anchor, candidate, RIGHT
                yield candidate, anchor, LEFT
            if row + 1 < scale:
                candidate = int(members[row + 1, column])
                yield anchor, candidate, BOTTOM
                yield candidate, anchor, TOP


def macro_internal_confidence(
    members: np.ndarray,
    piece_scores: np.ndarray,
    top_k: int,
) -> float:
    """Fraction of directed internal edges whose S1A candidate rank is <= K."""

    ranks = [
        candidate_rank(piece_scores, anchor, candidate, direction)
        for anchor, candidate, direction in _directed_internal_relations(members)
    ]
    return float(np.mean(np.asarray(ranks) <= int(top_k))) if ranks else 1.0


def _is_true_relation(
    anchor_piece: int,
    candidate_piece: int,
    direction: int,
    target: np.ndarray,
    grid: int,
) -> bool:
    anchor_position = int(target[int(anchor_piece)])
    candidate_position = int(target[int(candidate_piece)])
    anchor_row, anchor_column = divmod(anchor_position, grid)
    candidate_row, candidate_column = divmod(candidate_position, grid)
    expected = {
        TOP: (-1, 0),
        RIGHT: (0, 1),
        BOTTOM: (1, 0),
        LEFT: (0, -1),
    }[int(direction)]
    return (candidate_row - anchor_row, candidate_column - anchor_column) == expected


def macro_internal_is_correct(members: np.ndarray, target: np.ndarray, grid: int) -> bool:
    """Ground-truth diagnostic only; never use this value to modify scores."""

    return all(
        _is_true_relation(anchor, candidate, direction, target, grid)
        for anchor, candidate, direction in _directed_internal_relations(members)
    )


def macro_boundary_pairs(
    anchor_members: np.ndarray,
    candidate_members: np.ndarray,
    direction: int,
) -> list[tuple[int, int]]:
    """Return the fine-piece pairs along one current macro boundary."""

    if anchor_members.shape != candidate_members.shape or anchor_members.ndim != 2:
        raise ValueError("Macro memberships must be equally sized square matrices")
    direction = int(direction)
    if direction == RIGHT:
        return [
            (int(anchor_members[row, -1]), int(candidate_members[row, 0]))
            for row in range(anchor_members.shape[0])
        ]
    if direction == BOTTOM:
        return [
            (int(anchor_members[-1, column]), int(candidate_members[0, column]))
            for column in range(anchor_members.shape[1])
        ]
    raise ValueError("Only RIGHT and BOTTOM macro boundaries are enumerated directly")


def macro_boundary_is_correct(
    anchor_members: np.ndarray,
    candidate_members: np.ndarray,
    direction: int,
    target: np.ndarray,
    grid: int,
) -> bool:
    return all(
        _is_true_relation(anchor, candidate, direction, target, grid)
        for anchor, candidate in macro_boundary_pairs(anchor_members, candidate_members, direction)
    )


def iter_macro_grid_relations(macro_grid: int):
    """Enumerate each undirected current macro boundary once."""

    for row in range(macro_grid):
        for column in range(macro_grid):
            anchor = row * macro_grid + column
            if column + 1 < macro_grid:
                yield anchor, anchor + 1, RIGHT
            if row + 1 < macro_grid:
                yield anchor, anchor + macro_grid, BOTTOM


def macro_relation_rank_penalty(
    macro_scores: np.ndarray,
    anchor: int,
    candidate: int,
    direction: int,
) -> tuple[int, int, float]:
    """Average normalized forward/reverse ranks; rank 1 has zero penalty."""

    num_macros = int(macro_scores.shape[0])
    forward_rank = candidate_rank(macro_scores, anchor, candidate, direction)
    reverse_direction = int(OPPOSITE[int(direction)])
    reverse_rank = candidate_rank(macro_scores, candidate, anchor, reverse_direction)
    denominator = max(num_macros - 2, 1)
    penalty = 0.5 * (
        (forward_rank - 1) / denominator + (reverse_rank - 1) / denominator
    )
    return forward_rank, reverse_rank, float(np.clip(penalty, 0.0, 1.0))


def apply_macro_rank_penalties(
    piece_scores: np.ndarray,
    macro_scores: np.ndarray,
    memberships: np.ndarray,
    target: np.ndarray | None,
    grid: int,
    config: MacroRefinementConfig,
    round_index: int,
    confidence_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Penalize suspicious current macro boundaries and return diagnostic rows.

    Ground truth is optional and is copied only into diagnostic columns. The
    score update itself depends solely on model scores and the configured gate.
    """

    config.validate(grid)
    piece_scores = np.asarray(piece_scores, dtype=np.float32)
    refined_scores = piece_scores.copy()
    confidence_scores = (
        piece_scores
        if confidence_scores is None
        else np.asarray(confidence_scores, dtype=np.float32)
    )
    if confidence_scores.shape != piece_scores.shape:
        raise ValueError("confidence_scores and piece_scores must have the same shape")
    macro_grid = grid // config.scale
    expected_macros = macro_grid * macro_grid
    if memberships.shape != (expected_macros, config.scale, config.scale):
        raise ValueError(
            f"Expected memberships {(expected_macros, config.scale, config.scale)}, "
            f"got {memberships.shape}"
        )
    if macro_scores.shape != (expected_macros, expected_macros, 4):
        raise ValueError(
            f"Expected macro scores {(expected_macros, expected_macros, 4)}, got {macro_scores.shape}"
        )

    confidences = np.asarray(
        [
            macro_internal_confidence(members, confidence_scores, config.internal_top_k)
            for members in memberships
        ],
        dtype=np.float64,
    )
    gt_internal = None
    if target is not None:
        target = np.asarray(target, dtype=np.int64)
        gt_internal = np.asarray(
            [macro_internal_is_correct(members, target, grid) for members in memberships],
            dtype=bool,
        )

    diagnostics: list[dict] = []
    for anchor, candidate, direction in iter_macro_grid_relations(macro_grid):
        forward_rank, reverse_rank, rank_penalty = macro_relation_rank_penalty(
            macro_scores, anchor, candidate, direction
        )
        reliable = bool(
            confidences[anchor] >= config.internal_confidence_threshold
            and confidences[candidate] >= config.internal_confidence_threshold
        )
        passes_rank_threshold = bool(
            rank_penalty > 0.0 and rank_penalty >= config.min_rank_penalty
        )
        applied_penalty = (
            float(config.weight * rank_penalty)
            if reliable and passes_rank_threshold
            else 0.0
        )
        for anchor_piece, candidate_piece in macro_boundary_pairs(
            memberships[anchor], memberships[candidate], direction
        ):
            refined_scores[anchor_piece, candidate_piece, direction] += applied_penalty
            reverse_direction = int(OPPOSITE[int(direction)])
            refined_scores[candidate_piece, anchor_piece, reverse_direction] += applied_penalty

        boundary_correct = None
        anchor_gt_correct = None
        candidate_gt_correct = None
        clean_pair = None
        if target is not None and gt_internal is not None:
            anchor_gt_correct = bool(gt_internal[anchor])
            candidate_gt_correct = bool(gt_internal[candidate])
            clean_pair = bool(anchor_gt_correct and candidate_gt_correct)
            boundary_correct = macro_boundary_is_correct(
                memberships[anchor], memberships[candidate], direction, target, grid
            )

        diagnostics.append(
            {
                "round": int(round_index),
                "scale": int(config.scale),
                "macro_grid": int(macro_grid),
                "direction": "right" if direction == RIGHT else "bottom",
                "anchor_macro": int(anchor),
                "candidate_macro": int(candidate),
                "forward_rank": int(forward_rank),
                "reverse_rank": int(reverse_rank),
                "rank_penalty": float(rank_penalty),
                "anchor_internal_confidence": float(confidences[anchor]),
                "candidate_internal_confidence": float(confidences[candidate]),
                "reliable": reliable,
                "weight": float(config.weight),
                "min_rank_penalty": float(config.min_rank_penalty),
                "passes_rank_threshold": passes_rank_threshold,
                "applied_penalty": applied_penalty,
                "anchor_gt_internal_correct": anchor_gt_correct,
                "candidate_gt_internal_correct": candidate_gt_correct,
                "clean_pair": clean_pair,
                "boundary_correct": boundary_correct,
            }
        )

    return refined_scores, diagnostics


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with ties; used by the dependency-free AUROC."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def binary_ranking_metrics(labels: np.ndarray, anomaly_scores: np.ndarray) -> dict[str, float | int | None]:
    """AUROC/AP where label 1 and a larger score both mean an incorrect relation."""

    labels = np.asarray(labels, dtype=np.int64)
    anomaly_scores = np.asarray(anomaly_scores, dtype=np.float64)
    if labels.shape != anomaly_scores.shape or labels.ndim != 1:
        raise ValueError("labels and anomaly_scores must be equally sized vectors")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        auroc = None
    else:
        ranks = _average_ranks(anomaly_scores)
        positive_rank_sum = float(ranks[labels == 1].sum())
        auroc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

    if positives == 0:
        average_precision = None
    else:
        order = np.argsort(-anomaly_scores, kind="mergesort")
        sorted_labels = labels[order]
        sorted_scores = anomaly_scores[order]
        cumulative_positives = 0
        cumulative_relations = 0
        previous_recall = 0.0
        average_precision = 0.0
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            cumulative_positives += int(sorted_labels[start:end].sum())
            cumulative_relations += end - start
            recall = cumulative_positives / positives
            precision = cumulative_positives / cumulative_relations
            average_precision += (recall - previous_recall) * precision
            previous_recall = recall
            start = end
        average_precision = float(average_precision)

    return {
        "relations": int(len(labels)),
        "incorrect_relations": positives,
        "correct_relations": negatives,
        "incorrect_auroc": None if auroc is None else float(auroc),
        "incorrect_average_precision": average_precision,
    }


def summarize_macro_diagnostics(rows: list[dict]) -> dict:
    """Summarize each scale, including a clean-internal-block diagnostic subset."""

    summary: dict[str, dict] = {}
    scales = sorted({int(row["scale"]) for row in rows})
    for scale in scales:
        scale_rows = [row for row in rows if int(row["scale"]) == scale]
        scale_summary: dict[str, object] = {
            "relations": len(scale_rows),
            "reliable_relations": int(sum(bool(row["reliable"]) for row in scale_rows)),
            "threshold_selected_relations": int(
                sum(
                    bool(row["reliable"]) and bool(row["passes_rank_threshold"])
                    for row in scale_rows
                )
            ),
            "applied_relations": int(
                sum(float(row["applied_penalty"]) > 0.0 for row in scale_rows)
            ),
            "mean_rank_penalty": float(np.mean([row["rank_penalty"] for row in scale_rows])),
            "mean_applied_penalty": float(np.mean([row["applied_penalty"] for row in scale_rows])),
        }
        for subset_name, subset_rows in {
            "all": scale_rows,
            "clean_internal_blocks": [row for row in scale_rows if row["clean_pair"] is True],
            "predicted_reliable": [row for row in scale_rows if row["reliable"]],
        }.items():
            labeled = [row for row in subset_rows if row["boundary_correct"] is not None]
            if labeled:
                labels = np.asarray([not bool(row["boundary_correct"]) for row in labeled], dtype=np.int64)
                penalties = np.asarray([row["rank_penalty"] for row in labeled], dtype=np.float64)
                metrics = binary_ranking_metrics(labels, penalties)
                correct_penalties = penalties[labels == 0]
                incorrect_penalties = penalties[labels == 1]
                metrics.update(
                    {
                        "mean_penalty_correct": (
                            float(correct_penalties.mean()) if len(correct_penalties) else None
                        ),
                        "mean_penalty_incorrect": (
                            float(incorrect_penalties.mean()) if len(incorrect_penalties) else None
                        ),
                    }
                )
            else:
                metrics = {
                    "relations": 0,
                    "incorrect_relations": 0,
                    "correct_relations": 0,
                    "incorrect_auroc": None,
                    "incorrect_average_precision": None,
                    "mean_penalty_correct": None,
                    "mean_penalty_incorrect": None,
                }
            scale_summary[subset_name] = metrics
        summary[f"macro{scale}"] = scale_summary
    return summary


__all__ = [
    "MacroRefinementConfig",
    "apply_macro_rank_penalties",
    "binary_ranking_metrics",
    "candidate_rank",
    "compose_predicted_macro_pieces",
    "iter_macro_grid_relations",
    "macro_boundary_is_correct",
    "macro_boundary_pairs",
    "macro_internal_confidence",
    "macro_internal_is_correct",
    "macro_relation_rank_penalty",
    "positions_to_layout",
    "summarize_macro_diagnostics",
]
