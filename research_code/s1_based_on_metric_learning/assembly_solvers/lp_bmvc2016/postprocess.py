"""
@file: postprocess.py
@description: 忠实移植 do_greedy_assembly、TrimPuzzle 和 Type-1 FillPuzzleHoles_V2_rui3 后处理。
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.PuzzleDemoMGC_CVPR2012_Python.mgc import join_pieces_r, trim_puzzle

from .components import BlocksInfo, sco_update
from .matching import MIRROR, normalize_scores


def _find_block(blocks: list[np.ndarray | None], piece_id: int) -> int | None:
    for index, block in enumerate(blocks):
        if block is not None and np.any(block == piece_id):
            return index
    return None


def greedy_assemble_components(
    blocks_info: BlocksInfo,
    normalized_scores: np.ndarray,
    *,
    stop_threshold: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """以作者的 Gallagher 合并次序拼接 LP 刚性组件与剩余单 piece。"""

    scores = np.asarray(normalized_scores, dtype=np.float64).copy()
    blocks: list[np.ndarray | None] = [block.copy() for block in blocks_info.blocks]
    rotations: list[np.ndarray | None] = [rot.copy() for rot in blocks_info.rotations]
    n = scores.shape[0]
    accepted = 0
    rejected = 0
    iterations = 0

    while not np.all(np.isnan(scores)):
        try:
            flat_index = int(np.nanargmin(scores.ravel(order="F")))
        except ValueError:
            break
        best = float(scores.ravel(order="F")[flat_index])
        if not np.isfinite(best):
            break
        stop_after_iteration = best > stop_threshold
        anchor, neighbor, direction = np.unravel_index(
            flat_index, scores.shape, order="F"
        )
        piece1, piece2, how = anchor + 1, neighbor + 1, direction + 1
        first_index = _find_block(blocks, piece1)
        second_index = _find_block(blocks, piece2)
        first_block = piece1 if first_index is None else blocks[first_index]
        second_block = piece2 if second_index is None else blocks[second_index]
        first_rot = 1 if first_index is None else rotations[first_index]
        second_rot = 1 if second_index is None else rotations[second_index]

        if first_index is not None and first_index == second_index:
            success = False
            merged = np.empty((0, 0), dtype=np.int32)
            merged_rot = np.empty((0, 0), dtype=np.int32)
        else:
            merged, merged_rot, success = join_pieces_r(
                first_block, second_block, first_rot, second_rot, piece1, piece2, how
            )
        if success:
            accepted += 1
            if first_index is not None:
                blocks[first_index] = merged
                rotations[first_index] = merged_rot
                if second_index is not None:
                    blocks[second_index] = None
                    rotations[second_index] = None
            elif second_index is not None:
                blocks[second_index] = merged
                rotations[second_index] = merged_rot
            else:
                blocks.append(merged)
                rotations.append(merged_rot)

            scores[anchor, :, direction] = np.nan
            scores[:, neighbor, direction] = np.nan
            opposite = int(MIRROR[direction])
            scores[neighbor, :, opposite] = np.nan
            scores[:, anchor, opposite] = np.nan
            first_ids = np.asarray(first_block)[np.asarray(first_block) > 0].reshape(-1)
            second_ids = np.asarray(second_block)[np.asarray(second_block) > 0].reshape(-1)
            if np.asarray(first_block).size > 1 or np.asarray(second_block).size > 1:
                for first_id in first_ids:
                    for second_id in second_ids:
                        scores[int(first_id) - 1, int(second_id) - 1, :] = np.nan
                        scores[int(second_id) - 1, int(first_id) - 1, :] = np.nan
            if np.count_nonzero(merged > 0) > n - 1:
                break
        else:
            rejected += 1
            opposite = int(MIRROR[direction])
            scores[anchor, neighbor, direction] = np.nan
            scores[neighbor, anchor, opposite] = np.nan
        iterations += 1
        if stop_after_iteration:
            break

    sizes = [0 if block is None else int(np.count_nonzero(block > 0)) for block in blocks]
    if not sizes or max(sizes) == 0:
        return (
            np.zeros((1, 1), dtype=np.int32),
            np.zeros((1, 1), dtype=np.int32),
            {"iterations": iterations, "accepted": accepted, "rejected": rejected},
        )
    best_index = int(np.argmax(sizes))
    return blocks[best_index], rotations[best_index], {
        "iterations": iterations,
        "accepted": accepted,
        "rejected": rejected,
        "largest_block_pieces": sizes[best_index],
    }


def _hole_scores(
    block: np.ndarray, scores: np.ndarray, choices: np.ndarray, row: int, column: int
) -> np.ndarray:
    neighbors = [
        (row - 1, column, 0),
        (row, column - 1, 3),
        (row, column + 1, 1),
        (row + 1, column, 2),
    ]
    total = np.zeros(choices.size, dtype=np.float64)
    present = 0
    for neighbor_row, neighbor_column, direction in neighbors:
        if not (0 <= neighbor_row < block.shape[0] and 0 <= neighbor_column < block.shape[1]):
            continue
        known_piece = int(block[neighbor_row, neighbor_column])
        if known_piece <= 0:
            continue
        # Type-1 时 getAllScoresForCands_rui.m 恰为 SCO(cands,pieceID,Relposition)。
        total += scores[choices - 1, known_piece - 1, direction]
        present += 1
    return total if present else np.full(choices.size, np.inf)


def fill_puzzle_holes_type1(
    block: np.ndarray, rotations: np.ndarray, scores: np.ndarray, num_pieces: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """按“最可信洞优先、邻居代价求和”策略填充已知矩形内部空位。"""

    block = block.copy()
    rotations = rotations.copy()
    filled_count = 0
    while True:
        choices = np.setdiff1d(
            np.arange(1, num_pieces + 1, dtype=np.int32), block[block > 0]
        )
        if choices.size == 0:
            break
        holes = np.argwhere(block == 0)
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for row, column in holes:
            costs = _hole_scores(block, scores, choices, int(row), int(column))
            finite = np.sort(costs[np.isfinite(costs)])
            if finite.size == 0:
                continue
            ratio = float(finite[0] / finite[1]) if finite.size > 1 else float(finite[0])
            candidates.append((ratio, int(row), int(column), costs))
        if not candidates:
            break
        _, row, column, costs = min(candidates, key=lambda item: item[0])
        order = np.argsort(costs, kind="stable")
        selected = next((int(index) for index in order if np.isfinite(costs[index])), None)
        if selected is None:
            break
        block[row, column] = int(choices[selected])
        rotations[row, column] = 1
        filled_count += 1
    return block, rotations, filled_count


def trim_and_fill(
    original_scores: np.ndarray,
    blocks_info: BlocksInfo,
    labels: np.ndarray,
    grid: int,
    *,
    stop_threshold: float = 1.25,
) -> tuple[np.ndarray, dict]:
    """复现 ``trim_and_fill_v2.m`` 的 Type-1 完整后处理链。"""

    updated_scores = sco_update(original_scores, blocks_info, labels, remove_boundary=True)
    normalized = normalize_scores(updated_scores)
    assembled, rotations, greedy_diagnostics = greedy_assemble_components(
        blocks_info, normalized, stop_threshold=stop_threshold
    )
    trimmed, trimmed_rotations = trim_puzzle(assembled, rotations, grid, grid, 0)
    filled, _, filled_count = fill_puzzle_holes_type1(
        trimmed, trimmed_rotations, original_scores, grid * grid
    )
    return filled, {
        "greedy": greedy_diagnostics,
        "shape_before_trim": list(assembled.shape),
        "shape_after_trim": list(trimmed.shape),
        "holes_filled": filled_count,
        "final_pieces": int(np.count_nonzero(filled > 0)),
    }


__all__ = [
    "fill_puzzle_holes_type1",
    "greedy_assemble_components",
    "trim_and_fill",
]
