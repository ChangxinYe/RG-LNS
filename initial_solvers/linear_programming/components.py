"""
@file: components.py
@description: 忠实移植 check_match_lp、get_connected_from_sol、recover_connected_block 与 SCO_update 的 Type-1 路径。
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as image_label
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .linear_program import DIRECTION_OFFSETS
from .matching import MIRROR, MatchInfo, concatenate_info, info_filter, info_filter_loop, unique_info


@dataclass
class BlocksInfo:
    blocks: list[np.ndarray]
    rotations: list[np.ndarray]


def blocks_info_cut_type1(
    blocks_info: BlocksInfo, num_pieces: int
) -> tuple[BlocksInfo, np.ndarray]:
    """Type-1 等价移植的 ``BlocksInfo_cut.m``。

    原函数仅在同一原始 piece 的旋转复制发生冲突时切块。Type-1 不复制 piece，
    ``get_all_ind`` 对每个块内 id 恒产生四个不同旋转 id，因此其切块条件恒为假。
    """

    del num_pieces
    return blocks_info, np.empty((0, 2), dtype=np.int32)


def check_match_lp(
    info: MatchInfo, coordinates: np.ndarray, tolerance: float = 1e-4
) -> np.ndarray:
    """按作者的严格阈值检查 LP 坐标是否恰好满足候选邻接。"""

    observed = coordinates[info.neighbors] - coordinates[info.anchors]
    expected = DIRECTION_OFFSETS[info.directions]
    return np.all(np.abs(observed - expected) < tolerance, axis=1)


def labels_from_info(num_pieces: int, info: MatchInfo) -> np.ndarray:
    if len(info):
        rows = np.concatenate([info.anchors, info.neighbors])
        columns = np.concatenate([info.neighbors, info.anchors])
        graph = coo_matrix(
            (np.ones(rows.size), (rows, columns)), shape=(num_pieces, num_pieces)
        ).tocsr()
    else:
        graph = coo_matrix((num_pieces, num_pieces)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    labels = labels.astype(np.int32, copy=False)
    for component in np.unique(labels):
        mask = labels == component
        if np.count_nonzero(mask) == 1:
            labels[mask] = -1
    valid = [int(x) for x in np.unique(labels) if int(x) >= 0]
    remap = {old: new for new, old in enumerate(valid)}
    return np.asarray([remap.get(int(x), -1) for x in labels], dtype=np.int32)


def get_connected_from_solution(
    coordinates: np.ndarray,
    proposed: MatchInfo,
    num_pieces: int,
    *,
    previous: MatchInfo | None = None,
    buddy_check: bool = True,
    loop_check: bool = True,
    tolerance: float = 1e-4,
) -> tuple[np.ndarray, MatchInfo]:
    info = proposed
    if previous is not None:
        info = unique_info(concatenate_info(proposed, previous))
    kept = info.subset(check_match_lp(info, coordinates, tolerance))
    if buddy_check:
        kept = info_filter(kept)
    if loop_check:
        kept = info_filter_loop(kept)
    return labels_from_info(num_pieces, kept), kept


def _crop(array: np.ndarray) -> np.ndarray:
    occupied = np.argwhere(array != 0)
    if occupied.size == 0:
        return np.empty((0, 0), dtype=array.dtype)
    minimum = occupied.min(axis=0)
    maximum = occupied.max(axis=0) + 1
    return array[minimum[0] : maximum[0], minimum[1] : maximum[1]]


def recover_connected_blocks(
    labels: np.ndarray,
    coordinates: np.ndarray,
    num_pieces: int,
) -> tuple[np.ndarray, BlocksInfo]:
    """从连续 LP 坐标恢复无冲突的四连通二维刚性块；piece id 在块内保持 MATLAB 的 1 基表示。"""

    blocks: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    for component in [int(x) for x in np.unique(labels) if int(x) >= 0]:
        members = np.flatnonzero(labels == component)
        rows = coordinates[members, 0]
        columns = coordinates[members, 1]
        min_row, min_column = float(rows.min()), float(columns.min())
        height = int(round(float(rows.max()) - min_row + 1.0))
        width = int(round(float(columns.max()) - min_column + 1.0))
        if height <= 0 or width <= 0:
            continue
        block = np.zeros((height, width), dtype=np.int32)
        for row in range(height):
            for column in range(width):
                target_row = min_row + row
                target_column = min_column + column
                mask = (
                    (np.abs(rows - target_row) < 0.1)
                    & (np.abs(columns - target_column) < 0.1)
                )
                if np.count_nonzero(mask) == 1:
                    block[row, column] = int(members[np.flatnonzero(mask)[0]]) + 1
                elif np.count_nonzero(mask) > 1:
                    block[row, column] = -int(np.count_nonzero(mask))

        spatial_labels, count = image_label(block > 0, structure=structure)
        for spatial_component in range(1, count + 1):
            child = block.copy()
            child[spatial_labels != spatial_component] = 0
            child = _crop(child)
            # MATLAB 最终删除单 piece block；这里按其注释表达的语义执行。
            if np.count_nonzero(child > 0) <= 1:
                continue
            blocks.append(child)
            rotations.append((child > 0).astype(np.int32))

    recovered_labels = np.full(num_pieces, -1, dtype=np.int32)
    for component, block in enumerate(blocks):
        pieces = block[block > 0] - 1
        recovered_labels[pieces] = component
    return recovered_labels, BlocksInfo(blocks, rotations)


def sco_update(
    scores: np.ndarray,
    blocks_info: BlocksInfo,
    labels: np.ndarray,
    *,
    remove_boundary: bool = True,
) -> np.ndarray:
    """复现 ``SCO_update(..., method='conneced_comp')`` 的 Type-1 socket 淘汰。"""

    updated = np.asarray(scores, dtype=np.float64).copy()
    n = updated.shape[0]
    for component in [int(x) for x in np.unique(labels) if int(x) >= 0]:
        members = np.flatnonzero(labels == component)
        updated[np.ix_(members, members, np.arange(4))] = np.inf

    for block in blocks_info.blocks:
        good_size = block.size == n
        for direction in range(4):
            if direction == 0:
                first, second, boundary = block[1:, :], block[:-1, :], block[0, :]
            elif direction == 1:
                first, second, boundary = block[:, :-1], block[:, 1:], block[:, -1]
            elif direction == 2:
                first, second, boundary = block[:-1, :], block[1:, :], block[-1, :]
            else:
                first, second, boundary = block[:, 1:], block[:, :-1], block[:, 0]
            valid = (first > 0) & (second > 0)
            for first_id, second_id in zip(first[valid] - 1, second[valid] - 1):
                first_id, second_id = int(first_id), int(second_id)
                mirror = int(MIRROR[direction])
                updated[first_id, :, direction] = np.inf
                updated[:, first_id, mirror] = np.inf
                updated[:, second_id, direction] = np.inf
                updated[second_id, :, mirror] = np.inf
            if good_size and remove_boundary:
                for piece_id in boundary[boundary > 0] - 1:
                    piece_id = int(piece_id)
                    mirror = int(MIRROR[direction])
                    updated[piece_id, :, direction] = np.inf
                    updated[:, piece_id, mirror] = np.inf
    return updated


__all__ = [
    "BlocksInfo",
    "blocks_info_cut_type1",
    "check_match_lp",
    "get_connected_from_solution",
    "labels_from_info",
    "recover_connected_blocks",
    "sco_update",
]
