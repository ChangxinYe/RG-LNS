"""
@file: mgc.py
@description: PuzzleDemoMGC CVPR 2012 的 Python 核心实现，包含切块、边界特征计算、MGC 兼容分数、贪心拼装、补洞、评估和可视化。
@author: Changxin Ye
@created: 2026-07-06
@version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import convolve


@dataclass
class EdgeAttributes:
    pix: np.ndarray
    pix_r: np.ndarray
    d_mu: np.ndarray
    d_cov: np.ndarray


@dataclass
class DemoResult:
    gi: np.ndarray
    gr: np.ndarray
    solved_image: np.ndarray
    scrambled_before_damage: np.ndarray
    scrambled_after_damage: np.ndarray
    results: np.ndarray
    position_correct: int
    orientation_correct: int
    orientation_checked: int
    position_key: np.ndarray
    random_rotation_key: np.ndarray


def placement_key(nr: int, nc: int) -> np.ndarray:
    return np.arange(1, nr * nc + 1, dtype=np.int32).reshape((nr, nc), order="F")


def _pil_resize(arr: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return np.asarray(img.resize((w, h), resampling), dtype=np.uint8)


def puzzle_fun(image_path: str | Path, pp: int, wide_p: int, high_p: int):
    im = Image.open(image_path).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)

    src_h, src_w = arr.shape[:2]
    new_wide = min(src_w, int(np.floor(src_h / high_p * wide_p)))
    new_high = min(src_h, int(np.floor(src_w / wide_p * high_p)))

    if new_high < src_h:
        start = int(np.round((src_h - new_high) / 2.0))
        arr = arr[start : start + new_high, :, :]
    elif new_wide < src_w:
        start = int(np.round((src_w - new_wide) / 2.0))
        arr = arr[:, start : start + new_wide, :]

    pix_w = pp * wide_p
    pix_h = pp * high_p
    resized = _pil_resize(arr, (pix_h, pix_w))

    pieces: list[np.ndarray] = []
    piece_info = []
    piece_id = 1
    for col in range(wide_p):
        for row in range(high_p):
            piece = resized[row * pp : (row + 1) * pp, col * pp : (col + 1) * pp, :]
            pieces.append(piece.copy())

            above = np.nan if row == 0 else piece_id - 1
            right = np.nan if col == wide_p - 1 else piece_id + high_p
            below = np.nan if row == high_p - 1 else piece_id + 1
            left = np.nan if col == 0 else piece_id - high_p
            piece_info.append([piece_id, col + 1, row + 1, above, right, below, left])
            piece_id += 1

    return None, pieces, np.asarray(piece_info, dtype=float), None


def damage_piece_borders(pieces: list[np.ndarray], damage_pixels: int) -> list[np.ndarray]:
    damage_pixels = int(round(damage_pixels))
    if damage_pixels < 0:
        raise ValueError("damage_pixels must be nonnegative")
    if damage_pixels == 0:
        return [p.copy() for p in pieces]

    damaged = []
    for idx, piece in enumerate(pieces, start=1):
        h, w = piece.shape[:2]
        if damage_pixels * 2 >= min(h, w):
            raise ValueError(f"damage_pixels is too large for piece {idx}")
        inner = piece[damage_pixels : h - damage_pixels, damage_pixels : w - damage_pixels, :]
        damaged.append(_pil_resize(inner, (h, w)))
    return damaged


def compute_piece_attributes(pieces: list[np.ndarray]) -> list[EdgeAttributes]:
    dummy_diffs = np.asarray(
        [
            [0, 0, 0],
            [1, 1, 1],
            [-1, -1, -1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=np.float64,
    )
    n = len(pieces)
    psize = pieces[0].shape[0]
    attrs = [
        EdgeAttributes(
            pix=np.zeros((n, psize * 3), dtype=np.float64),
            pix_r=np.zeros((n, psize * 3), dtype=np.float64),
            d_mu=np.zeros((n, 3), dtype=np.float64),
            d_cov=np.zeros((n, 3, 3), dtype=np.float64),
        )
        for _ in range(4)
    ]

    for w, piece_u8 in enumerate(pieces):
        piece = piece_u8.astype(np.float64)
        for edge in range(4):
            if edge == 0:
                s = piece[0, :, :]
                r = np.flipud(s)
                dif = piece[0, :, :] - piece[1, :, :]
            elif edge == 1:
                s = piece[:, -1, :]
                r = np.flipud(s)
                dif = piece[:, -1, :] - piece[:, -2, :]
            elif edge == 2:
                r = piece[-1, :, :]
                s = np.flipud(r)
                dif = piece[-1, :, :] - piece[-2, :, :]
            else:
                r = piece[:, 0, :]
                s = np.flipud(r)
                dif = piece[:, 0, :] - piece[:, 1, :]

            attrs[edge].pix[w, :] = s.reshape(-1, order="F")
            attrs[edge].pix_r[w, :] = r.reshape(-1, order="F")
            attrs[edge].d_mu[w, :] = dif.mean(axis=0)
            attrs[edge].d_cov[w, :, :] = np.cov(
                np.vstack([dif, dummy_diffs]), rowvar=False, bias=False
            )

    return attrs


def _right_divide(x: np.ndarray, cov: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(cov.T, x.T).T
    except np.linalg.LinAlgError:
        return x @ np.linalg.pinv(cov)


def compare_piece_pair_a_rot(
    pieces: list[np.ndarray],
    mode: int,
    attrs: list[EdgeAttributes],
    rot_flag: int = 1,
) -> np.ndarray:
    n = len(pieces)
    layers = 16 if rot_flag else 4
    score = np.zeros((n, n, layers), dtype=np.float32)
    psize = pieces[0].shape[0]

    opp_side = np.asarray(
        [
            [3, 4, 1, 2],
            [4, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 1],
        ],
        dtype=np.int32,
    )
    mirror = np.asarray(
        [3, 4, 1, 2, 16, 13, 14, 15, 9, 10, 11, 12, 6, 7, 8, 5],
        dtype=np.int32,
    )

    inv_cov = []
    for edge_attr in attrs:
        invs = np.empty_like(edge_attr.d_cov)
        for i in range(n):
            invs[i] = np.linalg.pinv(edge_attr.d_cov[i])
        inv_cov.append(invs)

    for fit in range(4):
        print(f"Fit Number: {fit + 1}")
        for i in range(n - 1):
            p1_mu = attrs[fit].d_mu[i]
            p1_inv = inv_cov[fit][i]
            p1_strip = attrs[fit].pix[i]
            score[i, i, :] = np.inf

            for rr in range(0, rot_flag * 3 + 1):
                opposite = int(opp_side[rr, fit] - 1)
                layer = fit + rr * 4
                ks = np.arange(i + 1, n)
                if ks.size == 0:
                    continue

                p2_mu = attrs[opposite].d_mu[ks]
                p2_inv = inv_cov[opposite][ks]
                p2_strip = attrs[opposite].pix_r[ks]

                dif_flat = p1_strip[None, :] - p2_strip
                dif = dif_flat.reshape((ks.size, psize, 3), order="F")
                rev_dif = -dif

                centered12 = dif - p2_mu[:, None, :]
                d12 = np.einsum("mpi,mij,mpj->mp", centered12, p2_inv, centered12)
                d12 = np.sqrt(np.maximum(d12, 0.0))

                centered21 = rev_dif - p1_mu[None, None, :]
                d21 = np.einsum("mpi,ij,mpj->mp", centered21, p1_inv, centered21)
                d21 = np.sqrt(np.maximum(d21, 0.0))

                dist = (d12 + d21).sum(axis=1)
                if mode == 3:
                    dist = (dif * dif).sum(axis=(1, 2))

                score[i, ks, layer] = dist.astype(np.float32)
                score[ks, i, mirror[layer] - 1] = dist.astype(np.float32)

    score[n - 1, n - 1, :] = np.inf
    return score


def _as_block(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.astype(np.int32, copy=True)
        if arr.ndim == 0:
            return arr.reshape((1, 1))
        if arr.ndim == 1:
            return arr.reshape((1, -1))
        return arr
    return np.asarray([[x]], dtype=np.int32)


def join_pieces_r(b1, b2, r1, r2, p1: int, p2: int, how: int):
    b1 = _as_block(b1)
    b2 = _as_block(b2)
    r1 = _as_block(r1)
    r2 = _as_block(r2)

    b1nz = b1[b1 > 0]
    b2nz = b2[b2 > 0]
    if np.intersect1d(b1nz, b2nz).size > 0:
        return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.int32), False

    ccw_to_get_to = np.asarray(
        [[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]],
        dtype=np.int32,
    )
    rot_to_cw = np.asarray(
        [[1, 4, 3, 2], [2, 1, 4, 3], [3, 2, 1, 4], [4, 3, 2, 1]],
        dtype=np.int32,
    )
    rot_to_ccw = np.asarray(
        [[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]],
        dtype=np.int32,
    )

    how_pos = (how - 1) % 4 + 1
    how_rot = (how - 1) // 4 + 1

    loc1 = np.argwhere(b1 == p1)
    loc2 = np.argwhere(b2 == p2)
    if loc1.size == 0 or loc2.size == 0:
        return np.empty((0, 0), dtype=np.int32), np.empty((0, 0), dtype=np.int32), False
    r1p, c1p = loc1[0]
    r2p, c2p = loc2[0]

    rot_needed = int(r1[r1p, c1p] - 1)
    b1n = np.rot90(b1, k=-rot_needed)
    r1n = np.rot90(r1, k=-rot_needed)
    rtrans = np.concatenate([[0], rot_to_cw[:, rot_needed]])
    r1n = rtrans[r1n]

    rot_now2 = int(r2[r2p, c2p])
    rot_needed = int(ccw_to_get_to[rot_now2 - 1, how_rot - 1])
    b2n = np.rot90(b2, k=rot_needed)
    r2n = np.rot90(r2, k=rot_needed)
    rtrans = np.concatenate([[0], rot_to_ccw[:, rot_needed]])
    r2n = rtrans[r2n]

    b1 = b1n
    b2 = b2n
    loc1 = np.argwhere(b1 == p1)
    loc2 = np.argwhere(b2 == p2)
    r1p, c1p = loc1[0]
    r2p, c2p = loc2[0]

    offsets = np.asarray([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int32)
    offset = offsets[how_pos - 1]
    b2_key_in_b1 = np.asarray([r1p, c1p], dtype=np.int32) + offset
    c2_to_1 = b2_key_in_b1 - np.asarray([r2p, c2p], dtype=np.int32)

    ul_b2 = c2_to_1
    lr_b2 = np.asarray(b2.shape, dtype=np.int32) - 1 + c2_to_1
    all_coords = np.vstack(
        [
            [0, 0],
            np.asarray(b1.shape, dtype=np.int32) - 1,
            ul_b2,
            lr_b2,
        ]
    )
    ranges_min = all_coords.min(axis=0)
    ranges_max = all_coords.max(axis=0)
    b1_offset = -ranges_min
    b2_offset = b1_offset + c2_to_1
    out_shape = tuple((ranges_max - ranges_min + 1).tolist())

    out_block = np.zeros(out_shape, dtype=np.int32)
    out_rot = np.zeros(out_shape, dtype=np.int32)

    rr1, cc1 = b1.shape
    rr2, cc2 = b2.shape
    r0, c0 = b1_offset
    out_block[r0 : r0 + rr1, c0 : c0 + cc1] = b1
    temp = out_block.copy()
    r0, c0 = b2_offset
    temp[r0 : r0 + rr2, c0 : c0 + cc2] = b2
    out_block[temp > 0] = temp[temp > 0]

    r0, c0 = b1_offset
    out_rot[r0 : r0 + rr1, c0 : c0 + cc1] = r1n
    temp = out_rot.copy()
    r0, c0 = b2_offset
    temp[r0 : r0 + rr2, c0 : c0 + cc2] = r2n
    out_rot[temp > 0] = temp[temp > 0]

    success = int((out_block > 0).sum()) == int((b1 > 0).sum() + (b2 > 0).sum())
    return out_block, out_rot, bool(success)


def _unravel_f(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    return np.unravel_index(index, shape, order="F")


def greedy_assembly_5r(pieces: list[np.ndarray], sco: np.ndarray, nr: int, nc: int):
    norm_sco = sco.copy()
    n = norm_sco.shape[0]
    mirror = np.asarray([3, 4, 1, 2, 16, 13, 14, 15, 9, 10, 11, 12, 6, 7, 8, 5], dtype=np.int32)
    tiny = 1e-15

    for layer_idx in range(norm_sco.shape[2]):
        print(f"Processing Scores Matrix {layer_idx + 1}")
        layer = sco[:, :, layer_idx]
        row_order = np.argsort(layer, axis=1, kind="stable")
        row_sorted = np.take_along_axis(layer, row_order, axis=1)
        rowmins = row_sorted[:, :2]
        rowminloc = row_order[:, 0]

        col_order = np.argsort(layer, axis=0, kind="stable")
        col_sorted = np.take_along_axis(layer, col_order, axis=0)
        colmins = col_sorted[:2, :]
        colminloc = col_order[0, :]

        for row in range(n):
            values = layer[row, :]
            with np.errstate(invalid="ignore"):
                n1 = values * 0 + rowmins[row, 0]
                n2 = values * 0 + colmins[0, :]
            n1[rowminloc] = rowmins[row, 1]
            mask = row == colminloc
            n2[mask] = colmins[1, mask]
            norm_sco[row, :, layer_idx] = (values + tiny) / (np.minimum(n1, n2) + tiny)

    blocks: list[np.ndarray | None] = []
    rots: list[np.ndarray | None] = []
    hits = np.asarray([0, 0], dtype=np.int32)
    st = 1.25
    iters = 1

    while True:
        if np.all(np.isnan(norm_sco)):
            break
        flat = norm_sco.ravel(order="F")
        try:
            bb = int(np.nanargmin(flat))
        except ValueError:
            break
        aa = float(flat[bb])
        stop_after_this = np.isnan(aa) or aa > st

        r_idx, c_idx, how_idx = _unravel_f(bb, norm_sco.shape)
        p1 = r_idx + 1
        p2 = c_idx + 1
        how = how_idx + 1

        rb = p1
        cb = p2
        rr = 1
        cr = 1
        findr = None
        findc = None

        for idx, block in enumerate(blocks):
            if block is not None and np.any(block == p1):
                findr = idx
                rb = block
                rr = rots[idx]
            if block is not None and np.any(block == p2):
                findc = idx
                cb = block
                cr = rots[idx]

        if findr is not None and findc is not None:
            if findr != findc:
                b, rot, success = join_pieces_r(rb, cb, rr, cr, p1, p2, how)
                bsize = int((b > 0).sum())
            else:
                success = False
                bsize = 0
                b = np.empty((0, 0), dtype=np.int32)
                rot = np.empty((0, 0), dtype=np.int32)
        else:
            b, rot, success = join_pieces_r(rb, cb, rr, cr, p1, p2, how)
            bsize = int((b > 0).sum())

        hits += np.asarray([success == 1, success == 0], dtype=np.int32)

        if success:
            if findr is not None:
                blocks[findr] = b
                rots[findr] = rot
                if findc is not None:
                    blocks[findc] = None
                    rots[findc] = None
            elif findc is not None:
                blocks[findc] = b
                rots[findc] = rot
            else:
                blocks.append(b)
                rots.append(rot)

            norm_sco[p1 - 1, :, how - 1] = np.nan
            norm_sco[:, p2 - 1, how - 1] = np.nan
            how_n = int(mirror[how - 1])
            norm_sco[p2 - 1, :, how_n - 1] = np.nan
            norm_sco[:, p1 - 1, how_n - 1] = np.nan

            rb1 = np.asarray(rb).reshape(-1)
            cb1 = np.asarray(cb).reshape(-1)
            rb1 = rb1[rb1 > 0]
            cb1 = cb1[cb1 > 0]
            if rb1.size > 1 or cb1.size > 1:
                for id1 in rb1:
                    for id2 in cb1:
                        norm_sco[id1 - 1, id2 - 1, :] = np.nan
                        norm_sco[id2 - 1, id1 - 1, :] = np.nan
        else:
            how_n = int(mirror[how - 1])
            norm_sco[p1 - 1, p2 - 1, how - 1] = np.nan
            norm_sco[p2 - 1, p1 - 1, how_n - 1] = np.nan

        if bsize > n - 1:
            break
        if stop_after_this:
            break

        iters += 1
        if iters % 100 == 0:
            remaining = int(np.sum(norm_sco > 0))
            print(f"{iters}\t{p1} {p2}\t{how} {bsize} {len(blocks)} {remaining}")
            print(f"{aa:.2f}")

    sizes = [0 if b is None else int((b > 0).sum()) for b in blocks]
    if not sizes or max(sizes) == 0:
        gi = np.zeros((nr, nc), dtype=np.int32)
        gr = np.zeros((nr, nc), dtype=np.int32)
    else:
        best = int(np.argmax(sizes))
        gi = blocks[best]
        gr = rots[best]

    ap_rot = [p.copy() for p in pieces]
    for pos in np.argwhere(gr > 0):
        rv = int(gr[tuple(pos)])
        bid = int(gi[tuple(pos)])
        ap_rot[bid - 1] = np.rot90(ap_rot[bid - 1], k=rv - 1)

    image, graph_ids = render_pieces_from_graph_ids(ap_rot, gi, 0)
    res = evaluate_puzzle_assembly(graph_ids, nc, nr)
    return graph_ids, image, gr, res, hits


def trim_puzzle(block: np.ndarray, rot: np.ndarray, nr: int, nc: int, rot_flag: int):
    ss = block.shape
    new_block = block.copy()
    new_rot = rot.copy()

    if max(ss) <= max(nr, nc) and min(ss) <= min(nr, nc):
        return new_block, new_rot

    row_marg = (block > 0).sum(axis=1)
    col_marg = (block > 0).sum(axis=0)
    total1, rs1, cs1 = _find_best_crop(row_marg, col_marg, nr, nc)
    total2, rs2, cs2 = _find_best_crop(row_marg, col_marg, nc, nr)

    if total1 > 0 and (total1 < total2 or rot_flag == 0):
        new_block = block[rs1 : min(rs1 + nr, ss[0]), cs1 : min(cs1 + nc, ss[1])]
        new_rot = rot[rs1 : min(rs1 + nr, ss[0]), cs1 : min(cs1 + nc, ss[1])]
    elif total2 > 0 and (total2 < total1 and rot_flag == 1):
        new_block = block[rs2 : min(rs2 + nc, ss[0]), cs2 : min(cs2 + nr, ss[1])]
        new_rot = rot[rs2 : min(rs2 + nc, ss[0]), cs2 : min(cs2 + nr, ss[1])]

    return new_block, new_rot


def _find_best_crop(row_marg: np.ndarray, col_marg: np.ndarray, nr: int, nc: int):
    total_chopped = 0
    total_pieces = int(row_marg.sum())
    rs = 0
    cs = 0

    if row_marg.size > nr:
        kept = np.asarray([row_marg[i : i + nr].sum() for i in range(row_marg.size + 1 - nr)])
        best = int(np.argmax(kept))
        total_chopped += total_pieces - int(kept[best])
        rs = best

    if col_marg.size > nc:
        kept = np.asarray([col_marg[i : i + nc].sum() for i in range(col_marg.size + 1 - nc)])
        best = int(np.argmax(kept))
        total_chopped += total_pieces - int(kept[best])
        cs = best

    return total_chopped, rs, cs


def _find_zero_positions_col_major(block: np.ndarray):
    flat = np.flatnonzero(block.ravel(order="F") == 0)
    if flat.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)
    return np.unravel_index(flat, block.shape, order="F")


def get_all_scores_for_cands(
    sco: np.ndarray,
    cands: np.ndarray,
    piece_id: int,
    current_piece_rot: int,
    relposition: int,
    rot_flag: int,
):
    del current_piece_rot, relposition
    ss = sco.shape[2] * rot_flag + 4 * (1 - rot_flag)
    score_mat = np.zeros((len(cands), ss), dtype=sco.dtype)
    for i, cand in enumerate(cands):
        score_mat[i, :] = sco[int(cand) - 1, int(piece_id) - 1, :ss]
    return score_mat


def fill_puzzle_holes_v2(block: np.ndarray, rot: np.ndarray, sco: np.ndarray, nr: int, nc: int, rot_flag: int):
    new_block = block.copy()
    new_rot = rot.copy()
    kernel = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)

    def recompute():
        choices_local = np.setdiff1d(np.arange(1, nr * nc + 1, dtype=np.int32), new_block.reshape(-1), assume_unique=False)
        filled = new_block > 0
        nei_num_local = convolve(filled.astype(np.uint8), kernel, mode="constant", cval=0)
        rr_local, cc_local = _find_zero_positions_col_major(new_block)
        if rr_local.size == 0:
            return choices_local, nei_num_local, rr_local, cc_local, np.asarray([], dtype=np.int64)
        nn_local = nei_num_local[rr_local, cc_local].astype(np.int32, copy=False)
        order = np.argsort(-nn_local, kind="stable")
        return choices_local, nei_num_local, rr_local[order], cc_local[order], order

    choices, nei_num, rr_sorted, cc_sorted, order = recompute()
    failcnt = 0
    failt = 10
    gogo = True

    while gogo:
        if order.size == 0:
            break

        spot = np.asarray([rr_sorted[0], cc_sorted[0]], dtype=np.int32)
        neighbors = np.asarray(
            [
                [spot[0] - 1, spot[1]],
                [spot[0], spot[1] - 1],
                [spot[0], spot[1] + 1],
                [spot[0] + 1, spot[1]],
            ],
            dtype=np.int32,
        )
        nei_code = np.asarray([1, 4, 2, 3], dtype=np.int32)
        good = (
            (neighbors[:, 0] >= 0)
            & (neighbors[:, 1] >= 0)
            & (neighbors[:, 0] < new_block.shape[0])
            & (neighbors[:, 1] < new_block.shape[1])
        )
        neighbors = neighbors[good]
        nei_code = nei_code[good]

        if neighbors.size == 0:
            break
        nei_ids = new_block[neighbors[:, 0], neighbors[:, 1]]
        nei_rots = new_rot[neighbors[:, 0], neighbors[:, 1]]
        good = nei_ids > 0
        nei_ids = nei_ids[good]
        nei_rots = nei_rots[good]
        nei_code = nei_code[good]

        if neighbors.size == 0 or nei_ids.size == 0:
            break

        total_scores = np.zeros(
            (len(choices), sco.shape[2] * rot_flag + 4 * (1 - rot_flag), len(nei_ids)),
            dtype=sco.dtype,
        )
        for idx in range(len(nei_ids)):
            total_scores[:, :, idx] = get_all_scores_for_cands(
                sco, choices, int(nei_ids[idx]), int(nei_rots[idx]), int(nei_code[idx]), rot_flag
            )

        flat = total_scores.ravel(order="F")
        sorted_idx = np.argsort(flat, kind="stable")
        gogo = False
        for flat_idx in sorted_idx:
            cand_idx, rot_idx, nei_idx = _unravel_f(int(flat_idx), total_scores.shape)
            choice = int(choices[cand_idx])
            choice_rot = int(rot_idx + 1)
            nei = int(nei_ids[nei_idx])

            b, r, success = join_pieces_r(choice, new_block, 1, new_rot, choice, nei, choice_rot)
            holes_before = int((new_block == 0).sum())
            holes_now = int((b == 0).sum())
            if success and holes_now == holes_before - 1:
                new_block = b
                new_rot = r
                choices, nei_num, rr_sorted, cc_sorted, order = recompute()
                gogo = True
                break

            failcnt += 1
            if failcnt > failt:
                gogo = False

    return new_block, new_rot


def _apply_position_key(graph: np.ndarray, position_key: np.ndarray) -> np.ndarray:
    safe = graph.copy()
    undo = safe == 0
    safe[undo] = 1
    out = position_key[safe - 1]
    out[undo] = 0
    return out


def do_all_assembly_of_puzzle(
    pieces: list[np.ndarray],
    sco: np.ndarray,
    nr: int,
    nc: int,
    rot_flag: int = 1,
    position_key: np.ndarray | None = None,
):
    if position_key is None:
        position_key = np.random.permutation(np.arange(1, nr * nc + 1, dtype=np.int32))
    else:
        position_key = np.asarray(position_key, dtype=np.int32)

    g_raw, initial_image, gr = greedy_assembly_5r(pieces, sco, nr, nc)[:3]
    print("Start Trimming")
    new_block, new_rot = trim_puzzle(g_raw, gr, nr, nc, rot_flag)
    new_block_filled, new_rot_filled = fill_puzzle_holes_v2(new_block, new_rot, sco, nr, nc, rot_flag)

    holes = int((g_raw == 0).sum())
    holes_now = int((new_block_filled == 0).sum())
    if holes_now != holes:
        print(f"Hole Filling: was: {holes} is: {holes_now} ")
        g_work = new_block_filled
        gr_work = new_rot_filled
    else:
        g_work = g_raw
        gr_work = gr

    results = evaluate_puzzle_assembly(_apply_position_key(g_work, position_key), nc, nr)

    if rot_flag:
        rot_to_ccw = np.asarray(
            [[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]],
            dtype=np.int32,
        )
        best_correct = 0
        g_best = g_work
        r_best = gr_work
        for ii in range(4):
            lrot = np.concatenate([[0], rot_to_ccw[ii]])
            g_temp = np.rot90(g_work, k=ii)
            gr_temp = lrot[np.rot90(gr_work, k=ii)]
            res = evaluate_puzzle_assembly(_apply_position_key(g_temp, position_key), nc, nr)
            if res[2] > best_correct:
                g_best = g_temp
                r_best = gr_temp
                best_correct = int(res[2])
                results = res
    else:
        g_best = g_work
        r_best = gr_work

    rotated = [p.copy() for p in pieces]
    for pos in np.argwhere(r_best > 0):
        rv = int(r_best[tuple(pos)])
        bid = int(g_best[tuple(pos)])
        rotated[bid - 1] = np.rot90(pieces[bid - 1], k=rv - 1)
    image, gi = render_pieces_from_graph_ids(rotated, g_best, 0)
    return gi, r_best, image, results


def render_pieces_from_graph_ids(pieces: list[np.ndarray], graph_ids: np.ndarray, gap: int = 0):
    graph_ids = np.asarray(graph_ids, dtype=np.int32).copy()
    if graph_ids.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8), graph_ids

    ccc = graph_ids.sum(axis=0)
    rrr = graph_ids.sum(axis=1)
    if graph_ids.shape[0] >= 2 and rrr[0] == 0 and rrr[-1] == 0:
        graph_ids = graph_ids[1:-1, :]
        rrr = graph_ids.sum(axis=1) if graph_ids.size else np.asarray([])
    if graph_ids.shape[1] >= 2 and ccc[0] == 0 and ccc[-1] == 0:
        graph_ids = graph_ids[:, 1:-1]

    pp = pieces[0].shape[0]
    rows, cols = graph_ids.shape
    out = np.zeros((rows * pp + (rows - 1) * gap, cols * pp + (cols - 1) * gap, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            pid = int(graph_ids[r, c])
            if pid > 0:
                rr = r * (pp + gap)
                cc = c * (pp + gap)
                out[rr : rr + pp, cc : cc + pp, :] = pieces[pid - 1].astype(np.uint8)
    return out, graph_ids


def _truth_position_map(wide: int, high: int) -> dict[int, tuple[int, int]]:
    car = placement_key(high, wide)
    return {int(car[r, c]): (r, c) for r in range(car.shape[0]) for c in range(car.shape[1])}


def _is_true_neighbor(a: int, b: int, dr: int, dc: int, truth_pos: dict[int, tuple[int, int]]) -> bool:
    if a <= 0 or b <= 0 or a not in truth_pos or b not in truth_pos:
        return False
    ra, ca = truth_pos[a]
    rb, cb = truth_pos[b]
    return rb == ra + dr and cb == ca + dc


def evaluate_puzzle_assembly(graph_ids: np.ndarray, wide: int, high: int) -> np.ndarray:
    graph_ids = np.asarray(graph_ids, dtype=np.int32)
    car = placement_key(high, wide)
    min_r = min(car.shape[0], graph_ids.shape[0])
    min_c = min(car.shape[1], graph_ids.shape[1])
    score_abs = int((car[:min_r, :min_c] == graph_ids[:min_r, :min_c]).sum())

    truth_pos = _truth_position_map(wide, high)
    correct_pairwise = 0
    for r in range(max(graph_ids.shape[0] - 1, 0)):
        for c in range(graph_ids.shape[1]):
            if _is_true_neighbor(int(graph_ids[r, c]), int(graph_ids[r + 1, c]), 1, 0, truth_pos):
                correct_pairwise += 1
    for r in range(graph_ids.shape[0]):
        for c in range(max(graph_ids.shape[1] - 1, 0)):
            if _is_true_neighbor(int(graph_ids[r, c]), int(graph_ids[r, c + 1]), 0, 1, truth_pos):
                correct_pairwise += 1

    visited = np.zeros(graph_ids.shape, dtype=bool)
    max_blob = 0
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for r in range(graph_ids.shape[0]):
        for c in range(graph_ids.shape[1]):
            if graph_ids[r, c] <= 0 or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            blob = 0
            while stack:
                cr, cc = stack.pop()
                blob += 1
                current = int(graph_ids[cr, cc])
                for dr, dc in dirs:
                    nr0 = cr + dr
                    nc0 = cc + dc
                    if nr0 < 0 or nc0 < 0 or nr0 >= graph_ids.shape[0] or nc0 >= graph_ids.shape[1]:
                        continue
                    if visited[nr0, nc0] or graph_ids[nr0, nc0] <= 0:
                        continue
                    if _is_true_neighbor(current, int(graph_ids[nr0, nc0]), dr, dc, truth_pos):
                        visited[nr0, nc0] = True
                        stack.append((nr0, nc0))
            max_blob = max(max_blob, blob)

    return np.asarray([score_abs, correct_pairwise, max_blob], dtype=np.int32)


def orientation_score(gi: np.ndarray, gr: np.ndarray, randscram: np.ndarray, n_pieces: int):
    piece_rot = np.zeros(n_pieces, dtype=np.int32)
    valid = (gi > 0) & (gr > 0)
    piece_rot[gi[valid] - 1] = gr[valid]
    checked = int((piece_rot > 0).sum())
    if checked == 0:
        return 0, 0
    lut = np.asarray([1, 4, 3, 2], dtype=np.int32)
    idx = piece_rot > 0
    correct = int((lut[piece_rot[idx] - 1] == randscram[idx]).sum())
    return correct, checked


def save_image(path: str | Path, image: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path, quality=99)


def run_demo(
    image_path: str | Path,
    pp: int,
    nc: int,
    nr: int,
    damage_pixels: int = 0,
    scramble_positions: bool = True,
    scramble_rotations: bool = True,
    seed: int | None = None,
    output_dir: str | Path | None = None,
) -> DemoResult:
    rng = np.random.default_rng(seed)
    _, pieces, _, _ = puzzle_fun(image_path, pp, nc, nr)

    if scramble_positions:
        position_key = rng.permutation(np.arange(1, len(pieces) + 1, dtype=np.int32))
        pieces = [pieces[i - 1] for i in position_key]
    else:
        position_key = np.arange(1, len(pieces) + 1, dtype=np.int32)

    if scramble_rotations:
        randscram = rng.integers(1, 5, size=len(pieces), dtype=np.int32)
        pieces = [np.rot90(piece, k=int(rot) - 1) for piece, rot in zip(pieces, randscram)]
    else:
        randscram = np.ones(len(pieces), dtype=np.int32)

    before_image, _ = render_pieces_from_graph_ids(pieces, placement_key(nr, nc), 0)
    damaged_pieces = damage_piece_borders(pieces, damage_pixels)
    after_image, _ = render_pieces_from_graph_ids(damaged_pieces, placement_key(nr, nc), 0)

    if output_dir is not None:
        out_dir = Path(output_dir)
        save_image(out_dir / "ScrambledDemoBeforeBoundaryDamage.jpg", before_image)
        save_image(out_dir / "ScrambledDemoAfterBoundaryDamage.jpg", after_image)

    print("Compute Attributes for Puzzle")
    attrs = compute_piece_attributes(damaged_pieces)
    print("Done with Compute Attributes for Puzzle")
    print("Compute Pairwise Compatibility Scores for Puzzle, Method 7")
    sco = compare_piece_pair_a_rot(damaged_pieces, 7, attrs, int(scramble_rotations))
    print("Done with Score Computation for Puzzle Method 7")

    gi, gr, solved, results = do_all_assembly_of_puzzle(
        damaged_pieces, sco, nr, nc, int(scramble_rotations), position_key
    )

    if output_dir is not None:
        save_image(Path(output_dir) / "SolvedDemo.jpg", solved)

    g_safe = gi.copy()
    undo = g_safe == 0
    g_safe[undo] = 1
    g_temp = position_key[g_safe - 1]
    g_temp[undo] = 0
    eval_results = evaluate_puzzle_assembly(g_temp, nc, nr)
    rot_correct, rot_checked = orientation_score(gi, gr, randscram, len(damaged_pieces))

    print("")
    print(f"Accuracy: {int(eval_results[2])} pieces of {nr * nc} are exactly in the correct position.")
    print(
        f"Orientation: {rot_correct} pieces have correct orientation "
        f"({rot_checked} of {nr * nc} pieces checked)."
    )

    return DemoResult(
        gi=gi,
        gr=gr,
        solved_image=solved,
        scrambled_before_damage=before_image,
        scrambled_after_damage=after_image,
        results=eval_results,
        position_correct=int(eval_results[2]),
        orientation_correct=rot_correct,
        orientation_checked=rot_checked,
        position_key=position_key,
        random_rotation_key=randscram,
    )
