"""
@file: metric_lsej_data.py
@description: 基于官方 ImageNet-LSEJ DataLoader 的 s1a 多三元组采样与同图候选负样本数据适配器。
@author: Changxin Ye
@created: 2026-07-17
@version: 1.1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .metric_geometry import DIRECTIONS, canonical_piece, relation_candidate_edge
    from .metric_transforms import piece_to_tensor
except ImportError:
    from metric_geometry import DIRECTIONS, canonical_piece, relation_candidate_edge
    from metric_transforms import piece_to_tensor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OFFICIAL_LSEJ_DIR = PROJECT_ROOT / "datasets" / "ImageNet_LSEJ" / "ImageNet_LSEJ"
DEFAULT_LSEJ_DATA_ROOT = OFFICIAL_LSEJ_DIR
OFFICIAL_TASKS = (
    "grid10_erode2",
    "grid10_erode5",
    "grid10_erode8",
    "grid20_erode2",
    "grid20_erode5",
    "grid20_erode8",
)


def load_official_dataset_class():
    if str(OFFICIAL_LSEJ_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_LSEJ_DIR))
    from lsej_dataloader import ImageNetLSEJDataset

    return ImageNetLSEJDataset


def chw_uint8_to_hwc_numpy(pieces: torch.Tensor) -> np.ndarray:
    array = pieces.detach().cpu().numpy()
    if array.ndim != 4 or array.shape[1] not in {3, 4} or array.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 RGB/RGBA pieces with shape [N,C,H,W], got {array.shape} {array.dtype}"
        )
    return np.transpose(array, (0, 2, 3, 1)).copy()


def target_to_inverse(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.int64)
    expected = np.arange(target.size, dtype=np.int64)
    if target.ndim != 1 or not np.array_equal(np.sort(target), expected):
        raise ValueError("LSEJ target must be a permutation of all restored positions")
    inverse = np.empty_like(target)
    inverse[target] = expected
    return inverse


def neighbor_position(position: int, direction: int, grid: int) -> int | None:
    row, column = divmod(int(position), int(grid))
    direction = int(direction)
    if direction == 0 and row > 0:
        return position - grid
    if direction == 1 and column + 1 < grid:
        return position + 1
    if direction == 2 and row + 1 < grid:
        return position + grid
    if direction == 3 and column > 0:
        return position - 1
    return None


class MetricLSEJTripletDataset(Dataset):
    """Return several same-puzzle hard-triplet candidates per official puzzle.

    Output shapes before DataLoader collation are:
      anchor:    [T, 3, S, S]
      positive:  [T, 3, S, S]
      negatives: [T, K, 3, S, S]

    T is ``triplets_per_puzzle`` and K is ``negatives_per_anchor``.
    """

    def __init__(
        self,
        data_root: str | Path = DEFAULT_LSEJ_DATA_ROOT,
        task: str = "grid10_erode2",
        split: str = "train",
        input_size: int = 224,
        normalization: str = "vit",
        triplets_per_puzzle: int = 4,
        negatives_per_anchor: int = 0,
        permute_pieces: bool = True,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        if task not in OFFICIAL_TASKS:
            raise ValueError(f"Unknown LSEJ task: {task}. Choices: {OFFICIAL_TASKS}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if triplets_per_puzzle <= 0:
            raise ValueError("triplets_per_puzzle must be positive")
        if negatives_per_anchor < 0:
            raise ValueError("negatives_per_anchor must be non-negative; use 0 for all legal negatives")
        if max_samples is not None and int(max_samples) < 0:
            raise ValueError("max_samples must be non-negative")

        dataset_class = load_official_dataset_class()
        self.dataset = dataset_class(data_root, split=split, task=task, normalize=False)
        self.data_root = Path(data_root)
        self.task = task
        self.split = split
        self.input_size = int(input_size)
        self.normalization = normalization
        self.triplets_per_puzzle = int(triplets_per_puzzle)
        self.permute_pieces = bool(permute_pieces)
        self.seed = int(seed)
        self.grid = int(self.dataset.config["puzzle"]["grid_rows"])
        self.num_pieces = self.grid * self.grid
        self.requested_negatives_per_anchor = int(negatives_per_anchor)
        max_legal_negatives = self.num_pieces - 2
        if negatives_per_anchor > max_legal_negatives:
            raise ValueError(
                f"negatives_per_anchor={negatives_per_anchor} exceeds the {max_legal_negatives} legal negatives "
                f"for task {task}"
            )
        self.negatives_per_anchor = (
            max_legal_negatives if self.requested_negatives_per_anchor == 0 else self.requested_negatives_per_anchor
        )
        total = len(self.dataset)
        self.sample_count = total if max_samples is None else min(total, int(max_samples))

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict:
        index = int(index)
        sample = self.dataset[index]
        pieces = chw_uint8_to_hwc_numpy(sample["pieces"])
        target = sample["permutation"].cpu().numpy().astype(np.int64, copy=True)

        # Train sampling remains stochastic. Validation/test are derived from
        # the puzzle index so checkpoint selection is reproducible.
        rng = np.random if self.permute_pieces else np.random.RandomState(self.seed + index)
        if self.permute_pieces:
            order = rng.permutation(self.num_pieces)
            pieces = pieces[order]
            target = target[order]

        inverse = target_to_inverse(target)
        anchors = []
        positives = []
        negative_sets = []
        anchor_indices = []
        positive_indices = []
        directions = []

        for _ in range(self.triplets_per_puzzle):
            anchor_idx, positive_idx, direction = self._sample_positive_pair(target, inverse, rng)
            candidate_edge = relation_candidate_edge(direction)
            candidates = np.asarray(
                [idx for idx in range(self.num_pieces) if idx not in {anchor_idx, positive_idx}],
                dtype=np.int64,
            )
            negative_indices = rng.choice(candidates, size=self.negatives_per_anchor, replace=False)

            anchors.append(
                piece_to_tensor(canonical_piece(pieces[anchor_idx], direction), self.input_size, self.normalization)
            )
            positives.append(
                piece_to_tensor(
                    canonical_piece(pieces[positive_idx], candidate_edge),
                    self.input_size,
                    self.normalization,
                )
            )
            negative_sets.append(
                torch.stack(
                    [
                        piece_to_tensor(
                            canonical_piece(pieces[int(negative_idx)], candidate_edge),
                            self.input_size,
                            self.normalization,
                        )
                        for negative_idx in negative_indices
                    ],
                    dim=0,
                )
            )
            anchor_indices.append(anchor_idx)
            positive_indices.append(positive_idx)
            directions.append(direction)

        return {
            "anchor": torch.stack(anchors, dim=0),
            "positive": torch.stack(positives, dim=0),
            "negatives": torch.stack(negative_sets, dim=0),
            "image_index": torch.tensor(index, dtype=torch.long),
            "anchor_idx": torch.tensor(anchor_indices, dtype=torch.long),
            "positive_idx": torch.tensor(positive_indices, dtype=torch.long),
            "direction": torch.tensor(directions, dtype=torch.long),
            "target": torch.from_numpy(target.copy()).long(),
            "lsej_id": sample["lsej_id"],
        }

    def _sample_positive_pair(self, target: np.ndarray, inverse: np.ndarray, rng) -> tuple[int, int, int]:
        anchor_idx = int(rng.randint(self.num_pieces))
        restored_position = int(target[anchor_idx])
        valid_directions = [
            int(direction)
            for direction in DIRECTIONS
            if neighbor_position(restored_position, int(direction), self.grid) is not None
        ]
        direction = valid_directions[int(rng.randint(len(valid_directions)))]
        neighbor = neighbor_position(restored_position, direction, self.grid)
        positive_idx = int(inverse[int(neighbor)])
        return anchor_idx, positive_idx, direction

    def __repr__(self) -> str:
        return (
            f"MetricLSEJTripletDataset(task={self.task!r}, split={self.split!r}, "
            f"samples={self.sample_count}, grid={self.grid}, triplets_per_puzzle={self.triplets_per_puzzle}, "
            f"negatives_per_anchor={self.negatives_per_anchor}, "
            f"requested_negatives_per_anchor={self.requested_negatives_per_anchor}, input_size={self.input_size})"
        )


__all__ = [
    "DEFAULT_LSEJ_DATA_ROOT",
    "OFFICIAL_TASKS",
    "MetricLSEJTripletDataset",
    "chw_uint8_to_hwc_numpy",
    "load_official_dataset_class",
    "neighbor_position",
    "target_to_inverse",
]
