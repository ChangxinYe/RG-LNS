"""
@file: metric_jpleg_data.py
@description: JPLEG 数据集的同图合法候选采样，用于 puzzle-level hard triplet 训练。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EDGE2VEC_DIR = PROJECT_ROOT / "baselines" / "Edge2Vec_arxiv_2022"
if str(EDGE2VEC_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE2VEC_DIR))

from jpleg_data import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    JPLEG_CONFIGS,
    SPLIT_ALIASES,
    image_to_pieces,
    label_to_target_positions,
    load_jpleg_arrays,
    neighbor_position,
    target_to_inverse,
)

try:
    from .metric_geometry import (
        DIRECTIONS,
        canonical_piece,
        relation_candidate_edge,
    )
    from .metric_transforms import piece_to_tensor
except ImportError:
    from metric_geometry import (
        DIRECTIONS,
        canonical_piece,
        relation_candidate_edge,
    )
    from metric_transforms import piece_to_tensor


class JPLEGPuzzleHardTripletDataset(Dataset):
    """JPLEG samples for puzzle-level hard negative mining.

    Each item picks one legal anchor edge from one puzzle. The positive is the
    true adjacent edge in that puzzle. Negatives are all other pieces' legal
    opposite-direction edges from the same puzzle.
    """

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        dataset: str = "jpleg3",
        split: str = "train",
        input_size: int = 224,
        normalization: str = "vit",
        triplets_per_puzzle: int = 8,
        permute_pieces: bool = True,
        max_samples: int | None = None,
    ):
        if dataset not in JPLEG_CONFIGS:
            raise ValueError(f"Unknown JPLEG dataset: {dataset}. Choices: {list(JPLEG_CONFIGS)}")
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.split = SPLIT_ALIASES[split]
        self.config = JPLEG_CONFIGS[dataset]
        self.grid = self.config.grid
        self.num_pieces = self.config.num_pieces
        self.input_size = int(input_size)
        self.normalization = normalization
        self.triplets_per_puzzle = int(triplets_per_puzzle)
        if self.triplets_per_puzzle <= 0:
            raise ValueError("triplets_per_puzzle must be positive")
        self.permute_pieces = bool(permute_pieces)
        self.images, self.labels = load_jpleg_arrays(self.data_root, dataset, self.split)
        self.sample_count = len(self.images) if max_samples is None else min(len(self.images), int(max_samples))
        self.negatives_per_anchor = self.num_pieces - 2

    def __len__(self) -> int:
        return self.sample_count * self.triplets_per_puzzle

    def __getitem__(self, index: int):
        image_index = int(index) // self.triplets_per_puzzle
        image = np.asarray(self.images[image_index])
        label = np.asarray(self.labels[image_index])
        pieces = image_to_pieces(image, self.grid)
        target = label_to_target_positions(label, self.grid)

        if self.permute_pieces:
            order = np.random.permutation(self.num_pieces)
            pieces = pieces[order]
            target = target[order]

        anchor_idx, positive_idx, direction = self._sample_positive_pair(target)
        candidate_edge = relation_candidate_edge(direction)
        negative_indices = [
            idx for idx in range(self.num_pieces) if idx not in {int(anchor_idx), int(positive_idx)}
        ]

        anchor_piece = canonical_piece(pieces[anchor_idx], direction)
        positive_piece = canonical_piece(pieces[positive_idx], candidate_edge)
        negative_pieces = [
            canonical_piece(pieces[negative_idx], candidate_edge)
            for negative_idx in negative_indices
        ]

        return {
            "anchor": piece_to_tensor(anchor_piece, self.input_size, self.normalization),
            "positive": piece_to_tensor(positive_piece, self.input_size, self.normalization),
            "negatives": torch.stack(
                [piece_to_tensor(piece, self.input_size, self.normalization) for piece in negative_pieces],
                dim=0,
            ),
            "image_index": torch.tensor(image_index, dtype=torch.long),
            "anchor_idx": torch.tensor(int(anchor_idx), dtype=torch.long),
            "positive_idx": torch.tensor(int(positive_idx), dtype=torch.long),
            "direction": torch.tensor(int(direction), dtype=torch.long),
            "candidate_edge": torch.tensor(int(candidate_edge), dtype=torch.long),
        }

    def _sample_positive_pair(self, target: np.ndarray) -> tuple[int, int, int]:
        inverse = target_to_inverse(target)
        valid = []
        for anchor_idx, restored_pos in enumerate(target):
            if restored_pos < 0:
                continue
            for direction in DIRECTIONS:
                neighbor = neighbor_position(int(restored_pos), int(direction), self.grid)
                if neighbor is None:
                    continue
                positive_idx = int(inverse[neighbor])
                if positive_idx >= 0 and positive_idx != anchor_idx:
                    valid.append((int(anchor_idx), positive_idx, int(direction)))
        if not valid:
            raise RuntimeError("No valid neighboring pairs found in JPLEG sample")
        return valid[int(np.random.randint(0, len(valid)))]
