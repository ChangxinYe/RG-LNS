"""
@file: metric_gap_data.py
@description: GAP-3/GAP-5 的 HDF5 懒加载、RGBA piece 读取与 S1A 同图困难三元组采样。
              数据标签保持官方 piece-to-position 语义，训练可随机重排，验证采样固定可复现。
@author: Changxin Ye
@created: 2026-07-27
@version: 1.2
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
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
FAST_GAP_DATA_ROOT = PROJECT_ROOT / "datasets" / "GAP_fast"
# S1A GAP 实验只读取重新打包后的连续、无压缩 HDF5。
# 不回退到原始压缩数据，避免服务器漏解压 GAP_fast 时静默使用慢数据。
DEFAULT_GAP_DATA_ROOT = FAST_GAP_DATA_ROOT


@dataclass(frozen=True)
class GAPConfig:
    name: str
    grid: int
    num_pieces: int
    piece_size: int = 128
    channels: int = 4


GAP_CONFIGS = {
    "GAP-3": GAPConfig("GAP-3", grid=3, num_pieces=9),
    "GAP-5": GAPConfig("GAP-5", grid=5, num_pieces=25),
}
GAP_SPLITS = ("train", "val", "test")


def normalize_gap_name(dataset: str) -> str:
    normalized = str(dataset).strip().upper().replace("_", "-")
    aliases = {"GAP3": "GAP-3", "GAP5": "GAP-5"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in GAP_CONFIGS:
        raise ValueError(f"Unknown GAP dataset {dataset!r}; choices: {tuple(GAP_CONFIGS)}")
    return normalized


def target_to_inverse(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.int64)
    expected = np.arange(target.size, dtype=np.int64)
    if target.ndim != 1 or not np.array_equal(np.sort(target), expected):
        raise ValueError("GAP labels must be a permutation of all restored positions")
    inverse = np.empty_like(target)
    inverse[target] = expected
    return inverse


def neighbor_position(position: int, direction: int, grid: int) -> int | None:
    row, column = divmod(int(position), int(grid))
    if direction == 0 and row > 0:
        return position - grid
    if direction == 1 and column + 1 < grid:
        return position + 1
    if direction == 2 and row + 1 < grid:
        return position + grid
    if direction == 3 and column > 0:
        return position - 1
    return None


class GAPPuzzleDataset(Dataset):
    """Lazy official GAP split reader returning uint8 RGBA pieces and labels."""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_GAP_DATA_ROOT,
        dataset: str = "GAP-3",
        split: str = "train",
        *,
        start_index: int = 0,
        max_samples: int | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.dataset_name = normalize_gap_name(dataset)
        self.config = GAP_CONFIGS[self.dataset_name]
        if split not in GAP_SPLITS:
            raise ValueError(f"split must be one of {GAP_SPLITS}")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        self.split = split
        self.split_dir = self.data_root / self.dataset_name / split
        self.puzzles_path = self.split_dir / "puzzles.h5"
        self.labels_path = self.split_dir / "labels_indices.h5"
        self.metadata_path = self.split_dir / "metadata.json"
        for path in (self.puzzles_path, self.labels_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required GAP_fast file not found: {path}. "
                    "Extract or generate datasets/GAP_fast before running S1A GAP experiments; "
                    "no fallback dataset is enabled."
                )

        self.metadata: dict[str, Any] = {}
        if self.metadata_path.is_file():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        with h5py.File(self.puzzles_path, "r") as puzzle_file:
            puzzle_dataset = puzzle_file["puzzles"]
            if puzzle_dataset.chunks is not None or puzzle_dataset.compression is not None:
                raise ValueError(
                    f"S1A GAP requires contiguous, uncompressed GAP_fast puzzles, but {self.puzzles_path} "
                    f"uses chunks={puzzle_dataset.chunks}, compression={puzzle_dataset.compression!r}. "
                    "Run datasets/GAP_fast/repack_gap_hdf5.py and use its output directory."
                )
            shape = tuple(int(value) for value in puzzle_dataset.shape)
        with h5py.File(self.labels_path, "r") as label_file:
            label_shape = tuple(int(value) for value in label_file["labels"].shape)
        expected_tail = (
            self.config.num_pieces,
            self.config.piece_size,
            self.config.piece_size,
            self.config.channels,
        )
        if shape[1:] != expected_tail:
            raise ValueError(f"Unexpected GAP puzzle shape {shape}; expected (N,{expected_tail})")
        if label_shape != (shape[0], self.config.num_pieces):
            raise ValueError(f"Unexpected GAP label shape {label_shape} for puzzle shape {shape}")
        if start_index >= shape[0]:
            raise IndexError(f"start_index={start_index} is outside split with {shape[0]} samples")
        stop = shape[0] if max_samples is None else min(shape[0], start_index + int(max_samples))
        self.indices = range(int(start_index), int(stop))
        self._puzzles_file = None
        self._labels_file = None
        self._puzzles = None
        self._labels = None

    def _ensure_open(self) -> None:
        if self._puzzles_file is None:
            self._puzzles_file = h5py.File(self.puzzles_path, "r")
            self._puzzles = self._puzzles_file["puzzles"]
        if self._labels_file is None:
            self._labels_file = h5py.File(self.labels_path, "r")
            self._labels = self._labels_file["labels"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self._ensure_open()
        official_index = int(self.indices[int(index)])
        pieces = np.asarray(self._puzzles[official_index], dtype=np.uint8).copy()
        target = np.asarray(self._labels[official_index], dtype=np.int64).copy()
        target_to_inverse(target)
        return {
            "pieces": pieces,
            "target": target,
            "sample_index": official_index,
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_puzzles_file"] = None
        state["_labels_file"] = None
        state["_puzzles"] = None
        state["_labels"] = None
        return state

    def close(self) -> None:
        for name in ("_puzzles_file", "_labels_file"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.close()
                setattr(self, name, None)
        self._puzzles = None
        self._labels = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MetricGAPTripletDataset(Dataset):
    """Return T same-puzzle positive relations and K legal negatives per GAP puzzle."""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_GAP_DATA_ROOT,
        dataset: str = "GAP-3",
        split: str = "train",
        input_size: int = 224,
        normalization: str = "vit",
        triplets_per_puzzle: int = 4,
        negatives_per_anchor: int = 0,
        permute_pieces: bool = True,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        if triplets_per_puzzle <= 0:
            raise ValueError("triplets_per_puzzle must be positive")
        if negatives_per_anchor < 0:
            raise ValueError("negatives_per_anchor must be non-negative; use 0 for all legal negatives")
        self.puzzles = GAPPuzzleDataset(
            data_root=data_root,
            dataset=dataset,
            split=split,
            max_samples=max_samples,
        )
        self.dataset_name = self.puzzles.dataset_name
        self.split = split
        self.grid = self.puzzles.config.grid
        self.num_pieces = self.puzzles.config.num_pieces
        self.input_size = int(input_size)
        self.normalization = normalization
        self.triplets_per_puzzle = int(triplets_per_puzzle)
        self.permute_pieces = bool(permute_pieces)
        self.seed = int(seed)
        self.requested_negatives_per_anchor = int(negatives_per_anchor)
        max_legal_negatives = self.num_pieces - 2
        if negatives_per_anchor > max_legal_negatives:
            raise ValueError(
                f"negatives_per_anchor={negatives_per_anchor} exceeds the {max_legal_negatives} "
                f"legal negatives in {self.dataset_name}"
            )
        self.negatives_per_anchor = (
            max_legal_negatives if negatives_per_anchor == 0 else int(negatives_per_anchor)
        )

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.puzzles[int(index)]
        pieces = sample["pieces"]
        target = sample["target"]
        rng = np.random if self.split == "train" else np.random.RandomState(self.seed + int(index))
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
                [piece for piece in range(self.num_pieces) if piece not in {anchor_idx, positive_idx}],
                dtype=np.int64,
            )
            negative_indices = rng.choice(candidates, size=self.negatives_per_anchor, replace=False)
            anchors.append(
                piece_to_tensor(canonical_piece(pieces[anchor_idx], direction), self.input_size, self.normalization)
            )
            positives.append(
                piece_to_tensor(
                    canonical_piece(pieces[positive_idx], candidate_edge), self.input_size, self.normalization
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
            "sample_index": torch.tensor(sample["sample_index"], dtype=torch.long),
            "anchor_idx": torch.tensor(anchor_indices, dtype=torch.long),
            "positive_idx": torch.tensor(positive_indices, dtype=torch.long),
            "direction": torch.tensor(directions, dtype=torch.long),
            "target": torch.from_numpy(target.copy()).long(),
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
            f"MetricGAPTripletDataset(dataset={self.dataset_name!r}, split={self.split!r}, "
            f"samples={len(self)}, grid={self.grid}, triplets_per_puzzle={self.triplets_per_puzzle}, "
            f"negatives_per_anchor={self.negatives_per_anchor}, input_size={self.input_size})"
        )


__all__ = [
    "DEFAULT_GAP_DATA_ROOT",
    "FAST_GAP_DATA_ROOT",
    "GAP_CONFIGS",
    "GAP_SPLITS",
    "GAPConfig",
    "GAPPuzzleDataset",
    "MetricGAPTripletDataset",
    "neighbor_position",
    "normalize_gap_name",
    "target_to_inverse",
]
