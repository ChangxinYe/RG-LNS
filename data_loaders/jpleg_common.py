from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from compatibility.geometry import (
    DIRECTIONS,
    canonical_anchor_piece,
    canonical_candidate_piece,
    relation_candidate_edge,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets" / "MET_Dataset"
PIECE_SIZE = 96


@dataclass(frozen=True)
class JPLEGConfig:
    folder: str
    grid: int
    suffix: str
    train_image_name: str | None = None

    @property
    def num_pieces(self) -> int:
        return self.grid * self.grid


JPLEG_CONFIGS = {
    "jpleg3": JPLEGConfig(folder="JPLEG-3", grid=3, suffix="48gap_33"),
    "jpleg5": JPLEGConfig(
        folder="JPLEG-5",
        grid=5,
        suffix="12gap_55",
        train_image_name="train_img_12gap_55-002.npy",
    ),
}

SPLIT_ALIASES = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "test": "test",
}


def image_file_name(config: JPLEGConfig, split: str) -> str:
    if split == "train" and config.train_image_name:
        return config.train_image_name
    return f"{split}_img_{config.suffix}.npy"


def load_jpleg_arrays(data_root: str | Path, dataset: str, split: str):
    if dataset not in JPLEG_CONFIGS:
        raise ValueError(f"Unknown JPLEG dataset: {dataset}")
    split = SPLIT_ALIASES[split]
    config = JPLEG_CONFIGS[dataset]
    folder = Path(data_root) / config.folder
    image_path = folder / image_file_name(config, split)
    label_path = folder / f"{split}_label_{config.suffix}.npy"
    if not image_path.exists():
        raise FileNotFoundError(f"Image npy not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label npy not found: {label_path}")
    images = np.load(image_path, mmap_mode="r")
    labels = np.load(label_path, mmap_mode="r")
    if len(images) != len(labels):
        raise ValueError(f"Image/label length mismatch: {len(images)} != {len(labels)}")
    return images, labels


def image_to_pieces(image: np.ndarray, grid: int, piece_size: int = PIECE_SIZE) -> np.ndarray:
    expected = grid * piece_size
    if image.shape[:2] != (expected, expected):
        raise ValueError(f"Expected image shape {(expected, expected, 3)}, got {image.shape}")
    pieces = []
    for row in range(grid):
        for col in range(grid):
            pieces.append(
                np.asarray(
                    image[
                        row * piece_size : (row + 1) * piece_size,
                        col * piece_size : (col + 1) * piece_size,
                        :,
                    ],
                    dtype=np.uint8,
                ).copy()
            )
    return np.stack(pieces, axis=0)


def label_to_target_positions(label: np.ndarray, grid: int) -> np.ndarray:
    label = np.asarray(label)
    num_pieces = grid * grid
    center = num_pieces // 2
    non_center_positions = [index for index in range(num_pieces) if index != center]
    expected_shape = (num_pieces - 1, num_pieces - 1)
    if label.shape != expected_shape:
        raise ValueError(f"Expected label shape {expected_shape}, got {label.shape}")

    target = np.full(num_pieces, -1, dtype=np.int64)
    target[center] = center
    permutation = label.argmax(axis=1)
    for current_order, restored_order in enumerate(permutation):
        current_position = non_center_positions[current_order]
        restored_position = non_center_positions[int(restored_order)]
        target[current_position] = restored_position
    return target


def target_to_inverse(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.int64)
    inverse = np.full(target.shape[0], -1, dtype=np.int64)
    for piece_idx, restored_pos in enumerate(target):
        if 0 <= restored_pos < len(inverse):
            inverse[int(restored_pos)] = int(piece_idx)
    return inverse


def neighbor_position(position: int, direction: int, grid: int) -> int | None:
    row, col = divmod(int(position), grid)
    if direction == 0:
        row -= 1
    elif direction == 1:
        col += 1
    elif direction == 2:
        row += 1
    elif direction == 3:
        col -= 1
    else:
        raise ValueError(f"Unknown direction: {direction}")
    if row < 0 or col < 0 or row >= grid or col >= grid:
        return None
    return row * grid + col


def normalize_piece_tensor(tensor: torch.Tensor, normalization: str) -> torch.Tensor:
    tensor = tensor.float().div(255.0)
    if normalization == "zero_one":
        return tensor
    if normalization == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device)[:, None, None]
        return (tensor - mean) / std
    if normalization == "fragment":
        flat = tensor.flatten(1)
        mean = flat.mean(dim=1)[:, None, None]
        std = flat.std(dim=1)[:, None, None]
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (tensor - mean) / std
    raise ValueError("normalization must be one of: zero_one, imagenet, fragment")


def piece_to_tensor(piece: np.ndarray, input_size: int, normalization: str = "zero_one") -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(piece)).permute(2, 0, 1)
    tensor = normalize_piece_tensor(tensor, normalization=normalization)
    if tensor.shape[-1] != input_size or tensor.shape[-2] != input_size:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
    return tensor


def pieces_to_bchw(
    pieces: list[np.ndarray] | np.ndarray,
    input_size: int,
    normalization: str = "zero_one",
) -> torch.Tensor:
    return torch.stack([piece_to_tensor(piece, input_size, normalization) for piece in pieces], dim=0)


class Edge2VecJPLEGTripletDataset(Dataset):
    """Triplet dataset for Edge2Vec on JPLEG.

    It returns canonical single-encoder inputs and metadata used by HBT to
    avoid selecting a true positive edge as a hard negative.
    """

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        dataset: str = "jpleg3",
        split: str = "train",
        input_size: int = 96,
        normalization: str = "zero_one",
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
        self.input_size = int(input_size)
        self.normalization = normalization
        self.triplets_per_puzzle = int(triplets_per_puzzle)
        if self.triplets_per_puzzle <= 0:
            raise ValueError("triplets_per_puzzle must be positive")
        self.permute_pieces = bool(permute_pieces)
        self.images, self.labels = load_jpleg_arrays(self.data_root, dataset, self.split)
        self.sample_count = len(self.images) if max_samples is None else min(len(self.images), int(max_samples))

    def __len__(self) -> int:
        return self.sample_count * self.triplets_per_puzzle

    def __getitem__(self, index: int):
        image_index = int(index) // self.triplets_per_puzzle
        image = np.asarray(self.images[image_index])
        label = np.asarray(self.labels[image_index])
        pieces = image_to_pieces(image, self.grid)
        target = label_to_target_positions(label, self.grid)
        original_ids = np.arange(self.config.num_pieces, dtype=np.int64)

        if self.permute_pieces:
            order = np.random.permutation(self.config.num_pieces)
            pieces = pieces[order]
            target = target[order]
            original_ids = original_ids[order]

        anchor_idx, positive_idx, negative_idx, direction = self._sample_triplet_indices(target)
        candidate_edge = relation_candidate_edge(direction)

        anchor_piece = canonical_anchor_piece(pieces[anchor_idx], direction)
        positive_piece = canonical_candidate_piece(pieces[positive_idx], candidate_edge)
        negative_piece = canonical_candidate_piece(pieces[negative_idx], candidate_edge)

        return {
            "anchor": piece_to_tensor(anchor_piece, self.input_size, self.normalization),
            "positive": piece_to_tensor(positive_piece, self.input_size, self.normalization),
            "negative": piece_to_tensor(negative_piece, self.input_size, self.normalization),
            "image_index": torch.tensor(image_index, dtype=torch.long),
            "positive_orig_id": torch.tensor(int(original_ids[positive_idx]), dtype=torch.long),
            "positive_edge": torch.tensor(candidate_edge, dtype=torch.long),
            "negative_orig_id": torch.tensor(int(original_ids[negative_idx]), dtype=torch.long),
            "negative_edge": torch.tensor(candidate_edge, dtype=torch.long),
        }

    def _sample_triplet_indices(self, target: np.ndarray) -> tuple[int, int, int, int]:
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
                    valid.append((anchor_idx, positive_idx, int(direction)))
        if not valid:
            raise RuntimeError("No valid neighboring pairs found in JPLEG sample")

        anchor_idx, positive_idx, direction = valid[int(np.random.randint(0, len(valid)))]
        forbidden = {int(anchor_idx), int(positive_idx)}
        candidates = [idx for idx in range(self.config.num_pieces) if idx not in forbidden]
        negative_idx = candidates[int(np.random.randint(0, len(candidates)))]
        return int(anchor_idx), int(positive_idx), int(negative_idx), int(direction)
