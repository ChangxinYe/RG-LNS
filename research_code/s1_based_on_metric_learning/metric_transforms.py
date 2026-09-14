"""
@file: metric_transforms.py
@description: Puzzle-level hard triplet 拼图 baseline 的碎片张量转换与 ViT 输入归一化工具。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.1
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def normalize_piece_tensor(tensor: torch.Tensor, normalization: str) -> torch.Tensor:
    if tensor.ndim != 3 or tensor.shape[0] not in {3, 4}:
        raise ValueError(f"Expected a CHW RGB/RGBA tensor, got shape {tuple(tensor.shape)}")
    channels = int(tensor.shape[0])
    if normalization == "zero_one":
        return tensor
    if normalization == "vit":
        mean = torch.full((channels, 1, 1), 0.5, dtype=tensor.dtype, device=tensor.device)
        std = torch.full((channels, 1, 1), 0.5, dtype=tensor.dtype, device=tensor.device)
        return (tensor - mean) / std
    if normalization == "imagenet":
        mean_values = [0.485, 0.456, 0.406] + ([0.5] if channels == 4 else [])
        std_values = [0.229, 0.224, 0.225] + ([0.5] if channels == 4 else [])
        mean = torch.tensor(mean_values, dtype=tensor.dtype, device=tensor.device)[:, None, None]
        std = torch.tensor(std_values, dtype=tensor.dtype, device=tensor.device)[:, None, None]
        return (tensor - mean) / std
    if normalization == "fragment":
        flat = tensor.flatten(1)
        mean = flat.mean(dim=1)[:, None, None]
        std = flat.std(dim=1)[:, None, None]
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (tensor - mean) / std
    raise ValueError("normalization must be one of: zero_one, vit, imagenet, fragment")


def piece_to_tensor(piece: np.ndarray, input_size: int, normalization: str = "vit") -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(piece)).permute(2, 0, 1).float().div(255.0)
    if tensor.shape[-1] != input_size or tensor.shape[-2] != input_size:
        if tensor.shape[0] == 4:
            # Match PuzzleFlow's RGBA resizing and avoid bicubic overshoot on
            # GAP's binary alpha masks. RGB-only S1A inputs retain their
            # original bicubic preprocessing below.
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0).clamp_(0.0, 1.0)
        else:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(input_size, input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
    return normalize_piece_tensor(tensor, normalization=normalization)


def pieces_to_bchw(
    pieces: list[np.ndarray] | np.ndarray,
    input_size: int,
    normalization: str = "vit",
) -> torch.Tensor:
    return torch.stack([piece_to_tensor(piece, input_size, normalization) for piece in pieces], dim=0)
