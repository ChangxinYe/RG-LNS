"""
@file: metric_scorer.py
@description: handwritten ViT 度量学习 checkpoint 加载与拼图兼容性矩阵评分接口。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.0
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EDGE2VEC_DIR = PROJECT_ROOT / "baselines" / "Edge2Vec_arxiv_2022"
if str(EDGE2VEC_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE2VEC_DIR))

try:
    from .metric_geometry import (
        CANONICAL_EDGE,
        GEOMETRY_MODE,
        LEGACY_LEFT_MODE,
        LEGACY_RIGHT_MODE,
        DIRECTIONS,
        OPPOSITE,
        canonical_anchor_piece,
        canonical_candidate_piece,
        canonical_piece,
        relation_candidate_edge,
    )
    from .metric_transforms import pieces_to_bchw
    from .metric_vit_model import create_metric_encoder_from_config
except ImportError:
    from metric_geometry import (
        CANONICAL_EDGE,
        GEOMETRY_MODE,
        LEGACY_LEFT_MODE,
        LEGACY_RIGHT_MODE,
        DIRECTIONS,
        OPPOSITE,
        canonical_anchor_piece,
        canonical_candidate_piece,
        canonical_piece,
        relation_candidate_edge,
    )
    from metric_transforms import pieces_to_bchw
    from metric_vit_model import create_metric_encoder_from_config


def resolve_device(device: str = "auto", gpu_id: int = 0) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")
    return torch.device(device)


@contextlib.contextmanager
def _cross_platform_path_compat():
    original_posix_path = pathlib.PosixPath
    original_windows_path = pathlib.WindowsPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.PurePosixPath
    else:
        pathlib.WindowsPath = pathlib.PureWindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = original_posix_path
        pathlib.WindowsPath = original_windows_path


def load_torch_checkpoint(path: str | Path, device: torch.device):
    checkpoint_path = Path(path)

    def _load_once():
        try:
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=device)

    try:
        return _load_once()
    except NotImplementedError as exc:
        if "PosixPath" in str(exc) or "WindowsPath" in str(exc):
            with _cross_platform_path_compat():
                return _load_once()
        raise


def load_metric_checkpoint(path: str | Path, device: torch.device):
    checkpoint = load_torch_checkpoint(path, device)
    config = checkpoint.get("model_config", {})
    supported_encoder_types = {
        "vit_handwritten_metric",
        "vit_handwritten_metric_scale_heads",
    }
    if config.get("encoder_type") not in supported_encoder_types:
        raise ValueError(f"Unsupported metric encoder type: {config.get('encoder_type')}")
    input_size = int(config.get("input_size", checkpoint.get("input_size", 224)))
    model = create_metric_encoder_from_config(config, pretrained=False)
    state = checkpoint.get("model", checkpoint.get("model_state", checkpoint.get("state_dict")))
    if state is None:
        raise KeyError(f"Checkpoint does not contain a model state: {path}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    normalization = checkpoint.get("normalization", config.get("normalization", "vit"))
    score_metric = checkpoint.get("score_metric", config.get("score_metric", "euclidean"))
    canonical_edge = config.get("canonical_edge", "left")
    geometry_mode = config.get("geometry_mode")
    if geometry_mode is None:
        geometry_mode = LEGACY_RIGHT_MODE if canonical_edge == "right" else LEGACY_LEFT_MODE
    return model, {
        "input_size": input_size,
        "embedding_dim": int(config.get("embedding_dim", 128)),
        "backbone": config.get("backbone", "vit-b"),
        "normalization": normalization,
        "score_metric": score_metric,
        "canonical_edge": canonical_edge,
        "geometry_mode": geometry_mode,
    }, checkpoint


class MetricCompatibilityScorer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        input_size: int = 224,
        normalization: str = "vit",
        batch_size: int = 128,
        postprocess: bool = True,
        score_metric: str = "euclidean",
        canonical_edge: str = CANONICAL_EDGE,
        geometry_mode: str = GEOMETRY_MODE,
    ):
        if score_metric not in {"cosine", "euclidean"}:
            raise ValueError("score_metric must be 'cosine' or 'euclidean'")
        self.model = model.to(device).eval()
        self.device = device
        self.input_size = int(input_size)
        self.normalization = normalization
        self.batch_size = int(batch_size)
        self.postprocess = bool(postprocess)
        self.score_metric = score_metric
        self.canonical_edge = str(canonical_edge).lower()
        if self.canonical_edge not in {"left", "right"}:
            raise ValueError("canonical_edge must be 'left' or 'right'")
        self.geometry_mode = str(geometry_mode)
        valid_geometry_modes = {GEOMETRY_MODE, LEGACY_LEFT_MODE, LEGACY_RIGHT_MODE}
        if self.geometry_mode not in valid_geometry_modes:
            raise ValueError(f"Unsupported geometry_mode: {self.geometry_mode}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device,
        batch_size: int = 128,
        postprocess: bool = True,
    ) -> "MetricCompatibilityScorer":
        model, config, _checkpoint = load_metric_checkpoint(checkpoint_path, device)
        return cls(
            model=model,
            device=device,
            input_size=config["input_size"],
            normalization=config["normalization"],
            batch_size=batch_size,
            postprocess=postprocess,
            score_metric=config["score_metric"],
            canonical_edge=config["canonical_edge"],
            geometry_mode=config["geometry_mode"],
        )

    @torch.no_grad()
    def _embed(self, tensors: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(0, tensors.shape[0], self.batch_size):
            batch = tensors[start : start + self.batch_size].to(self.device, non_blocking=True)
            outputs.append(self.model(batch).detach().cpu())
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def score_pieces(self, pieces: list[np.ndarray] | np.ndarray, rot_flag: int = 0) -> np.ndarray:
        if rot_flag:
            raise NotImplementedError("Type-2 metric-learning scoring is reserved for a later implementation")
        pieces = list(pieces)
        n_pieces = len(pieces)
        if self.geometry_mode == GEOMETRY_MODE:
            edge_embeddings = []
            for edge in DIRECTIONS:
                edge_pieces = [canonical_piece(piece, edge) for piece in pieces]
                edge_tensor = pieces_to_bchw(edge_pieces, self.input_size, self.normalization)
                edge_embeddings.append(self._embed(edge_tensor))
            anchor_embeddings = edge_embeddings
            candidate_embeddings = [edge_embeddings[relation_candidate_edge(d)] for d in DIRECTIONS]
        else:
            anchor_embeddings = []
            candidate_embeddings = []
            for direction in DIRECTIONS:
                candidate_edge = relation_candidate_edge(direction)
                anchor_pieces = [
                    canonical_anchor_piece(piece, direction, self.canonical_edge) for piece in pieces
                ]
                candidate_pieces = [
                    canonical_candidate_piece(piece, candidate_edge, self.canonical_edge) for piece in pieces
                ]
                anchor_tensor = pieces_to_bchw(anchor_pieces, self.input_size, self.normalization)
                candidate_tensor = pieces_to_bchw(candidate_pieces, self.input_size, self.normalization)
                anchor_embeddings.append(self._embed(anchor_tensor))
                candidate_embeddings.append(self._embed(candidate_tensor))

        scores = np.empty((n_pieces, n_pieces, 4), dtype=np.float32)
        for direction in DIRECTIONS:
            left = anchor_embeddings[direction]
            right = candidate_embeddings[direction]
            if self.score_metric == "cosine":
                left = F.normalize(left, dim=1)
                right = F.normalize(right, dim=1)
                dist = 1.0 - torch.matmul(left, right.T)
            else:
                dist = torch.cdist(left, right, p=2)
            scores[:, :, direction] = dist.numpy().astype(np.float32)

        for direction in DIRECTIONS:
            np.fill_diagonal(scores[:, :, direction], np.inf)

        if self.postprocess:
            scores = metric_postprocess_scores(scores)
        return scores


def metric_postprocess_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 3 or scores.shape[2] != 4:
        raise ValueError(f"Expected score shape (N, N, 4), got {scores.shape}")
    norm = scores.copy()
    n_pieces = scores.shape[0]

    for direction in DIRECTIONS:
        layer = norm[:, :, direction]
        for row in range(n_pieces):
            finite = np.isfinite(layer[row])
            if not np.any(finite):
                continue
            row_min = float(layer[row, finite].min())
            row_max = float(layer[row, finite].max())
            denom = max(row_max - row_min, eps)
            layer[row, finite] = (layer[row, finite] - row_min) / denom
        norm[:, :, direction] = layer

    out = norm.copy()
    for direction in DIRECTIONS:
        reverse = int(OPPOSITE[direction])
        out[:, :, direction] = 0.5 * (norm[:, :, direction] + norm[:, :, reverse].T)

    for direction in DIRECTIONS:
        np.fill_diagonal(out[:, :, direction], np.inf)
    return out.astype(np.float32)
