"""
@file: metric_vit_model.py
@description: 基于 handwritten ViT 预训练 backbone 的度量学习 embedding encoder。
@author: Changxin Ye
@created: 2026-07-10
@version: 1.1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vit_handwritten import (  # noqa: E402
    VIT_MODEL_CONFIGS,
    create_vit,
    resolve_vit_model_name,
)


def default_pretrained_weights_path(model_name: str) -> Path:
    resolved_name = resolve_vit_model_name(model_name)
    return (
        PROJECT_ROOT
        / "models"
        / "vit_handwritten_pretrained_weights"
        / f"{resolved_name}_augreg_in21k_ft_in1k_handwritten.pth"
    )


def load_torch_file(path: str | Path, map_location="cpu"):
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


class ViTMetricEncoder(nn.Module):
    """Handwritten ViT backbone plus a trainable metric projection head."""

    def __init__(
        self,
        backbone: str = "vit-b",
        embedding_dim: int = 128,
        image_size: int = 224,
        input_channels: int = 3,
        pretrained: bool = True,
        pretrained_weights_path: str | Path | None = None,
        normalize_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = resolve_vit_model_name(backbone)
        self.embedding_dim = int(embedding_dim)
        self.image_size = int(image_size)
        self.input_channels = int(input_channels)
        if self.input_channels not in {3, 4}:
            raise ValueError("input_channels must be 3 (RGB) or 4 (RGBA)")
        self.pretrained = bool(pretrained)
        self.normalize_embedding = bool(normalize_embedding)
        self.embedding_normalization = "l2" if self.normalize_embedding else "none"
        self.pretrained_weights_path = (
            Path(pretrained_weights_path) if pretrained_weights_path else default_pretrained_weights_path(self.backbone)
        )

        self.vit = create_vit(self.backbone, num_classes=1000, image_size=self.image_size)
        if self.pretrained:
            self._load_pretrained_weights(self.pretrained_weights_path)

        # GAP fragments are RGBA.  Keep the pretrained ViT strictly RGB and
        # learn the same lightweight 4->3 adapter used by PuzzleFlow.  Identity
        # initialization preserves the pretrained RGB behavior at step zero;
        # training can then learn how much alpha-mask information to inject.
        self.channel_adapter = None
        if self.input_channels == 4:
            self.channel_adapter = nn.Conv2d(4, 3, kernel_size=1, bias=True)
            with torch.no_grad():
                self.channel_adapter.weight.zero_()
                self.channel_adapter.bias.zero_()
                for channel in range(3):
                    self.channel_adapter.weight[channel, channel, 0, 0] = 1.0

        feature_dim = int(VIT_MODEL_CONFIGS[self.backbone]["embed_dim"])
        self.project = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.embedding_dim),
        )

    def _load_pretrained_weights(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained handwritten ViT weights not found: {checkpoint_path}. "
                "Run models/utils1_download_convert_timm_to_handwritten_vit.py first, "
                "or pass --pretrained-backbone False."
            )
        checkpoint = load_torch_file(checkpoint_path, map_location="cpu")
        checkpoint_model = resolve_vit_model_name(checkpoint.get("model", self.backbone))
        checkpoint_image_size = int(checkpoint.get("image_size", self.image_size))
        if checkpoint_model != self.backbone:
            raise ValueError(f"Checkpoint model is {checkpoint_model}, but requested backbone is {self.backbone}")
        if checkpoint_image_size != self.image_size:
            raise ValueError(
                f"Checkpoint image_size is {checkpoint_image_size}, but requested input_size is {self.image_size}. "
                "Use --input-size 224 with the provided pretrained weights."
            )
        self.vit.load_state_dict(checkpoint["state_dict"], strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected BCHW input with {self.input_channels} channels, got {tuple(x.shape)}"
            )
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        features = self.vit.forward_features(x)
        embedding = self.project(features)
        if self.normalize_embedding:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class ViTMetricScaleHeadsEncoder(ViTMetricEncoder):
    """Shared handwritten ViT backbone with one projection head per scale."""

    def __init__(
        self,
        macro_scales: tuple[int, ...] | list[int] = (1, 2, 5),
        inference_macro_scale: int = 1,
        backbone: str = "vit-b",
        embedding_dim: int = 128,
        image_size: int = 224,
        input_channels: int = 3,
        pretrained: bool = True,
        pretrained_weights_path: str | Path | None = None,
        normalize_embedding: bool = True,
    ) -> None:
        scales = tuple(int(scale) for scale in macro_scales)
        if not scales or len(set(scales)) != len(scales) or any(scale <= 0 for scale in scales):
            raise ValueError("macro_scales must contain unique positive integers")
        if int(inference_macro_scale) not in scales:
            raise ValueError("inference_macro_scale must be included in macro_scales")

        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            image_size=image_size,
            input_channels=input_channels,
            pretrained=pretrained,
            pretrained_weights_path=pretrained_weights_path,
            normalize_embedding=normalize_embedding,
        )
        feature_dim = int(VIT_MODEL_CONFIGS[self.backbone]["embed_dim"])
        inference_project = self.project
        del self.project
        self.macro_scales = scales
        self.inference_macro_scale = int(inference_macro_scale)
        self.project_by_scale = nn.ModuleDict(
            {
                str(scale): (
                    inference_project
                    if scale == self.inference_macro_scale
                    else nn.Sequential(
                        nn.LayerNorm(feature_dim),
                        nn.Linear(feature_dim, self.embedding_dim),
                    )
                )
                for scale in self.macro_scales
            }
        )

    def forward(self, x: torch.Tensor, macro_scale: int | None = None) -> torch.Tensor:
        scale = self.inference_macro_scale if macro_scale is None else int(macro_scale)
        scale_key = str(scale)
        if scale_key not in self.project_by_scale:
            raise ValueError(f"No projection head for macro scale {scale}; available: {self.macro_scales}")
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected BCHW input with {self.input_channels} channels, got {tuple(x.shape)}"
            )
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        features = self.vit.forward_features(x)
        embedding = self.project_by_scale[scale_key](features)
        if self.normalize_embedding:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class ViTContextToPieceTeacherEncoder(ViTMetricEncoder):
    """One ViT Teacher with separate context-query and 1x1-piece heads.

    The backbone is shared because both inputs must describe the same visual
    puzzle domain. The two lightweight heads absorb the substantial input-role
    difference: a masked multi-piece context acts as a query, while a single
    canonicalized piece acts as a candidate.
    """

    def __init__(
        self,
        backbone: str = "vit-b",
        embedding_dim: int = 128,
        image_size: int = 224,
        input_channels: int = 3,
        pretrained: bool = True,
        pretrained_weights_path: str | Path | None = None,
        normalize_embedding: bool = True,
    ) -> None:
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            image_size=image_size,
            input_channels=input_channels,
            pretrained=pretrained,
            pretrained_weights_path=pretrained_weights_path,
            normalize_embedding=normalize_embedding,
        )
        feature_dim = int(VIT_MODEL_CONFIGS[self.backbone]["embed_dim"])
        self.context_project = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.embedding_dim),
        )

    def _forward_with_head(self, x: torch.Tensor, head: nn.Module) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected BCHW input with {self.input_channels} channels, got {tuple(x.shape)}"
            )
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)
        embedding = head(self.vit.forward_features(x))
        if self.normalize_embedding:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward_context(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_head(x, self.context_project)

    def forward_piece(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_with_head(x, self.project)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default to the 1x1-piece head; explicit role methods are preferred."""

        return self.forward_piece(x)


def model_config_dict(model: ViTMetricEncoder) -> dict:
    return {
        "encoder_type": "vit_handwritten_metric",
        "backbone": model.backbone,
        "embedding_dim": model.embedding_dim,
        "input_channels": model.input_channels,
        "input_adapter": "conv1x1_rgba_to_rgb" if model.channel_adapter is not None else "identity_rgb",
        "embedding_normalization": model.embedding_normalization,
        "input_size": model.image_size,
        "pretrained_backbone": model.pretrained,
        "pretrained_weights_path": str(model.pretrained_weights_path),
    }


def scale_heads_model_config_dict(model: ViTMetricScaleHeadsEncoder) -> dict:
    return {
        "encoder_type": "vit_handwritten_metric_scale_heads",
        "backbone": model.backbone,
        "embedding_dim": model.embedding_dim,
        "embedding_normalization": model.embedding_normalization,
        "input_size": model.image_size,
        "pretrained_backbone": model.pretrained,
        "pretrained_weights_path": str(model.pretrained_weights_path),
        "macro_scales": list(model.macro_scales),
        "inference_macro_scale": model.inference_macro_scale,
        "projection_head_policy": "scale_specific",
    }


def context_to_piece_teacher_config_dict(model: ViTContextToPieceTeacherEncoder) -> dict:
    return {
        **model_config_dict(model),
        "encoder_type": "vit_handwritten_context_to_piece_teacher",
        "backbone_policy": "shared_between_context_and_piece",
        "projection_head_policy": "role_specific",
        "context_projection_head": "context_project",
        "piece_projection_head": "project",
    }


def create_metric_encoder_from_config(
    config: dict,
    pretrained: bool = False,
) -> ViTMetricEncoder | ViTMetricScaleHeadsEncoder | ViTContextToPieceTeacherEncoder:
    # Checkpoints created before hyperspherical embeddings did not store this
    # field, so keep their original unnormalized inference behavior.
    normalize_embedding = config.get("embedding_normalization", "none") == "l2"
    common_args = {
        "backbone": config.get("backbone", "vit-b"),
        "embedding_dim": int(config.get("embedding_dim", 128)),
        "image_size": int(config.get("input_size", 224)),
        "input_channels": int(config.get("input_channels", 3)),
        "pretrained": pretrained,
        "pretrained_weights_path": config.get("pretrained_weights_path") or None,
        "normalize_embedding": normalize_embedding,
    }
    if config.get("encoder_type") == "vit_handwritten_metric_scale_heads":
        return ViTMetricScaleHeadsEncoder(
            macro_scales=tuple(int(scale) for scale in config.get("macro_scales", (1, 2, 5))),
            inference_macro_scale=int(config.get("inference_macro_scale", 1)),
            **common_args,
        )
    if config.get("encoder_type") == "vit_handwritten_context_to_piece_teacher":
        return ViTContextToPieceTeacherEncoder(**common_args)
    return ViTMetricEncoder(
        **common_args,
    )
