"""
@file: vit_handwritten.py
@description: 手写版 Vision Transformer 模型定义，用于分类和特征提取。
@author: Changxin Ye
@created: 2026-06-17
@version: 1.0
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of patch embeddings."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    """Feed-forward network used inside the Transformer encoder."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """One pre-normalized Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Standard Vision Transformer for image classification."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        representation_size: int | None = None,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if representation_size is None:
            self.pre_logits = nn.Identity()
            classifier_dim = embed_dim
        else:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh(),
            )
            classifier_dim = representation_size

        self.head = nn.Linear(classifier_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)

        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_feature = x[:, 0]
        return self.pre_logits(cls_feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        logits = self.head(features)
        return logits


VIT_MODEL_CONFIGS = {
    "vit_tiny_patch16_224": {
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_dim": 768,
    },
    "vit_small_patch16_224": {
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_dim": 1536,
    },
    "vit_base_patch16_224": {
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
    },
}

VIT_MODEL_ALIASES = {
    "tiny": "vit_tiny_patch16_224",
    "small": "vit_small_patch16_224",
    "base": "vit_base_patch16_224",
    "vit-t": "vit_tiny_patch16_224",
    "vit-s": "vit_small_patch16_224",
    "vit-b": "vit_base_patch16_224",
    "vit_t": "vit_tiny_patch16_224",
    "vit_s": "vit_small_patch16_224",
    "vit_b": "vit_base_patch16_224",
    "vit-tiny": "vit_tiny_patch16_224",
    "vit-small": "vit_small_patch16_224",
    "vit-base": "vit_base_patch16_224",
}

VIT_MODEL_NAMES = tuple(VIT_MODEL_CONFIGS.keys())
VIT_LEGACY_MODEL_ALIASES = {
    "handwritten_vit_tiny": "vit_tiny_patch16_224",
    "handwritten_vit_small": "vit_small_patch16_224",
    "handwritten_vit_base": "vit_base_patch16_224",
}


def resolve_vit_model_name(model_name: str) -> str:
    normalized_name = model_name.lower()
    model_aliases = {**VIT_MODEL_ALIASES, **VIT_LEGACY_MODEL_ALIASES}
    resolved_name = model_aliases.get(normalized_name, normalized_name)
    if resolved_name not in VIT_MODEL_CONFIGS:
        valid_names = ", ".join((*VIT_MODEL_NAMES, *sorted(VIT_MODEL_ALIASES)))
        raise ValueError(f"Unsupported ViT model '{model_name}'. Choose one of: {valid_names}")
    return resolved_name


def create_vit(
    model_name: str = "vit_small_patch16_224",
    num_classes: int = 1000,
    image_size: int = 224,
    **kwargs,
) -> VisionTransformer:
    resolved_name = resolve_vit_model_name(model_name)
    config = dict(VIT_MODEL_CONFIGS[resolved_name])
    config.update(kwargs)
    return VisionTransformer(
        image_size=image_size,
        num_classes=num_classes,
        **config,
    )


def vit_tiny(num_classes: int = 1000, image_size: int = 224, **kwargs) -> VisionTransformer:
    return create_vit(
        "vit_tiny_patch16_224",
        num_classes=num_classes,
        image_size=image_size,
        **kwargs,
    )


def vit_small(num_classes: int = 1000, image_size: int = 224, **kwargs) -> VisionTransformer:
    return create_vit(
        "vit_small_patch16_224",
        num_classes=num_classes,
        image_size=image_size,
        **kwargs,
    )


def vit_base(num_classes: int = 1000, image_size: int = 224, **kwargs) -> VisionTransformer:
    return create_vit(
        "vit_base_patch16_224",
        num_classes=num_classes,
        image_size=image_size,
        **kwargs,
    )


if __name__ == "__main__":
    #####################固定随机种子#################################
    import os
    import random
    import numpy as np
    seed = 1
    print('Using random seed : ', seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # 新增：固定 Python 哈希顺序（会影响 set/dict 转 list 的顺序）
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 强制 cuBLAS 使用“可复现/确定性”的工作区策略
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
        torch.backends.cudnn.deterministic = True  # 确保每次运行的计算结果一致
        torch.backends.cudnn.benchmark = False  # 禁用动态优化
    ######################################################

    inputs = torch.randn(2, 3, 224, 224)
    print(f"inputs.shape = {inputs.shape}")
    print(f"inputs[0][0][0]={inputs[0][0][0]}")

    pretrained_dir = Path(__file__).resolve().parent / "vit_handwritten_pretrained_weights"
    model_names = {
        "vit_tiny": "vit_tiny_patch16_224",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",
    }

    for display_name, model_name in model_names.items():
        checkpoint_path = pretrained_dir / f"{model_name}_augreg_in21k_ft_in1k_handwritten.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = create_vit(
            resolve_vit_model_name(checkpoint["model"]),
            num_classes=checkpoint["num_classes"],
            image_size=checkpoint["image_size"],
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()

        with torch.no_grad():
            outputs = model(inputs)
            features = model.forward_features(inputs)

        print(f"{display_name}:")
        print("  checkpoint:", checkpoint_path)
        print("  source:", checkpoint.get("source", "unknown"))
        print("  logits shape:", outputs.shape)
        print("  features shape:", features.shape)
        print("  outputs[0, :5]:", outputs[0, :5])
