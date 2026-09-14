"""
@file: policy.py
@description: S8A 无 GNN 的空间 CNN 与共享候选动作 Actor-Critic 网络。
@author: Changxin Ye
@created: 2026-08-04
@version: 1.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .actions import ACTION_FEATURE_DIM, STATE_CHANNELS


@dataclass(frozen=True)
class PolicyConfig:
    state_channels: int = STATE_CHANNELS
    action_feature_dim: int = ACTION_FEATURE_DIM
    hidden_dim: int = 128
    dropout: float = 0.10

    def validate(self) -> None:
        if min(self.state_channels, self.action_feature_dim, self.hidden_dim) <= 0:
            raise ValueError("策略网络维度必须为正")
        if self.hidden_dim % 8 != 0:
            raise ValueError("hidden-dim 必须是 8 的倍数，以满足 GroupNorm 分组")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)")

    def to_dict(self) -> dict:
        return asdict(self)


class SpatialCandidateActorCritic(nn.Module):
    """CNN 编码 H×W×C 布局，MLP 对任意数量的组件平移动作共享打分。"""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        config.validate()
        self.config = config
        hidden = config.hidden_dim
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(config.state_channels, hidden // 2, 3, padding=1),
            nn.GroupNorm(4, hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        spatial_state: torch.Tensor,
        action_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_state.ndim == 3:
            spatial_state = spatial_state.unsqueeze(0)
        if spatial_state.ndim != 4 or spatial_state.shape[0] != 1:
            raise ValueError("当前 S8A 策略一次评估一个可变动作集合，state batch 必须为 1")
        if action_features.ndim != 2:
            raise ValueError("action_features 必须为 [A,F]")
        spatial = self.pool(self.spatial_encoder(spatial_state)).flatten(1)
        action = self.action_encoder(action_features)
        repeated_spatial = spatial.expand(action.shape[0], -1)
        logits = self.actor(torch.cat((repeated_spatial, action), dim=1)).squeeze(1)
        value = self.critic(spatial).squeeze()
        return logits, value


__all__ = ["PolicyConfig", "SpatialCandidateActorCritic"]
