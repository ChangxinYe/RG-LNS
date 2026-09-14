"""S8A：基于强化学习的可靠组件重组。"""

from .actions import ACTION_FEATURE_DIM, STATE_CHANNELS, ReassemblyAction, build_action_set
from .environment import (
    PreparedReassemblyDataset,
    ReassemblyEnvironment,
    ReassemblyEnvironmentConfig,
    RewardConfig,
    layout_metrics,
)
from .policy import PolicyConfig, SpatialCandidateActorCritic
from .partial_layout_completion import (
    complete_partial_layout,
    is_complete_layout,
    partial_layout_metrics,
)

__all__ = [
    "ACTION_FEATURE_DIM",
    "STATE_CHANNELS",
    "PolicyConfig",
    "PreparedReassemblyDataset",
    "ReassemblyAction",
    "ReassemblyEnvironment",
    "ReassemblyEnvironmentConfig",
    "RewardConfig",
    "SpatialCandidateActorCritic",
    "build_action_set",
    "complete_partial_layout",
    "is_complete_layout",
    "layout_metrics",
    "partial_layout_metrics",
]
