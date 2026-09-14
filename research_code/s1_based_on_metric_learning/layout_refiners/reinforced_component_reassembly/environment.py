"""
@file: environment.py
@description: S8A 组件重组强化学习环境、离线缓存读取器和训练奖励定义。
@author: Changxin Ye
@created: 2026-08-04
@version: 1.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

try:
    from ...metric_lsej_evaluator import compute_relationship_counts
    from .actions import ReassemblyAction, build_action_set
except ImportError:
    from metric_lsej_evaluator import compute_relationship_counts
    from layout_refiners.reinforced_component_reassembly.actions import ReassemblyAction, build_action_set


@dataclass(frozen=True)
class RewardConfig:
    """训练奖励。AA/SRA 只用于 reward，不进入策略观察。"""

    aa_weight: float = 1.0
    sra_weight: float = 0.5
    e1_weight: float = 0.10
    step_penalty: float = 0.01
    perfect_bonus: float = 1.0
    repeat_penalty: float = 0.10


@dataclass(frozen=True)
class ReassemblyEnvironmentConfig:
    mutual_top_k: int = 3
    min_component_size: int = 4
    max_components: int = 2
    translations_per_component: int = 6
    completion_beam_width: int = 2
    max_steps: int = 3

    def validate(self) -> None:
        values = (
            self.mutual_top_k,
            self.min_component_size,
            self.max_components,
            self.translations_per_component,
            self.completion_beam_width,
            self.max_steps,
        )
        if min(values) <= 0:
            raise ValueError("环境中的 Top-K、组件数、平移数和最大步数都必须为正")


def validate_positions(positions: np.ndarray, grid: int, name: str) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int32)
    expected = np.arange(grid * grid, dtype=np.int32)
    if positions.shape != expected.shape or not np.array_equal(np.sort(positions), expected):
        raise ValueError(f"{name} 必须是 0..{grid * grid - 1} 的完整排列")
    return positions


def layout_metrics(prediction: np.ndarray, target: np.ndarray, grid: int) -> dict[str, float]:
    prediction = validate_positions(prediction, grid, "prediction")
    target = validate_positions(target, grid, "target")
    correct = int(np.sum(prediction == target))
    relationships = compute_relationship_counts(prediction, target, grid)
    relation_correct = relationships["horizontal_correct"] + relationships["vertical_correct"]
    relation_total = relationships["horizontal_total"] + relationships["vertical_total"]
    return {
        "PA": float(correct == grid * grid),
        "AA": correct / (grid * grid),
        "SRA": relation_correct / relation_total if relation_total else 0.0,
    }


class PreparedReassemblyDataset(Dataset):
    """读取 S8A 缓存，也兼容已经生成的 S7A v2 缓存。"""

    def __init__(self, root: str | Path, split: str, maximum: int = 0):
        self.root = Path(root).expanduser().resolve()
        self.split = split
        manifest_path = self.root / split / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"找不到准备数据 manifest：{manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.grid = int(self.manifest["grid"])
        self.task = str(self.manifest["task"])
        self.solver = str(self.manifest.get("solver", "unknown"))
        self.records = list(self.manifest["records"])
        if maximum > 0:
            self.records = self.records[:maximum]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        metadata = self.records[index]
        path = self.root / self.split / metadata["file"]
        with np.load(path, allow_pickle=False) as arrays:
            required = {"scores", "initial_positions", "target_positions"}
            missing = required - set(arrays.files)
            if missing:
                raise ValueError(f"{path} 缺少字段：{sorted(missing)}")
            return {
                "scores": arrays["scores"].astype(np.float32, copy=True),
                "initial_positions": validate_positions(
                    arrays["initial_positions"], self.grid, "initial_positions"
                ).copy(),
                "target_positions": validate_positions(
                    arrays["target_positions"], self.grid, "target_positions"
                ).copy(),
                "dataset_index": int(metadata.get("dataset_index", index)),
                "lsej_id": str(metadata.get("lsej_id", f"sample_{index:08d}")),
                "source_path": str(path),
            }

    def __repr__(self) -> str:
        return (
            f"PreparedReassemblyDataset(task={self.task!r}, split={self.split!r}, "
            f"solver={self.solver!r}, samples={len(self)})"
        )


class ReassemblyEnvironment:
    """有限步 MDP：动作选择可靠组件及平移，环境用 E1 完成剩余 piece 回填。"""

    def __init__(
        self,
        grid: int,
        config: ReassemblyEnvironmentConfig,
        reward_config: RewardConfig,
        *,
        terminate_on_perfect: bool = False,
    ):
        config.validate()
        self.grid = int(grid)
        self.config = config
        self.reward_config = reward_config
        # 训练时可用 GT 完成状态缩短轨迹；正式推理必须保持 False，避免测试标签控制流程。
        self.terminate_on_perfect = bool(terminate_on_perfect)
        self.scores: np.ndarray | None = None
        self.target: np.ndarray | None = None
        self.prediction: np.ndarray | None = None
        self.initial_prediction: np.ndarray | None = None
        self.steps = 0
        self.done = True
        self.seen: set[bytes] = set()
        self.actions: list[ReassemblyAction] = []
        self.trace: list[dict] = []
        self._state: np.ndarray | None = None
        self._sample_cache_key: str | None = None
        self._action_cache: dict[tuple[str, bytes], tuple[np.ndarray, list[ReassemblyAction], dict]] = {}

    def reset(self, sample: dict) -> tuple[np.ndarray, np.ndarray]:
        self.scores = np.asarray(sample["scores"], dtype=np.float32)
        target = sample.get("target_positions")
        self.target = (
            validate_positions(target, self.grid, "target") if target is not None else None
        )
        self.prediction = validate_positions(
            sample["initial_positions"], self.grid, "initial"
        ).copy()
        self.initial_prediction = self.prediction.copy()
        # 离线训练样本具有稳定 source_path，可跨 epoch 复用昂贵的 profiled 回填候选。
        self._sample_cache_key = str(sample["source_path"]) if sample.get("source_path") else None
        self.steps = 0
        self.done = False
        self.seen = {self.prediction.tobytes()}
        self.trace = []
        return self._refresh_actions()

    def _refresh_actions(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.scores is not None and self.prediction is not None
        cache_key = (
            (self._sample_cache_key, self.prediction.tobytes())
            if self._sample_cache_key is not None
            else None
        )
        cached = self._action_cache.get(cache_key) if cache_key is not None else None
        if cached is None:
            state, actions, graph_stats = build_action_set(
                self.scores,
                self.prediction,
                self.grid,
                mutual_top_k=self.config.mutual_top_k,
                min_component_size=self.config.min_component_size,
                max_components=self.config.max_components,
                translations_per_component=self.config.translations_per_component,
                completion_beam_width=self.config.completion_beam_width,
            )
            if cache_key is not None:
                self._action_cache[cache_key] = (state, actions, graph_stats)
        else:
            state, actions, graph_stats = cached
        self.actions = actions
        self._state = state
        self.graph_stats = graph_stats
        return state, np.stack([action.features for action in self.actions])

    def _metrics(self, prediction: np.ndarray) -> dict[str, float]:
        # 正式推理可以完全不把 GT 交给环境；此时 reward 不参与决策，返回零占位。
        if self.target is None:
            return {"PA": 0.0, "AA": 0.0, "SRA": 0.0}
        return layout_metrics(prediction, self.target, self.grid)

    def step(self, action_index: int) -> tuple[np.ndarray | None, np.ndarray | None, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode 已结束，必须先 reset")
        if not 0 <= action_index < len(self.actions):
            raise IndexError(f"动作 {action_index} 超出当前动作集合 0..{len(self.actions) - 1}")
        assert self.prediction is not None and self.scores is not None
        action = self.actions[action_index]
        before = self._metrics(self.prediction)
        before_e1 = float(self.actions[0].mean_e1)
        if action.is_stop:
            self.done = True
            info = {
                "action": action.label,
                "action_index": int(action_index),
                "action_count": len(self.actions),
                "component_size": 0,
                "row_shift": 0,
                "column_shift": 0,
                "completion_origin": "stop",
                "reward": 0.0,
                "reward_components": {
                    "aa": 0.0,
                    "sra": 0.0,
                    "e1": 0.0,
                    "step": 0.0,
                    "perfect": 0.0,
                    "repeat": 0.0,
                },
                "before": before,
                "after": before,
                "done_reason": "policy_stop",
            }
            self.trace.append(info)
            return None, None, 0.0, True, info

        candidate = action.prediction.copy()
        repeated = candidate.tobytes() in self.seen
        after = self._metrics(candidate)
        e1_delta = float(np.tanh((before_e1 - action.mean_e1) / max(abs(before_e1), 1e-6)))
        reward_components = {
            "aa": self.reward_config.aa_weight * (after["AA"] - before["AA"]),
            "sra": self.reward_config.sra_weight * (after["SRA"] - before["SRA"]),
            "e1": self.reward_config.e1_weight * e1_delta,
            "step": -self.reward_config.step_penalty,
            "perfect": 0.0,
            "repeat": -self.reward_config.repeat_penalty * float(repeated),
        }
        if after["PA"] > before["PA"]:
            reward_components["perfect"] = self.reward_config.perfect_bonus
        reward = float(sum(reward_components.values()))
        self.prediction = candidate
        self.seen.add(candidate.tobytes())
        self.steps += 1
        reason = "running"
        if self.terminate_on_perfect and after["PA"] == 1.0:
            reason = "perfect"
        elif repeated:
            reason = "repeated_layout"
        elif self.steps >= self.config.max_steps:
            reason = "max_steps"
        self.done = reason != "running"
        info = {
            "action": action.label,
            "action_index": int(action_index),
            "action_count": len(self.actions),
            "component_size": int(len(action.component)),
            "row_shift": action.row_shift,
            "column_shift": action.column_shift,
            "completion_origin": action.completion_origin,
            "reward": float(reward),
            "reward_components": reward_components,
            "before": before,
            "after": after,
            "before_e1": before_e1,
            "after_e1": action.mean_e1,
            "done_reason": reason,
        }
        self.trace.append(info)
        if self.done:
            return None, None, float(reward), True, info
        next_state, next_features = self._refresh_actions()
        return next_state, next_features, float(reward), False, info


__all__ = [
    "PreparedReassemblyDataset",
    "ReassemblyEnvironment",
    "ReassemblyEnvironmentConfig",
    "RewardConfig",
    "layout_metrics",
    "validate_positions",
]
