"""
@file: experiment_stage_timing.py
@description: 为评估脚本提供可复用的分阶段推理计时、CUDA 同步和统计汇总。
              计时器只记录调用方显式包围的正式推理阶段，不包含模型加载、
              离线指标、可视化或结果写盘。
@author: Changxin Ye
@created: 2026-09-01
@version: 1.0
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch


DEFAULT_RUNTIME_WARMUP_SAMPLES = 20
RUNTIME_STAGE_NAMES = (
    "compatibility",
    "initial_solver",
    "partial_completion",
    "rgls",
    "refinement_total",
    "inference_total",
)


@dataclass
class TimingMeasurement:
    """由 ``measure`` 返回，并在上下文退出后写入秒数。"""

    seconds: float = 0.0


def add_stage_timing_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """向现有评估入口增加统一的计时参数。"""

    parser.add_argument(
        "--runtime-warmup-samples",
        default=DEFAULT_RUNTIME_WARMUP_SAMPLES,
        type=int,
        help="前多少个样本仅用于运行时预热，不计入汇总；小样本运行会至少保留一个计时样本",
    )
    return parser


def _percentile(values: list[float], percentile: float) -> float:
    """使用线性插值计算百分位数，避免额外依赖。"""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stage_statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "p95": 0.0,
        }
    return {
        "count": len(values),
        "total": float(sum(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)),
        "p95": float(_percentile(values, 0.95)),
    }


class StageTimingCollector:
    """收集逐样本分阶段时间，并生成论文可复现的统计量。"""

    def __init__(
        self,
        *,
        device: str | torch.device,
        warmup_samples: int,
        total_samples: int,
    ) -> None:
        if warmup_samples < 0:
            raise ValueError("runtime-warmup-samples 必须非负")
        if total_samples <= 0:
            raise ValueError("运行时间评估至少需要一个样本")
        self.device = torch.device(device)
        self.configured_warmup_samples = int(warmup_samples)
        self.warmup_samples = min(int(warmup_samples), max(0, int(total_samples) - 1))
        self.total_samples = int(total_samples)
        self._rows: list[dict[str, float | int]] = []

    def _synchronize_cuda(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, *, synchronize_cuda: bool = False) -> Iterator[TimingMeasurement]:
        """测量一个代码块；GPU 阶段必须显式请求同步。"""

        measurement = TimingMeasurement()
        if synchronize_cuda:
            self._synchronize_cuda()
        started = time.perf_counter()
        try:
            yield measurement
        finally:
            if synchronize_cuda:
                self._synchronize_cuda()
            measurement.seconds = float(time.perf_counter() - started)

    def add_sample(
        self,
        *,
        sample_position: int,
        sample_index: int,
        stages: Mapping[str, float],
    ) -> dict[str, float | int]:
        """记录一个样本，并返回可直接并入 ``sample_results.csv`` 的字段。"""

        missing = [name for name in RUNTIME_STAGE_NAMES if name not in stages]
        if missing:
            raise KeyError(f"缺少运行时间阶段: {missing}")
        measured = int(sample_position >= self.warmup_samples)
        row: dict[str, float | int] = {
            "runtime_sample_position": int(sample_position),
            "runtime_sample_index": int(sample_index),
            "runtime_measured": measured,
        }
        for name in RUNTIME_STAGE_NAMES:
            value = float(stages[name])
            if value < 0:
                raise ValueError(f"阶段 {name} 的运行时间不能为负")
            row[f"{name}_seconds"] = value
        self._rows.append(row)
        return row.copy()

    def summary(self) -> dict:
        measured_rows = [row for row in self._rows if int(row["runtime_measured"]) == 1]
        stages = {
            name: _stage_statistics(
                [float(row[f"{name}_seconds"]) for row in measured_rows]
            )
            for name in RUNTIME_STAGE_NAMES
        }
        return {
            "unit": "seconds_per_puzzle",
            "configured_warmup_samples": self.configured_warmup_samples,
            "warmup_samples": self.warmup_samples,
            "recorded_samples": len(self._rows),
            "measured_samples": len(measured_rows),
            "stages": stages,
        }


def format_stage_timing_summary(summary: Mapping) -> list[str]:
    """生成适合写入 ``summary.txt`` 的紧凑文本。"""

    lines = [
        "Stage Runtime",
        "-------------",
        (
            "unit=seconds_per_puzzle | "
            f"warmup={summary['warmup_samples']} | "
            f"measured={summary['measured_samples']}"
        ),
    ]
    for name in RUNTIME_STAGE_NAMES:
        stats = summary["stages"][name]
        lines.append(
            f"{name}: mean={stats['mean']:.6f} | median={stats['median']:.6f} | "
            f"std={stats['std']:.6f} | p95={stats['p95']:.6f} | "
            f"total={stats['total']:.6f}"
        )
    return lines


__all__ = [
    "DEFAULT_RUNTIME_WARMUP_SAMPLES",
    "RUNTIME_STAGE_NAMES",
    "StageTimingCollector",
    "TimingMeasurement",
    "add_stage_timing_arguments",
    "format_stage_timing_summary",
]
