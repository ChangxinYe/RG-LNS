"""
@file: experiment_stage_timing.py
@description: Reusable staged-inference timing, CUDA synchronization, and summary statistics for evaluation scripts.
              Timers record only formal inference stages explicitly delimited by the caller; model loading,
              offline metrics, visualization, and result writing are excluded.
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
    """Returned by ``measure`` and populated with elapsed seconds on context exit."""

    seconds: float = 0.0


def add_stage_timing_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add unified timing arguments to an existing evaluation entry point."""

    parser.add_argument(
        "--runtime-warmup-samples",
        default=DEFAULT_RUNTIME_WARMUP_SAMPLES,
        type=int,
        help="Number of initial samples used only for runtime warm-up; small runs retain at least one timed sample",
    )
    return parser


def _percentile(values: list[float], percentile: float) -> float:
    """Compute a percentile by linear interpolation without extra dependencies."""

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
    """Collect per-sample stage timings and generate reproducible paper statistics."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        warmup_samples: int,
        total_samples: int,
    ) -> None:
        if warmup_samples < 0:
            raise ValueError("runtime-warmup-samples must be non-negative")
        if total_samples <= 0:
            raise ValueError("runtime evaluation requires at least one sample")
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
        """Measure a code block; GPU stages must explicitly request synchronization."""

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
        """Record one sample and return fields ready to merge into ``sample_results.csv``."""

        missing = [name for name in RUNTIME_STAGE_NAMES if name not in stages]
        if missing:
            raise KeyError(f"missing runtime stages: {missing}")
        measured = int(sample_position >= self.warmup_samples)
        row: dict[str, float | int] = {
            "runtime_sample_position": int(sample_position),
            "runtime_sample_index": int(sample_index),
            "runtime_measured": measured,
        }
        for name in RUNTIME_STAGE_NAMES:
            value = float(stages[name])
            if value < 0:
                raise ValueError(f"runtime for stage {name} cannot be negative")
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
    """Generate compact text suitable for writing to ``summary.txt``."""

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
