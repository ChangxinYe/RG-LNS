"""
@file: experiment_naming.py
@description: 统一度量学习训练日志、评估结果和实验名称的生成规则。
@author: Changxin Ye
@created: 2026-07-19
@version: 1.1
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path


def script_output_root(script_file: str | Path, category: str) -> Path:
    """Return <module>/<category>/<script_stem> for a train/eval entrypoint."""
    script_path = Path(script_file).resolve()
    return script_path.parent / category / script_path.stem


def build_run_name(*parts: object, timestamp: str | None = None) -> str:
    """Join compact experiment tags and append a sortable timestamp."""
    clean_parts = [str(part).strip("_") for part in parts if str(part).strip("_")]
    timestamp = timestamp or dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return "_".join([*clean_parts, timestamp])


def checkpoint_run_name(checkpoint: str | Path) -> str:
    """Read the run directory name from a checkpoint path."""
    checkpoint = Path(checkpoint)
    return checkpoint.parent.parent.name if checkpoint.parent.name == "checkpoints" else checkpoint.parent.name


def prepare_unique_output_dir(
    output_dir: str | Path,
    *,
    timestamp: str | None = None,
) -> Path:
    """原子地创建评估输出目录，避免重复或并行运行相互覆盖。

    首次运行直接使用 ``output_dir``。只要目标已经存在（即使为空），后续运行就使用
    ``<output_dir>_<YYYY-mm-dd-HH-MM-SS>``；若同一秒内仍发生冲突，则继续追加
    ``_1``、``_2``。``mkdir(exist_ok=False)`` 保证多个服务器进程同时启动时只有一个
    进程能够占用同一个目录。
    """

    requested = Path(output_dir).expanduser()
    timestamp = timestamp or dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    candidate = requested
    suffix = 0
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            # 只把“候选目录本身已被占用”视为正常重名；若是某级父路径不是目录，
            # 直接暴露原始异常，避免在无效父路径下无限尝试新后缀。
            if not candidate.exists():
                raise
            suffix += 1
            suffix_text = "" if suffix == 1 else f"_{suffix - 1}"
            candidate = Path(f"{requested}_{timestamp}{suffix_text}")
