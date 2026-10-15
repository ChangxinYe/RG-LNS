"""
@file: experiment_naming.py
@description: Unified naming rules for metric-learning training logs, evaluation results, and experiments.
@author: Changxin Ye
@created: 2026-07-19
@version: 1.1
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def script_output_root(script_file: str | Path, category: str) -> Path:
    """Return a repository-level output directory for one entry point."""
    script_path = Path(script_file).resolve()
    if category == "logs":
        return PROJECT_ROOT / "checkpoints" / script_path.stem
    if category == "eval_result":
        return PROJECT_ROOT / "results" / "runs" / script_path.stem
    return PROJECT_ROOT / category / script_path.stem


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
    """Atomically create an evaluation output directory to avoid collisions or overwrites from parallel runs.

    The first run uses ``output_dir`` directly. Once the target exists, even if empty, later runs use
    ``<output_dir>_<YYYY-mm-dd-HH-MM-SS>``; if a collision remains within the same second, append
    ``_1``, ``_2``, and so on. ``mkdir(exist_ok=False)`` ensures that only one of several server
    processes starting simultaneously can claim a directory.
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
            # Treat only the candidate directory itself being occupied as a normal name collision.
            # If a parent path is not a directory, expose the original exception instead of
            # trying suffixes indefinitely under an invalid parent path.
            if not candidate.exists():
                raise
            suffix += 1
            suffix_text = "" if suffix == 1 else f"_{suffix - 1}"
            candidate = Path(f"{requested}_{timestamp}{suffix_text}")
