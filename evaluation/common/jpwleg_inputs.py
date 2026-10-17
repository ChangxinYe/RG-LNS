"""Shared JPwLEG inputs; importing this module never starts an evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from evaluation.output import script_output_root
from paths import COMPATIBILITY_ROOT


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "commands"
DEFAULT_LOG_ROOT = script_output_root(SCRIPT_DIR / "train_jpwleg.py", "logs")
LEGACY_LOG_ROOT = SCRIPT_DIR / "logs" / "puzzle_hard_triplet_jpwleg"
DEFAULT_RUN_DIR = None
DEFAULT_CHECKPOINT_NAME = "best.pth"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected True or False")


def checkpoint_from_run_dir(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if checkpoint.exists():
        return checkpoint
    available = []
    dataset_root = run_dir.parent
    if dataset_root.exists():
        available = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    suffix = f" Available runs under {dataset_root}: {available}" if available else ""
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}.{suffix}")


def resolve_default_checkpoint(dataset: str, checkpoint_name: str, log_root: Path = DEFAULT_LOG_ROOT) -> Path:
    if (dataset == "jpwleg5" and checkpoint_name == DEFAULT_CHECKPOINT_NAME
            and Path(log_root).resolve() == DEFAULT_LOG_ROOT.resolve()):
        released = COMPATIBILITY_ROOT / "jpwleg5_best.pth"
        if released.is_file():
            return released
    log_roots = [log_root]
    if log_root == DEFAULT_LOG_ROOT and LEGACY_LOG_ROOT != DEFAULT_LOG_ROOT:
        log_roots.append(LEGACY_LOG_ROOT)
    search_roots = [root if dataset == "all" else root / dataset for root in log_roots]
    candidates = sorted(
        (path for root in search_roots for path in root.glob(f"**/checkpoints/{checkpoint_name}")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (path for root in search_roots for path in root.glob("**/checkpoints/last.pth")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        "No default metric-learning checkpoint was found. "
        f"Searched under: {search_roots} for {checkpoint_name} and last.pth. "
        f"Download the matching weights to {COMPATIBILITY_ROOT}, pass --checkpoint, "
        "or train a matching model with train_compatibility.py."
    )


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint).resolve()
    if args.run_dir is not None:
        checkpoint = checkpoint_from_run_dir(Path(args.run_dir), args.checkpoint_name)
        print(f"Using checkpoint from run dir: {checkpoint}")
        return checkpoint.resolve()
    run_name = (args.run_name or "").strip()
    if run_name:
        if args.dataset == "all":
            raise ValueError("--run-name requires a concrete --dataset, not --dataset all")
        run_dirs = [
            DEFAULT_LOG_ROOT / args.dataset / run_name,
            LEGACY_LOG_ROOT / args.dataset / run_name,
        ]
        checkpoint = next(
            (run_dir / "checkpoints" / args.checkpoint_name for run_dir in run_dirs
             if (run_dir / "checkpoints" / args.checkpoint_name).exists()),
            None,
        )
        if checkpoint is None:
            searched = [run_dir / "checkpoints" / args.checkpoint_name for run_dir in run_dirs]
            raise FileNotFoundError(f"Checkpoint not found. Searched: {searched}")
        print(f"Using checkpoint from run name: {checkpoint}")
        return checkpoint.resolve()
    checkpoint = resolve_default_checkpoint(args.dataset, args.checkpoint_name)
    print(f"Using default checkpoint: {checkpoint}")
    return checkpoint.resolve()
