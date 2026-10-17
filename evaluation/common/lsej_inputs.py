"""Shared ImageNet-LSEJ inputs; importing this module never starts an evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from evaluation.output import script_output_root
from compatibility.scorer import load_torch_checkpoint, resolve_device
from paths import COMPATIBILITY_ROOT


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "commands"
LOSS_TYPES = (
    "hard_triplet",
    "semi_hard_triplet",
    "random_triplet",
    "distance_weighted_triplet",
    "infonce",
)
DEFAULT_LOG_ROOT = script_output_root(SCRIPT_DIR / "train_lsej.py", "logs")
DEFAULT_LOG_ROOTS = {loss_type: DEFAULT_LOG_ROOT for loss_type in LOSS_TYPES}
LEGACY_LOG_ROOTS = {
    "hard_triplet": SCRIPT_DIR / "logs" / "puzzle_hard_triplet_lsej",
    "semi_hard_triplet": SCRIPT_DIR / "logs" / "puzzle_semi_hard_triplet_lsej",
    "random_triplet": SCRIPT_DIR / "logs" / "puzzle_random_triplet_lsej",
    "distance_weighted_triplet": SCRIPT_DIR / "logs" / "puzzle_distance_weighted_triplet_lsej",
    "infonce": SCRIPT_DIR / "logs" / "puzzle_infonce_lsej",
}
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
    raise argparse.ArgumentTypeError("please enter True or False")


def checkpoint_from_run_dir(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = Path(run_dir) / "checkpoints" / checkpoint_name
    if checkpoint.is_file():
        return checkpoint
    available = sorted(path.name for path in (Path(run_dir) / "checkpoints").glob("*.pth"))
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Available checkpoint files: {available}")


def resolve_default_checkpoint(
    task: str,
    checkpoint_name: str,
    loss_type: str,
    log_root: Path | None = None,
) -> Path:
    if log_root is None:
        log_root = DEFAULT_LOG_ROOTS[loss_type]
    if (task in {"grid10_erode2", "grid10_erode5"} and loss_type == "hard_triplet"
            and checkpoint_name == DEFAULT_CHECKPOINT_NAME
            and Path(log_root).resolve() == DEFAULT_LOG_ROOTS[loss_type].resolve()):
        released = COMPATIBILITY_ROOT / f"lsej_{task}_best.pth"
        if released.is_file():
            return released
    log_roots = [Path(log_root)]
    if Path(log_root) == DEFAULT_LOG_ROOTS[loss_type]:
        log_roots.append(LEGACY_LOG_ROOTS[loss_type])
    search_roots = [root / task for root in log_roots]
    candidates = sorted(
        (path for root in search_roots for path in root.glob(f"*/checkpoints/{checkpoint_name}")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            (path for root in search_roots for path in root.glob("*/checkpoints/last.pth")),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No compatibility LSEJ {loss_type} checkpoint found for task {task}. Searched under {search_roots}. "
        f"Download the matching weights to {COMPATIBILITY_ROOT}, pass --checkpoint, or train this task first."
    )


def resolve_checkpoint_arg(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint).resolve()
    if args.run_dir is not None:
        return checkpoint_from_run_dir(Path(args.run_dir), args.checkpoint_name).resolve()
    run_name = (args.run_name or "").strip()
    log_root = DEFAULT_LOG_ROOTS[args.loss_type]
    if run_name:
        run_dirs = [
            log_root / args.task / run_name,
            LEGACY_LOG_ROOTS[args.loss_type] / args.task / run_name,
        ]
        for run_dir in run_dirs:
            checkpoint = run_dir / "checkpoints" / args.checkpoint_name
            if checkpoint.is_file():
                return checkpoint.resolve()
        searched = [run_dir / "checkpoints" / args.checkpoint_name for run_dir in run_dirs]
        raise FileNotFoundError(f"Checkpoint not found. Searched: {searched}")
    return resolve_default_checkpoint(args.task, args.checkpoint_name, args.loss_type, log_root).resolve()


def checkpoint_loss_type(checkpoint: dict) -> str:
    loss_config = checkpoint.get("loss_config", {})
    loss_type = loss_config.get("loss_type")
    if loss_type in LOSS_TYPES:
        return str(loss_type)
    loss_name = str(loss_config.get("loss", "puzzle_level_sampled_hard_triplet")).lower()
    if "distance_weighted" in loss_name:
        return "distance_weighted_triplet"
    if "random_triplet" in loss_name:
        return "random_triplet"
    if "semi_hard" in loss_name:
        return "semi_hard_triplet"
    return "infonce" if "infonce" in loss_name else "hard_triplet"


def validate_checkpoint(checkpoint_path: Path, task: str, loss_type: str) -> None:
    checkpoint = load_torch_checkpoint(checkpoint_path, resolve_device("cpu", 0))
    checkpoint_task = checkpoint.get("task") or checkpoint.get("args", {}).get("task")
    if checkpoint_task is not None and str(checkpoint_task) != task:
        raise ValueError(
            f"Checkpoint was trained for task {checkpoint_task}, but evaluation requested {task}. "
            "Cross-task checkpoint guessing is disabled."
        )
    checkpoint_loss = checkpoint_loss_type(checkpoint)
    if checkpoint_loss != loss_type:
        raise ValueError(
            f"Checkpoint uses loss_type={checkpoint_loss}, but evaluation requested {loss_type}. "
            "Pass the matching --loss-type or select another checkpoint."
        )
