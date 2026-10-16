"""Shared GAP inputs; importing this module never starts an evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from evaluation.output import script_output_root
from data_loaders.gap import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, GAP_SPLITS


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "commands"
DEFAULT_LOG_ROOT = script_output_root(SCRIPT_DIR / "train_gap.py", "logs")
DEFAULT_DATASET = "GAP-5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
DEFAULT_SAVE_VISUALS = True
DEFAULT_VISUAL_LIMIT = 10
DEFAULT_RUN_NAMES = {
    "GAP-3": "hard_triplet_d128_s224_kall_2026-07-31-20-21-04",
    "GAP-5": "hard_triplet_d128_s224_k15_2026-07-31-20-43-26",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate a compatibility metric checkpoint on GAP-3/GAP-5 with Gallagher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=GAP_CONFIGS.keys())
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=GAP_SPLITS)
    parser.add_argument("--data-root", default=DEFAULT_GAP_DATA_ROOT, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path, help="Exact checkpoint path; highest priority")
    parser.add_argument(
        "--run-name",
        default=None,
        help="experiment name under checkpoints/train_gap/<dataset>; leave empty to use DEFAULT_RUN_NAMES by dataset",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int)
    parser.add_argument("--start-index", default=0, type=int)
    parser.add_argument("--progress-interval", default=25, type=int)
    parser.add_argument("--save-visuals", default=DEFAULT_SAVE_VISUALS, type=parse_bool)
    parser.add_argument("--visual-limit", default=DEFAULT_VISUAL_LIMIT, type=int)
    parser.add_argument("--visual-interval", default=1, type=int)
    parser.add_argument("--save-predictions", default=True, type=parse_bool)
    parser.add_argument("--postprocess", default=True, type=parse_bool)
    parser.add_argument("--score-batch-size", default=128, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--verbose-solver", action="store_true")
    return parser


def checkpoint_from_run_dir(run_dir: Path, checkpoint_name: str) -> Path:
    checkpoint = run_dir / "checkpoints" / checkpoint_name
    if checkpoint.is_file():
        return checkpoint.resolve()
    available = sorted(path.name for path in (run_dir / "checkpoints").glob("*.pth")) \
        if (run_dir / "checkpoints").is_dir() else []
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Available: {available}")


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint
    if args.run_dir is not None:
        return checkpoint_from_run_dir(Path(args.run_dir).expanduser().resolve(), args.checkpoint_name)
    dataset_root = Path(args.log_root).expanduser().resolve() / args.dataset
    if args.run_name.strip():
        return checkpoint_from_run_dir(dataset_root / args.run_name.strip(), args.checkpoint_name)
    candidates = sorted(
        dataset_root.glob(f"*/checkpoints/{args.checkpoint_name}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        available = sorted(path.name for path in dataset_root.iterdir() if path.is_dir()) \
            if dataset_root.is_dir() else []
        raise FileNotFoundError(
            f"No {args.checkpoint_name} found under {dataset_root}. "
            f"Train with train_compatibility.py gap or set DEFAULT_RUN_NAMES. Available runs: {available}"
        )
    print(f"DEFAULT_RUN_NAME is empty; using latest checkpoint: {candidates[0]}")
    return candidates[0].resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples < 0 or args.start_index < 0:
        raise ValueError("max_samples and start_index must be non-negative")
    if args.score_batch_size <= 0 or args.visual_limit < 0 or args.visual_interval <= 0:
        raise ValueError("invalid score/visual settings")
