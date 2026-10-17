"""
@file: evaluate_lsej_gallagher.py
@description: Load a compatibility checkpoint and evaluate official ImageNet-LSEJ tasks with Gallagher.
@author: Changxin Ye
@created: 2026-07-17
@version: 1.3
"""

from __future__ import annotations

from evaluation.common.lsej_inputs import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_LOG_ROOT,
    DEFAULT_LOG_ROOTS,
    DEFAULT_RUN_DIR,
    LEGACY_LOG_ROOTS,
    LOSS_TYPES,
    SCRIPT_DIR,
    checkpoint_from_run_dir,
    checkpoint_loss_type,
    parse_bool,
    resolve_checkpoint_arg,
    resolve_default_checkpoint,
    validate_checkpoint,
)

import argparse
from pathlib import Path

from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
from data_loaders.lsej import DEFAULT_LSEJ_DATA_ROOT, LSEJ_TASKS, normalize_lsej_task
from evaluation.lsej import evaluate_lsej_split
from compatibility.scorer import MetricCompatibilityScorer, load_torch_checkpoint, resolve_device


DEFAULT_LOSS_TYPE = "hard_triplet"
DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET_NAME = "ImageNet_LSEJ"
DEFAULT_TASK = "lsej_grid10_erode2"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
if DEFAULT_LOSS_TYPE == "hard_triplet":
    if DEFAULT_TASK == "lsej_grid10_erode2":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-07-25-10-22-41"
    elif DEFAULT_TASK == "lsej_grid10_erode5":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-08-02-21-16-37"
elif DEFAULT_LOSS_TYPE == "semi_hard_triplet":
    # Empty selects the latest semi-hard experiment for the current task; set an exact run name after training.
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_semi_hard_triplet_d128_s224_kall_2026-07-18-15-51-36"
elif DEFAULT_LOSS_TYPE == "random_triplet":
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_random_triplet_d128_s224_kall_2026-07-18-22-33-41"
elif DEFAULT_LOSS_TYPE == "distance_weighted_triplet":
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_distance_weighted_triplet_c0.5_n1.4_d128_s224_kall_2026-07-18-22-34-25"
elif DEFAULT_LOSS_TYPE == "infonce":
    DEFAULT_RUN_NAME = "grid10_erode2_vit_tiny_patch16_224_pretrained_puzzle_infonce_tau0.1_d128_s224_kall_2026-07-18-11-34-05"
else:
    raise ValueError(f"DEFAULT_LOSS_TYPE must be one of {LOSS_TYPES}, got {DEFAULT_LOSS_TYPE!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate compatibility scores on ImageNet-LSEJ with Gallagher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, type=normalize_lsej_task, choices=LSEJ_TASKS, help="ImageNet-LSEJ benchmark setting")
    parser.add_argument("--loss-type", default=DEFAULT_LOSS_TYPE, choices=LOSS_TYPES, help="training objective for the default checkpoint and result directory")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "val", "test"], help="data split to evaluate")
    parser.add_argument("--data-root", default=DEFAULT_LSEJ_DATA_ROOT, type=Path, help="ImageNet-LSEJ dataset root")
    parser.add_argument("--checkpoint", default=None, type=Path, help="exact checkpoint path; highest priority")
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="experiment directory under the task log root; empty selects the latest run for the current task",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="experiment directory containing the checkpoints subdirectory")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="filename loaded from the experiment checkpoints directory")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="maximum puzzles; 0 uses the full split")
    parser.add_argument("--start-index", default=0, type=int, help="starting puzzle index in the split")
    parser.add_argument("--score-batch-size", default=128, type=int, help="images per forward batch when extracting edge embeddings")
    parser.add_argument("--postprocess", default=True, type=parse_bool, help="whether to normalize rows and symmetrize compatibility scores")
    parser.add_argument("--device", default="auto", help="execution device, e.g. auto, cpu, or cuda:0; auto prefers a GPU")
    parser.add_argument("--gpu-id", default=0, type=int, help="GPU index used when device=auto")
    parser.add_argument("--progress-interval", default=1, type=int, help="print PA/AA/SRA every N puzzles; 0 disables output")
    parser.add_argument("--verbose-solver", action="store_true", help="show Gallagher solver output; silent by default")
    parser.add_argument("--output-dir", default=None, type=Path, help="evaluation-result directory; empty combines run, checkpoint, task, and split")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="whether to save visualizations with correct/incorrect borders")
    parser.add_argument("--visual-limit", default=10, type=int, help="maximum number of visualizations to save")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="whether to save predictions.npz for later metric recomputation")
    return parser.parse_args()


def default_output_dir(checkpoint: Path, task: str, split: str) -> Path:
    checkpoint = Path(checkpoint)
    folder_name = f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{task}_{split}"
    return DEFAULT_OUTPUT_ROOT / DEFAULT_DATASET_NAME / folder_name


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    validate_checkpoint(args.checkpoint, args.task, args.loss_type)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.checkpoint, args.task, args.split)
    args.output_dir = prepare_unique_output_dir(args.output_dir)

    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    metrics = evaluate_lsej_split(
        scorer=scorer,
        data_root=args.data_root,
        task=args.task,
        split=args.split,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
        start_index=args.start_index,
        progress_interval=args.progress_interval,
        verbose_solver=args.verbose_solver,
        save_visuals=args.save_visuals,
        visual_limit=args.visual_limit,
        save_predictions=args.save_predictions,
        method_name="Baseline (ViT-T + Gallagher)",
    )
    print(
        f"Result: PA={metrics.pa:.2%}, AA={metrics.aa:.2%}, "
        f"Horizontal SRA={metrics.horizontal_sra:.2%}, Vertical SRA={metrics.vertical_sra:.2%}, "
        f"SRA={metrics.sra:.2%}, time={metrics.elapsed_seconds:.1f}s"
    )
    print(f"Saved: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
