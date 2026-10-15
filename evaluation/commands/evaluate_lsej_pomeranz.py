"""
@file: evaluate_lsej_pomeranz.py
@description: Evaluate ImageNet-LSEJ with the unified E1 compatibility distance and the pure-Python Pomeranz CVPR 2011 solver.
@author: Changxin Ye
@created: 2026-07-28
@version: 1.0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

try:
    from initial_solvers.pomeranz import PomeranzSolverConfig, solve_with_pomeranz
    from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from data_loaders.lsej import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from evaluation.lsej import evaluate_lsej_split
    from compatibility.scorer import MetricCompatibilityScorer, resolve_device
    from evaluation.commands.evaluate_lsej_gallagher import (
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_RUN_DIR,
        LOSS_TYPES,
        parse_bool,
        resolve_checkpoint_arg,
        validate_checkpoint,
    )
except ImportError:
    from initial_solvers.pomeranz import PomeranzSolverConfig, solve_with_pomeranz
    from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from data_loaders.lsej import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from evaluation.lsej import evaluate_lsej_split
    from compatibility.scorer import MetricCompatibilityScorer, resolve_device
    from evaluation.commands.evaluate_lsej_gallagher import (
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_RUN_DIR,
        LOSS_TYPES,
        parse_bool,
        resolve_checkpoint_arg,
        validate_checkpoint,
    )


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET_NAME = "ImageNet_LSEJ"
DEFAULT_TASK = "grid10_erode2"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 0
DEFAULT_LOSS_TYPE = "hard_triplet"
DEFAULT_RUN_NAME = None
if DEFAULT_LOSS_TYPE == "hard_triplet":
    if DEFAULT_TASK == "grid10_erode2":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-07-25-10-22-41"
    elif DEFAULT_TASK == "grid10_erode5":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-08-02-21-16-37"
    elif DEFAULT_TASK == "grid10_erode8":
        DEFAULT_RUN_NAME = "hard_triplet_d128_s224_k15_2026-08-02-21-17-06"

DEFAULT_POMERANZ_RESTARTS = 10
DEFAULT_POMERANZ_MAX_REFINEMENT_ROUNDS = 0
DEFAULT_POMERANZ_RANDOM_SEED = 0


def add_pomeranz_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--pomeranz-restarts",
        default=DEFAULT_POMERANZ_RESTARTS,
        type=int,
        help="number of distinct initial seeds for independent Pomeranz runs; select the result by final BBM",
    )
    parser.add_argument(
        "--pomeranz-max-refinement-rounds",
        default=DEFAULT_POMERANZ_MAX_REFINEMENT_ROUNDS,
        type=int,
        help="maximum reassembly rounds per seed; 0 runs until BBM no longer strictly improves",
    )
    parser.add_argument(
        "--pomeranz-random-seed",
        default=DEFAULT_POMERANZ_RANDOM_SEED,
        type=int,
        help="fixed random seed for generating initial pieces across restarts",
    )
    return parser


def pomeranz_config_from_args(args: argparse.Namespace) -> PomeranzSolverConfig:
    return PomeranzSolverConfig(
        restarts=args.pomeranz_restarts,
        max_refinement_rounds=args.pomeranz_max_refinement_rounds,
        random_seed=args.pomeranz_random_seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate ImageNet-LSEJ with the E1 compatibility distance and Pomeranz CVPR 2011 solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, choices=OFFICIAL_TASKS, help="official LSEJ task to evaluate")
    parser.add_argument("--loss-type", default=DEFAULT_LOSS_TYPE, choices=LOSS_TYPES, help="training loss for the checkpoint")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["train", "val", "test"], help="data split to evaluate")
    parser.add_argument("--data-root", default=DEFAULT_LSEJ_DATA_ROOT, type=Path, help="ImageNet-LSEJ dataset root")
    parser.add_argument("--checkpoint", default=None, type=Path, help="exact checkpoint path; highest priority")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="experiment directory under the current task log root")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="directory containing checkpoints")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="checkpoint filename to load")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="maximum puzzles; 0 uses the full split")
    parser.add_argument("--start-index", default=0, type=int, help="starting puzzle index in the split")
    parser.add_argument("--score-batch-size", default=128, type=int, help="forward batch size for edge embeddings")
    parser.add_argument("--postprocess", default=True, type=parse_bool, help="whether to apply the existing compatibility post-processing to E1 distances")
    parser.add_argument("--device", default="auto", help="execution device, e.g. auto, cpu, or cuda:0")
    parser.add_argument("--gpu-id", default=0, type=int, help="GPU index used when device=auto")
    parser.add_argument("--progress-interval", default=1, type=int, help="print metrics every N puzzles; 0 disables output")
    parser.add_argument("--verbose-solver", action="store_true", help="print Pomeranz restart and refinement summaries")
    parser.add_argument("--output-dir", default=None, type=Path, help="result directory; leave empty to generate automatically")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="whether to save puzzle visualizations")
    parser.add_argument("--visual-limit", default=10, type=int, help="maximum number of visualizations to save")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="whether to save predictions.npz")
    return add_pomeranz_arguments(parser)


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def default_output_dir(checkpoint: Path, args: argparse.Namespace) -> Path:
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{args.task}_{args.split}_"
        f"pomeranz_r{args.pomeranz_restarts}_m{args.pomeranz_max_refinement_rounds}_"
        f"s{args.pomeranz_random_seed}"
    )
    return DEFAULT_OUTPUT_ROOT / DEFAULT_DATASET_NAME / folder


def validate_args(args: argparse.Namespace, config: PomeranzSolverConfig) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index and max-samples must be non-negative")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size must be a positive integer")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit and progress-interval must be non-negative")
    grid = int(str(args.task).split("_", 1)[0].removeprefix("grid"))
    config.validate(grid * grid)


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    validate_args(args, config)
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    validate_checkpoint(args.checkpoint, args.task, args.loss_type)
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.checkpoint, args)
    args.output_dir = prepare_unique_output_dir(args.output_dir)

    device = resolve_device(args.device, args.gpu_id)
    scorer = MetricCompatibilityScorer.from_checkpoint(
        args.checkpoint,
        device=device,
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
    )
    solver = partial(solve_with_pomeranz, config=config)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"Pomeranz configuration: {asdict(config)}")
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
        method_name="E1 compatibility + Pomeranz CVPR 2011",
        solver=solver,
        configuration={
            "implementation": "paper_guided_python_reimplementation_with_s1a_e1",
            "postprocess": args.postprocess,
            **asdict(config),
        },
    )
    print(
        f"Result: PA={metrics.pa:.2%}, AA={metrics.aa:.2%}, "
        f"Horizontal SRA={metrics.horizontal_sra:.2%}, "
        f"Vertical SRA={metrics.vertical_sra:.2%}, SRA={metrics.sra:.2%}, "
        f"time={metrics.elapsed_seconds:.1f}s"
    )
    print(f"Saved: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
