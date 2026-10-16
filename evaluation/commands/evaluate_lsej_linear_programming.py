"""
@file: evaluate_lsej_linear_programming.py
@description: Load the compatibility checkpoint and evaluate ImageNet-LSEJ with the faithful pure-Python BMVC 2016 Type-1 port.
@author: Changxin Ye
@created: 2026-07-25
@version: 2.0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

try:
    from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from data_loaders.lsej import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from evaluation.lsej import evaluate_lsej_split
    from initial_solvers.linear_programming import LPSolverConfig, solve_with_lp
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
    from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from data_loaders.lsej import DEFAULT_LSEJ_DATA_ROOT, OFFICIAL_TASKS
    from evaluation.lsej import evaluate_lsej_split
    from initial_solvers.linear_programming import LPSolverConfig, solve_with_lp
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

# Type-1 defaults from PuzzleParametersSet.m.
DEFAULT_LP_TOP_K = 1
DEFAULT_LP_REFINEMENT_ITERATIONS = 5
DEFAULT_LP_PROBABILITY_LAMBDA = 5.0
DEFAULT_LP_MINIMUM_WEIGHT = 1e-15
DEFAULT_LP_MATCH_TOLERANCE = 1e-4
DEFAULT_LP_RIGID_COMPONENTS = True
DEFAULT_LP_BUDDY_CHECK = True
DEFAULT_LP_LOOP_CHECK = True
DEFAULT_LP_GREEDY_STOP_THRESHOLD = 1.25
DEFAULT_LP_TIME_LIMIT = 0.0


def add_lp_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add LP parameters corresponding one by one to the author's MATLAB Type-1 implementation."""

    parser.add_argument("--lp-top-k", default=DEFAULT_LP_TOP_K, type=int, help="candidate count kk retained per piece and direction")
    parser.add_argument("--lp-refinement-iterations", default=DEFAULT_LP_REFINEMENT_ITERATIONS, type=int, help="maximum rigid-component optimization rounds after the initial LP")
    parser.add_argument("--lp-probability-lambda", default=DEFAULT_LP_PROBABILITY_LAMBDA, type=float, help="lambda in the probability weight exp(-lambda * ratio^2)")
    parser.add_argument("--lp-minimum-weight", default=DEFAULT_LP_MINIMUM_WEIGHT, type=float, help="numeric cutoff t in the author's code; candidates at the cutoff are removed")
    parser.add_argument("--lp-match-tolerance", default=DEFAULT_LP_MATCH_TOLERANCE, type=float, help="absolute tolerance for LP adjacency equalities")
    parser.add_argument("--lp-rigid-components", default=DEFAULT_LP_RIGID_COMPONENTS, type=parse_bool, help="whether hard equality constraints preserve prior connected-component shapes")
    parser.add_argument("--lp-buddy-check", default=DEFAULT_LP_BUDDY_CHECK, type=parse_bool, help="whether to retain only candidates with reverse-mirror records")
    parser.add_argument("--lp-loop-check", default=DEFAULT_LP_LOOP_CHECK, type=parse_bool, help="whether to run the author's eight four-step cycle checks")
    parser.add_argument("--lp-greedy-stop-threshold", default=DEFAULT_LP_GREEDY_STOP_THRESHOLD, type=float, help="normalized stop threshold for component-level Gallagher merging")
    parser.add_argument("--lp-time-limit", default=DEFAULT_LP_TIME_LIMIT, type=float, help="time limit in seconds for each HiGHS solve; 0 means unlimited")
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate learned compatibility with the faithful pure-Python BMVC 2016 Type-1 LP solver",
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
    parser.add_argument("--postprocess", default=False, type=parse_bool, help="whether to apply compatibility-score post-processing before LP; disabled by default for faithful reproduction")
    parser.add_argument("--device", default="auto", help="execution device, e.g. auto, cpu, or cuda:0")
    parser.add_argument("--gpu-id", default=0, type=int, help="GPU index used when device=auto")
    parser.add_argument("--progress-interval", default=1, type=int, help="print metrics every N puzzles; 0 disables output")
    parser.add_argument("--verbose-solver", action="store_true", help="show candidates, components, and objective values for each LP round")
    parser.add_argument("--output-dir", default=None, type=Path, help="result directory; leave empty to generate automatically")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="whether to save puzzle visualizations")
    parser.add_argument("--visual-limit", default=10, type=int, help="maximum number of visualizations to save")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="whether to save predictions.npz")
    return add_lp_arguments(parser).parse_args()


def lp_config_from_args(args: argparse.Namespace) -> LPSolverConfig:
    return LPSolverConfig(
        top_k=args.lp_top_k,
        refinement_iterations=args.lp_refinement_iterations,
        probability_lambda=args.lp_probability_lambda,
        minimum_weight=args.lp_minimum_weight,
        match_tolerance=args.lp_match_tolerance,
        rigid_components=args.lp_rigid_components,
        buddy_check=args.lp_buddy_check,
        loop_check=args.lp_loop_check,
        greedy_stop_threshold=args.lp_greedy_stop_threshold,
        highs_time_limit=args.lp_time_limit,
    )


def default_output_dir(checkpoint: Path, args: argparse.Namespace) -> Path:
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{args.task}_{args.split}_"
        f"lpfaithful_k{args.lp_top_k}_r{args.lp_refinement_iterations}"
    )
    return DEFAULT_OUTPUT_ROOT / DEFAULT_DATASET_NAME / folder


def validate_args(args: argparse.Namespace, config: LPSolverConfig) -> None:
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
    config = lp_config_from_args(args)
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
    solver = partial(solve_with_lp, config=config)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"LP configuration: {asdict(config)}")
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
        method_name="Baseline (ViT-T + LP)",
        solver=solver,
        configuration={
            "implementation": "faithful_bmvc2016_type1_python_port",
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
