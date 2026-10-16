"""
@file: evaluate_jpwleg_linear_programming.py
@description: Load the JPwLEG compatibility checkpoint and evaluate with the pure-Python BMVC 2016 Type-1 LP solver.
@author: Changxin Ye
@created: 2026-07-26
@version: 1.0
"""

from __future__ import annotations

from evaluation.common.solver_options import (
    DEFAULT_LP_BUDDY_CHECK,
    DEFAULT_LP_GREEDY_STOP_THRESHOLD,
    DEFAULT_LP_LOOP_CHECK,
    DEFAULT_LP_MATCH_TOLERANCE,
    DEFAULT_LP_MINIMUM_WEIGHT,
    DEFAULT_LP_PROBABILITY_LAMBDA,
    DEFAULT_LP_REFINEMENT_ITERATIONS,
    DEFAULT_LP_RIGID_COMPONENTS,
    DEFAULT_LP_TIME_LIMIT,
    DEFAULT_LP_TOP_K,
    add_lp_arguments,
    lp_config_from_args,
)

import argparse
from dataclasses import asdict
from functools import partial
from pathlib import Path

from initial_solvers.linear_programming import LPSolverConfig, solve_with_lp
from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
from data_loaders.jpwleg import DEFAULT_DATA_ROOT, JPWLEG_CONFIGS, SPLIT_ALIASES
from evaluation.jpwleg import evaluate_jpwleg_split
from compatibility.scorer import MetricCompatibilityScorer, resolve_device
from evaluation.common.jpwleg_inputs import (
    DEFAULT_CHECKPOINT_NAME,
    DEFAULT_RUN_DIR,
    parse_bool,
    resolve_checkpoint_arg,
)


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "jpwleg5"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_RUN_NAMES = {
    "jpwleg3": "hard_triplet_d128_s224_2026-07-27-08-14-53",
    "jpwleg5": "hard_triplet_d128_s224_2026-07-28-23-27-31",
}
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]

# Type-1 defaults from PuzzleParametersSet.m.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate JPwLEG learned compatibility with the pure-Python BMVC 2016 Type-1 LP solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=JPWLEG_CONFIGS.keys(), help="JPwLEG dataset to evaluate; each dataset uses its own checkpoint")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLIT_ALIASES.keys(), help="data split to evaluate")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path, help="MET_Dataset root")
    parser.add_argument("--checkpoint", default=None, type=Path, help="exact checkpoint path; highest priority")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="experiment directory under logs/train_jpwleg/<dataset>")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="directory containing checkpoints")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="checkpoint filename to load")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="maximum puzzles; 0 uses the full split")
    parser.add_argument("--start-index", default=0, type=int, help="starting puzzle index in the split")
    parser.add_argument("--score-batch-size", default=128, type=int, help="forward batch size for edge embeddings")
    parser.add_argument("--postprocess", default=False, type=parse_bool, help="whether to apply compatibility-score post-processing before LP; disabled by default")
    parser.add_argument("--device", default="auto", help="execution device, e.g. auto, cpu, or cuda:0")
    parser.add_argument("--gpu-id", default=0, type=int, help="GPU index used when device=auto")
    parser.add_argument("--progress-interval", default=25, type=int, help="print metrics every N puzzles; 0 disables output")
    parser.add_argument("--verbose-solver", action="store_true", help="show candidates, components, and objective values for each LP round")
    parser.add_argument("--output-dir", default=None, type=Path, help="result directory; leave empty to generate automatically")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="whether to save puzzle visualizations")
    parser.add_argument("--visual-limit", default=10, type=int, help="maximum number of visualizations to save")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="whether to save predictions.npz")
    return add_lp_arguments(parser)


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def default_output_dir(checkpoint: Path, args: argparse.Namespace, split: str) -> Path:
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{split}_"
        f"lpfaithful_k{args.lp_top_k}_r{args.lp_refinement_iterations}"
    )
    return DEFAULT_OUTPUT_ROOT / args.dataset / folder


def validate_args(args: argparse.Namespace, config: LPSolverConfig) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index and max-samples must be non-negative")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size must be a positive integer")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit and progress-interval must be non-negative")
    config.validate(JPWLEG_CONFIGS[args.dataset].num_pieces)


def main() -> None:
    args = parse_args()
    config = lp_config_from_args(args)
    validate_args(args, config)
    args.checkpoint = resolve_checkpoint_arg(args)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    split = SPLIT_ALIASES[args.split]
    if args.output_dir is None:
        args.output_dir = default_output_dir(args.checkpoint, args, split)
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
    metrics = evaluate_jpwleg_split(
        scorer=scorer,
        data_root=args.data_root,
        dataset=args.dataset,
        split=split,
        solver=solver,
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
        configuration={
            "implementation": "faithful_bmvc2016_type1_python_port",
            "postprocess": args.postprocess,
            **asdict(config),
        },
    )
    print(
        f"Result: PA={metrics.pa:.2%}, AA={metrics.aa:.2%}, HA={metrics.ha:.2%}, "
        f"VA={metrics.va:.2%}, SRA={metrics.sra:.2%}, NA={metrics.na:.2%}, "
        f"time={metrics.elapsed_seconds:.1f}s"
    )
    print(f"Saved: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
