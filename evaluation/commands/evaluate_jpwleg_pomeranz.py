"""
@file: evaluate_jpwleg_pomeranz.py
@description: Evaluate JPwLEG with the unified learned compatibility distance and the pure-Python Pomeranz CVPR 2011 solver.
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
    from initial_solvers.pomeranz import solve_with_pomeranz
    from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from data_loaders.jpwleg import DEFAULT_DATA_ROOT, JPWLEG_CONFIGS, SPLIT_ALIASES
    from evaluation.jpwleg import evaluate_jpwleg_split
    from compatibility.scorer import MetricCompatibilityScorer, resolve_device
    from evaluation.commands.evaluate_jpwleg_gallagher import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_DIR, parse_bool, resolve_checkpoint_arg
    from evaluation.commands.evaluate_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args
except ImportError:
    from initial_solvers.pomeranz import solve_with_pomeranz
    from evaluation.output import checkpoint_run_name, prepare_unique_output_dir, script_output_root
    from data_loaders.jpwleg import DEFAULT_DATA_ROOT, JPWLEG_CONFIGS, SPLIT_ALIASES
    from evaluation.jpwleg import evaluate_jpwleg_split
    from compatibility.scorer import MetricCompatibilityScorer, resolve_device
    from evaluation.commands.evaluate_jpwleg_gallagher import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_DIR, parse_bool, resolve_checkpoint_arg
    from evaluation.commands.evaluate_lsej_pomeranz import add_pomeranz_arguments, pomeranz_config_from_args


DEFAULT_OUTPUT_ROOT = script_output_root(__file__, "eval_result")
DEFAULT_DATASET = "jpwleg3"
DEFAULT_SPLIT = "test"
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_RUN_NAMES = {
    "jpwleg3": "hard_triplet_d128_s224_2026-07-27-08-14-53",
    "jpwleg5": "hard_triplet_d128_s224_2026-07-28-23-27-31",
}
DEFAULT_RUN_NAME = DEFAULT_RUN_NAMES[DEFAULT_DATASET]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate JPwLEG with the learned compatibility distance and Pomeranz CVPR 2011 solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=JPWLEG_CONFIGS.keys(), help="JPwLEG dataset to evaluate")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLIT_ALIASES.keys(), help="data split to evaluate")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path, help="MET_Dataset root")
    parser.add_argument("--checkpoint", default=None, type=Path, help="exact checkpoint path; highest priority")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="experiment directory under logs/train_jpwleg/<dataset>")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="directory containing checkpoints")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="checkpoint filename to load")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="maximum puzzles to evaluate; 0 uses the full split")
    parser.add_argument("--start-index", default=0, type=int, help="starting puzzle index in the split")
    parser.add_argument("--score-batch-size", default=128, type=int, help="forward batch size for edge embeddings")
    parser.add_argument("--postprocess", default=True, type=parse_bool, help="whether to apply the existing compatibility post-processing to compatibility distances")
    parser.add_argument("--device", default="auto", help="execution device, e.g. auto, cpu, or cuda:0")
    parser.add_argument("--gpu-id", default=2, type=int, help="GPU index used when device=auto")
    parser.add_argument("--progress-interval", default=25, type=int, help="print metrics every N puzzles; 0 disables output")
    parser.add_argument("--verbose-solver", action="store_true", help="print Pomeranz restart and refinement summaries")
    parser.add_argument("--output-dir", default=None, type=Path, help="result directory; leave empty to generate automatically")
    parser.add_argument("--save-visuals", default=True, type=parse_bool, help="whether to save puzzle visualizations")
    parser.add_argument("--visual-limit", default=10, type=int, help="maximum number of visualizations to save")
    parser.add_argument("--save-predictions", default=True, type=parse_bool, help="whether to save predictions.npz")
    return add_pomeranz_arguments(parser)


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def default_output_dir(checkpoint: Path, args: argparse.Namespace, split: str) -> Path:
    folder = (
        f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}_{split}_"
        f"pomeranz_r{args.pomeranz_restarts}_m{args.pomeranz_max_refinement_rounds}_"
        f"s{args.pomeranz_random_seed}"
    )
    return DEFAULT_OUTPUT_ROOT / args.dataset / folder


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index and max-samples must be non-negative")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size must be a positive integer")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit and progress-interval must be non-negative")
    pomeranz_config_from_args(args).validate(JPWLEG_CONFIGS[args.dataset].num_pieces)


def main() -> None:
    args = parse_args()
    config = pomeranz_config_from_args(args)
    validate_args(args)
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
    solver = partial(solve_with_pomeranz, config=config)
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"Pomeranz configuration: {asdict(config)}")
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
        method_name="Baseline (ViT-T + Pomeranz)",
        configuration={
            "implementation": "paper_guided_python_reimplementation_with_s1a_e1",
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
