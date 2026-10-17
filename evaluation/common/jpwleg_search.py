"""Shared JPwLEG search; importing this module never starts an evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from data_loaders.jpwleg import DEFAULT_DATA_ROOT, SPLIT_ALIASES
from evaluation.common.jpwleg_inputs import DEFAULT_CHECKPOINT_NAME, DEFAULT_RUN_DIR, parse_bool


DEFAULT_DATASET = "jpwleg5"
DEFAULT_SPLIT = "test"
DEFAULT_START_INDEX = 0
DEFAULT_MAX_SAMPLES = 2000
DEFAULT_MUTUAL_TOP_K = 3
DEFAULT_COMPONENT_INDEX = 0
DEFAULT_MIN_COMPONENT_SIZE = 4
DEFAULT_REFINEMENT_ROUNDS = 1
DEFAULT_SCORE_BATCH_SIZE = 128
DEFAULT_POSTPROCESS = True
DEFAULT_SAVE_VISUALS = True
DEFAULT_VISUAL_LIMIT = 10
DEFAULT_SAVE_PREDICTIONS = True
DEFAULT_PROGRESS_INTERVAL = 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Run greedy high-confidence component reassembly on JPwLEG with frozen learned compatibility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=("jpwleg5",), help="JPwLEG-5 dataset to evaluate")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=SPLIT_ALIASES.keys(), help="data split to evaluate")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, type=Path, help="MET_Dataset root")
    parser.add_argument("--checkpoint", default=None, type=Path, help="exact checkpoint path; highest priority")
    parser.add_argument(
        "--run-name",
        default=None,
        help="experiment directory under logs/train_jpwleg/<dataset>; leave empty to select the dataset default",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="directory containing checkpoints")
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME, help="checkpoint filename to load")
    parser.add_argument("--start-index", default=DEFAULT_START_INDEX, type=int, help="starting puzzle index in the split")
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int, help="maximum puzzles to evaluate; 0 uses the full split")
    parser.add_argument("--mutual-top-k", default=DEFAULT_MUTUAL_TOP_K, type=int, help="candidates retained per direction when building mutual adjacency")
    parser.add_argument("--component-index", default=DEFAULT_COMPONENT_INDEX, type=int, help="cycle-supported component index after size sorting; 0 is the largest")
    parser.add_argument("--min-component-size", default=DEFAULT_MIN_COMPONENT_SIZE, type=int, help="minimum component size in pieces for reassembly")
    parser.add_argument("--refinement-rounds", default=DEFAULT_REFINEMENT_ROUNDS, type=int, help="maximum rounds of component extraction and translated reassembly")
    parser.add_argument("--score-batch-size", default=DEFAULT_SCORE_BATCH_SIZE, type=int, help="forward batch size for edge embeddings")
    parser.add_argument("--postprocess", default=DEFAULT_POSTPROCESS, type=parse_bool, help="whether to apply compatibility-score post-processing")
    parser.add_argument("--device", default="auto", help="execution device, e.g. auto, cpu, or cuda:0")
    parser.add_argument("--gpu-id", default=0, type=int, help="GPU index used when device=auto")
    parser.add_argument("--verbose-solver", action="store_true", help="show initial-solver internals")
    parser.add_argument("--output-dir", default=None, type=Path, help="result directory; leave empty to generate automatically")
    parser.add_argument("--save-visuals", default=DEFAULT_SAVE_VISUALS, type=parse_bool, help="whether to save before/after reassembly visualizations")
    parser.add_argument("--visual-limit", default=DEFAULT_VISUAL_LIMIT, type=int, help="maximum visualizations to save; 0 means unlimited")
    parser.add_argument("--save-predictions", default=DEFAULT_SAVE_PREDICTIONS, type=parse_bool, help="whether to save predictions.npz")
    parser.add_argument("--progress-interval", default=DEFAULT_PROGRESS_INTERVAL, type=int, help="print metrics every N puzzles; 0 disables output")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0 or args.max_samples < 0:
        raise ValueError("start-index and max-samples must be non-negative")
    if args.mutual_top_k <= 0:
        raise ValueError("mutual-top-k must be a positive integer")
    if args.component_index < 0:
        raise ValueError("component-index cannot be negative")
    if args.min_component_size <= 0 or args.refinement_rounds <= 0:
        raise ValueError("min-component-size and refinement-rounds must be positive integers")
    if args.score_batch_size <= 0:
        raise ValueError("score-batch-size must be a positive integer")
    if args.visual_limit < 0 or args.progress_interval < 0:
        raise ValueError("visual-limit and progress-interval must be non-negative")
