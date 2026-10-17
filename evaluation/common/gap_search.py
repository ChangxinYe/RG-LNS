"""Shared GAP search; importing this module never starts an evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from evaluation.output import checkpoint_run_name, prepare_unique_output_dir
from data_loaders.gap import DEFAULT_GAP_DATA_ROOT, GAP_CONFIGS, GAP_SPLITS
from evaluation.gap import DEFAULT_NEIGHBOR_RECALL_KS
from evaluation.lsej import NeighborRetrievalMetrics
from compatibility.scorer import MetricCompatibilityScorer, load_metric_checkpoint, resolve_device
from evaluation.common.gap_inputs import DEFAULT_LOG_ROOT, parse_bool


DEFAULT_DATASET = "GAP-3"
DEFAULT_SPLIT = "test"
DEFAULT_START_INDEX = 0
DEFAULT_MAX_SAMPLES = 0
DEFAULT_RUN_DIR = None
DEFAULT_CHECKPOINT_NAME = "best.pth"
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
METRIC_KEYS = ("PA", "AA", "HA", "VA", "SRA", "NA")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Evaluate greedy component reassembly on GAP-3/GAP-5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=GAP_CONFIGS.keys())
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=GAP_SPLITS)
    parser.add_argument("--data-root", default=DEFAULT_GAP_DATA_ROOT, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path, help="exact checkpoint path; highest priority")
    parser.add_argument(
        "--run-name",
        default=None,
        help="experiment name under checkpoints/train_gap/<dataset>; leave empty to use released weights or the latest training run",
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--start-index", default=DEFAULT_START_INDEX, type=int)
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int)
    parser.add_argument("--mutual-top-k", default=DEFAULT_MUTUAL_TOP_K, type=int)
    parser.add_argument("--component-index", default=DEFAULT_COMPONENT_INDEX, type=int)
    parser.add_argument("--min-component-size", default=DEFAULT_MIN_COMPONENT_SIZE, type=int)
    parser.add_argument("--refinement-rounds", default=DEFAULT_REFINEMENT_ROUNDS, type=int)
    parser.add_argument("--score-batch-size", default=DEFAULT_SCORE_BATCH_SIZE, type=int)
    parser.add_argument("--postprocess", default=DEFAULT_POSTPROCESS, type=parse_bool)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--save-visuals", default=DEFAULT_SAVE_VISUALS, type=parse_bool)
    parser.add_argument("--visual-limit", default=DEFAULT_VISUAL_LIMIT, type=int)
    parser.add_argument("--save-predictions", default=DEFAULT_SAVE_PREDICTIONS, type=parse_bool)
    parser.add_argument("--progress-interval", default=DEFAULT_PROGRESS_INTERVAL, type=int)
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


def _prepare_output_dir(output_dir: Path, *, automatically_named: bool) -> Path:
    """Avoid mixing repeated evaluations with different parameters in one result directory."""

    del automatically_named
    return prepare_unique_output_dir(output_dir)


def _default_output_dir(checkpoint: Path, dataset: str, *, output_root: Path) -> Path:
    # Keep only checkpoint identity in the directory name; write all evaluation parameters to summary.txt/metrics.json.
    folder = f"{checkpoint_run_name(checkpoint)}_{checkpoint.stem}"
    return Path(output_root) / dataset / folder


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_scorer(args: argparse.Namespace, checkpoint: Path):
    device = resolve_device(args.device, args.gpu_id)
    model, model_config, payload = load_metric_checkpoint(checkpoint, device)
    checkpoint_dataset = payload.get("dataset") or payload.get("args", {}).get("dataset")
    if checkpoint_dataset not in {None, args.dataset}:
        raise ValueError(
            f"Checkpoint dataset {checkpoint_dataset} does not match requested {args.dataset}"
        )
    if int(payload.get("model_config", {}).get("input_channels", 3)) != 4:
        raise ValueError("Selected checkpoint is not an RGBA GAP metric checkpoint")
    scorer = MetricCompatibilityScorer(
        model=model,
        device=device,
        input_size=model_config["input_size"],
        normalization=model_config["normalization"],
        batch_size=args.score_batch_size,
        postprocess=args.postprocess,
        score_metric=model_config["score_metric"],
        canonical_edge=model_config["canonical_edge"],
        geometry_mode=model_config["geometry_mode"],
    )
    return scorer, device, model_config


def _retrieval_metrics(
    rank_histogram: np.ndarray,
    horizontal_rank_histogram: np.ndarray,
    vertical_rank_histogram: np.ndarray,
    num_pieces: int,
) -> NeighborRetrievalMetrics:
    recall_ks = tuple(k for k in DEFAULT_NEIGHBOR_RECALL_KS if k <= num_pieces - 1)
    return NeighborRetrievalMetrics(
        recall_ks=recall_ks,
        rank_histogram=rank_histogram,
        horizontal_rank_histogram=horizontal_rank_histogram,
        vertical_rank_histogram=vertical_rank_histogram,
    )
